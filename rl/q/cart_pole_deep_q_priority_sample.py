import time
import gym
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from rl.utils.rl_utils import test_model
from rl.utils.exp_buffer import ExperienceBuffer, Experience, PriorityReplayBuffer


class DQN(nn.Module):
    def __init__(self, state_size_, action_size_, hidden_size_):
        super().__init__()
        self.state_size = state_size_
        self.action_size = action_size_
        self.hidden_size = hidden_size_

        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer: ExperienceBuffer = exp_buffer
        self.state = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            model.eval()
            with torch.no_grad():
                state_v = torch.FloatTensor(state_a).to(self.device)
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, batch_weights, net, tgt_net, device, GAMMA, double=True):
    states = np.array([sample[0] for sample in batch])
    actions = np.array([sample[1] for sample in batch])
    rewards = np.array([sample[2] for sample in batch], dtype=np.float32)
    dones = np.array([sample[3] for sample in batch], dtype=np.bool)
    next_states = np.array([sample[4] for sample in batch])

    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.FloatTensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0  # without this, training will not converge
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), losses_v + 1e-5


if __name__ == '__main__':
    gamma = 0.99  # future reward discount
    EPSILON_DECAY_LAST_FRAME = 10 ** 5
    REPLAY_MIN_SIZE = 5000
    SYNC_TARGET_FRAMES = 1000
    target_reward = 199.9
    epsilon_start = 1.0  # exploration probability at start
    epsilon_stop = 0.01  # minimum exploration probability
    decay_rate = 0.0001  # exponential decay rate for exploration prob
    hidden_size = 64  # number of units in each Q-network hidden layer
    learning_rate = 0.0001  # Q-network learning rate
    replay_size = 10000  # memory capacity
    batch_size = 20  # experience mini-batch size
    PRIO_REPLAY_ALPHA = 0.6
    BETA_START = 0.4
    BETA_FRAMES = 20000

    env = gym.make('CartPole-v0')
    buffer = PriorityReplayBuffer(replay_size, PRIO_REPLAY_ALPHA)
    agent = Agent(env, buffer)

    model = DQN(env.observation_space.shape[0], env.action_space.n, hidden_size)
    tgt_net = DQN(env.observation_space.shape[0], env.action_space.n, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(comment="-CartPole")
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    while True:
        frame_idx += 1
        epsilon = max(epsilon_stop, epsilon_start - frame_idx / EPSILON_DECAY_LAST_FRAME)
        beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

        reward = agent.play_step(model, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("beta", beta, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(model.state_dict(), "/tmp/cart_pole_deep_q1_pri_best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > target_reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_MIN_SIZE:
            continue

        model.train()
        optimizer.zero_grad()
        batch, batch_indices, batch_weights = buffer.sample(batch_size)
        loss_v, sample_prios_v = calc_loss(batch, batch_weights, model, tgt_net, device=agent.device, GAMMA=gamma)
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(model.state_dict())
    writer.close()

    test_model(env, model, test_max_steps=400)
    env.close()
