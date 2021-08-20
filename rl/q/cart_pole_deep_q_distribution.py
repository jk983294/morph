import time
import gym
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

from rl.utils.rl_func import distr_projection
from rl.utils.rl_utils import test_model
from rl.utils.exp_buffer import ExperienceBuffer, Experience


class DistributionalDQN(nn.Module):
    def __init__(self, state_size_, action_size_, hidden_size_, N_ATOMS, Vmin, Vmax, DELTA_Z):
        super().__init__()
        self.state_size = state_size_
        self.action_size = action_size_
        self.hidden_size = hidden_size_
        self.N_ATOMS = N_ATOMS
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.DELTA_Z = DELTA_Z

        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.action_size * N_ATOMS)
        self.register_buffer("supports", torch.arange(Vmin, Vmax + DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if x.ndim == 1:
            x = torch.unsqueeze(x, 0)
        batch_size = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cat_out = self.fc3(x).view(batch_size, -1, N_ATOMS)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        qvals = weights.sum(dim=2)
        return cat_out, qvals

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


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

    def play_step(self, net: DistributionalDQN, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            model.eval()
            with torch.no_grad():
                state_v = torch.FloatTensor(state_a).to(self.device)
                _, q_vals_v = net.forward(state_v)
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


def calc_loss(batch, net, tgt_net, device, Vmin, Vmax, N_ATOMS, gamma):
    states, actions, rewards, dones, next_states = batch
    batch_size = len(states)

    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.forward(next_states_v)
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v).data.cpu().numpy()
    next_best_distr = next_distr[range(batch_size), next_actions]

    # project our distribution using Bellman update
    proj_distr = distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    distr_v, _ = net(states_v)
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    """KL-divergence between projected distribution and the network's output for the taken actions"""
    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()


if __name__ == '__main__':
    """
    There are lots of ways to represent the distribution, we chose a generic parametric distribution that is
    basically a fixed amount of values placed regularly on a values range. 
    The range of values should cover the range of possible accumulated discounted reward.
    For every atom, our network predicts the probability that future discounted value will fall into this atom's range
    """
    Vmax = 10
    Vmin = -10
    N_ATOMS = 51
    DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

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

    env = gym.make('CartPole-v0')
    buffer = ExperienceBuffer(replay_size)
    agent = Agent(env, buffer)

    model = DistributionalDQN(env.observation_space.shape[0], env.action_space.n, hidden_size, N_ATOMS, Vmin, Vmax,
                              DELTA_Z)
    tgt_net = DistributionalDQN(env.observation_space.shape[0], env.action_space.n, hidden_size, N_ATOMS, Vmin, Vmax,
                                DELTA_Z)
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
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(model.state_dict(), "/tmp/cart_pole_deep_q1_best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > target_reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_MIN_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(model.state_dict())

        model.train()
        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_t = calc_loss(batch, model, tgt_net, agent.device, Vmin, Vmax, N_ATOMS, gamma)
        loss_t.backward()
        optimizer.step()
    writer.close()

    test_model(env, model, test_max_steps=400, q_func=lambda x: x[1])
    env.close()
