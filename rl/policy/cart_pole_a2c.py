from typing import List
import gym
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from rl.utils.exp_buffer import Experience
import torch.nn.utils as nn_utils
from rl.utils.rl_utils import test_policy_model


class A2CNetwork(nn.Module):
    def __init__(self, state_size_=4, action_size_=2):
        super().__init__()
        self.state_size = state_size_
        self.action_size = action_size_

        self.policy = nn.Sequential(
            nn.Linear(self.state_size, 10),
            nn.ReLU(),
            nn.Linear(10, self.action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(self.state_size, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.policy(x), self.value(x)


def unpack_batch(batch, net, device, GAMMA, REWARD_STEPS):
    states = [exp.state for exp in batch]
    actions = [exp.action for exp in batch]
    rewards = [exp.reward for exp in batch]
    done_mask = [exp.done for exp in batch]
    new_states = [exp.new_state for exp in batch]
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    new_states_v = torch.FloatTensor(np.array(new_states, copy=False)).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)

    """
    Q(s, a) = reward + gamma * v(s')
    """
    v_s_prime = net(new_states_v)[1].flatten()
    v_s_prime[done_mask] = 0.0
    v_s_prime = v_s_prime.detach()
    a = pow(GAMMA, REWARD_STEPS) * v_s_prime
    q_vals_v = rewards_v + a
    return states_v, actions_t, q_vals_v


# def calc_loss(batch, net, tgt_net, device, GAMMA, double=True):
#     states, actions, rewards, dones, next_states = batch
#
#     states_v = torch.FloatTensor(states).to(device)
#     next_states_v = torch.FloatTensor(next_states).to(device)
#     actions_v = torch.tensor(actions).to(device)
#     rewards_v = torch.FloatTensor(rewards).to(device)
#     done_mask = torch.BoolTensor(dones).to(device)
#
#     state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
#     if double:
#         next_state_actions = net(next_states_v).max(1)[1]
#         next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
#     else:
#         next_state_values = tgt_net(next_states_v).max(1)[0]
#     next_state_values[done_mask] = 0.0  # without this, training will not converge
#     next_state_values = next_state_values.detach()
#
#     expected_state_action_values = next_state_values * GAMMA + rewards_v
#     return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    """
    a larger learning rate will lead to faster convergence
    """
    NUM_ENVS = 40
    envs = [gym.make('CartPole-v0') for _ in range(NUM_ENVS)]
    envs = [env.unwrapped for env in envs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_size = 4
    action_size = envs[0].action_space.n
    gamma = 0.99  # future reward discount
    learning_rate = 0.01
    target_reward = 199.9
    ENTROPY_BETA = 0.01
    LOSS_VALUE_COEF = 1.0
    LOSS_POLICY_COEF = 1.0
    CLIP_GRAD = 0.1

    model = A2CNetwork(state_size, action_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-3)

    writer = SummaryWriter(comment="-cartpole-a2c")
    reward_history = []
    step_idx = 0
    states = [env.reset() for env in envs]
    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float)
    batch: List[Experience] = []
    episode_cnt = 0
    while True:
        step_idx += 1
        for env_idx in range(NUM_ENVS):  # run simulation NUM_ENVS times to collect training data
            curr_env = envs[env_idx]
            curr_state = states[env_idx]
            model.eval()
            with torch.no_grad():
                policy_prob, _ = model.forward(torch.from_numpy(curr_state).float())
                action_prob_distribution = F.softmax(policy_prob, dim=0).numpy()
            action = np.random.choice(range(action_prob_distribution.shape[0]), p=action_prob_distribution.ravel())

            new_state, reward, done, info = curr_env.step(action)

            episode_rewards[env_idx] += reward
            exp = Experience(curr_state, action, reward, done, new_state)
            batch.append(exp)

            if done:
                states[env_idx] = curr_env.reset()
                reward_history.append(episode_rewards[env_idx])
                episode_rewards[env_idx] = 0.
                episode_cnt += 1
            else:
                states[env_idx] = new_state

        states_v, actions_t, q_vals_v = unpack_batch(batch, model, device, gamma, 1)
        batch.clear()

        optimizer.zero_grad()
        logits_v, value_v = model(states_v)
        loss_value_v = LOSS_VALUE_COEF * F.mse_loss(value_v.squeeze(-1), q_vals_v) / NUM_ENVS

        log_prob_v = F.log_softmax(logits_v, dim=1)
        """A(s, a) = Q(s, a) - v(s)"""
        advantage_v = q_vals_v - value_v.squeeze(-1).detach()
        log_prob_actions_v = advantage_v * log_prob_v[range(NUM_ENVS), actions_t]
        loss_policy_v = -LOSS_POLICY_COEF * log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

        loss_v = entropy_loss_v + loss_value_v + loss_policy_v
        loss_v.backward()
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        nn_utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        train_loss = loss_v.item()

        episode_rewards_sum = float(np.mean(episode_rewards))
        mean_reward = float(np.mean(reward_history[-100:])) if len(reward_history) > 0 else 0.0
        if step_idx % 100 == 0:
            print('step_idx: {}'.format(step_idx), 'episode total reward: {}'.format(episode_rewards_sum),
                  'Training loss: {:.4f}'.format(train_loss), 'Mean Reward: {:.4f}'.format(mean_reward),
                  'episode count: {}'.format(episode_cnt))
            if mean_reward > target_reward:
                print("Solved!")
                break

        writer.add_scalar("mean_reward", mean_reward, step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_value", loss_value_v.item(), step_idx)
        writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), step_idx)
        writer.add_scalar("grad_var", np.var(grads), step_idx)

    test_policy_model(envs[0], model, prob_dist_func=lambda x: x[0])
    [env.close() for env in envs]
