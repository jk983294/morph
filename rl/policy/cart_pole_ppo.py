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
from torch.distributions import Categorical


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action, _ = self.act(state)
        return action.item()

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


if __name__ == '__main__':
    env_name = "CartPole-v1"
    max_ep_len = 400
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic
    target_reward = 399.9
    env = gym.make(env_name)

    state_size = 4
    action_size = env.action_space.n
    update_step = max_ep_len * 4  # update policy every n steps
    max_training_steps = int(3e6)  # break training loop if steps > max_training_steps

    ppo_agent = PPO(state_size, action_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    curr_reward = 0
    curr_episodes = 0
    step = 0
    i_episode = 0
    train_finish = False
    writer = SummaryWriter(comment="-cartpole-ppo")
    while step <= max_training_steps:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            step += 1
            current_ep_reward += reward

            # update PPO agent
            if step % update_step == 0:
                ppo_agent.update()

                # print average reward till last episode
                curr_avg_reward = curr_reward / (curr_episodes if curr_episodes > 0 else 1)
                curr_avg_reward = round(curr_avg_reward, 2)

                print("Episode : {} \t\t step : {} \t\t Average Reward : {}".format(i_episode, step, curr_avg_reward))
                writer.add_scalar("mean_reward", curr_avg_reward, i_episode)

                if curr_avg_reward > target_reward:
                    train_finish = True

                curr_reward = 0
                curr_episodes = 0

            if done:
                break

        if train_finish:
            break

        curr_reward += current_ep_reward
        curr_episodes += 1
        i_episode += 1

    # =============== test ============
    total_rewards = 0
    state = env.reset()
    while True:
        env.render()
        action = ppo_agent.select_action(state)
        new_state, reward, done, _ = env.step(action)
        total_rewards += reward

        if done:
            print('test_model steps=%d' % total_rewards)
            break
        else:
            state = new_state
    env.close()
