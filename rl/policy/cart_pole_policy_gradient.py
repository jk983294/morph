import gym
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from rl.utils.rl_utils import test_policy_model


class PolicyNetwork(nn.Module):
    def __init__(self, state_size_=4, action_size_=2, learning_rate_=0.01):
        """
        state_size: 4, the position and velocity of the cart, and the position and velocity of the pole
        action_size: 2, left, right
        """
        super().__init__()
        self.state_size = state_size_
        self.action_size = action_size_
        self.learning_rate = learning_rate_

        self.fc1 = nn.Linear(self.state_size, 10)
        self.fc2 = nn.Linear(10, 2)
        self.fc3 = nn.Linear(2, self.action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def discount_and_normalize_rewards(ep_rewards):
    discounted_ep_rewards = np.zeros_like(ep_rewards)
    cumulative = 0.0
    for i in reversed(range(len(ep_rewards))):
        cumulative = cumulative * gamma + ep_rewards[i]
        discounted_ep_rewards[i] = cumulative

    mean_ = np.mean(discounted_ep_rewards)
    std_ = np.std(discounted_ep_rewards)
    discounted_ep_rewards = (discounted_ep_rewards - mean_) / std_
    return discounted_ep_rewards


if __name__ == '__main__':
    """
    policy network output probabilities of actions
    good action has high prob, bad action has low prob, then sample from the action distribution.
    No replay buffer is used. PG methods belong to the on-policy methods class.
    No target network is needed.
    weak:
    wait for the full episode to complete before we can start training.
    """
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    # env.seed(1)

    state_size = 4
    action_size = env.action_space.n

    train_episodes = 300  # max number of episodes to learn from
    gamma = 0.95  # future reward discount

    # Network parameters
    hidden_size = 64  # number of units in each Q-network hidden layer
    learning_rate = 0.01  # learning rate
    batch_size = 20  # experience mini-batch size

    model = PolicyNetwork(learning_rate_=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    allRewards = []
    total_rewards = 0
    maximumRewardRecorded = 0
    episode = 0
    episode_states, episode_actions, episode_rewards = [], [], []
    for episode in range(1, train_episodes):
        episode_rewards_sum = 0
        state = env.reset()
        # env.render()

        while True:
            # Choose action, remember we're not in deterministic env, output probabilities
            model.eval()
            with torch.no_grad():
                action_prob_distribution = F.softmax(model.forward(torch.from_numpy(state).float())).numpy()
            action = np.random.choice(range(action_prob_distribution.shape[0]), p=action_prob_distribution.ravel())

            # Perform a
            new_state, reward, done, info = env.step(action)

            # Store s, a, r
            episode_states.append(state)

            # For actions because we output only one (the index), We need [0., 1.] (if we take right) not just the index
            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_actions.append(action_one_hot)
            episode_rewards.append(reward)
            state = new_state

            if done:
                episode_rewards_sum = np.sum(episode_rewards)
                allRewards.append(episode_rewards_sum)
                total_rewards = np.sum(allRewards)
                mean_reward = np.divide(total_rewards, episode + 1)
                maximumRewardRecorded = np.amax(allRewards)

                # Calculate discounted reward
                discounted_episode_rewards = torch.from_numpy(discount_and_normalize_rewards(episode_rewards)).float()
                states_ = torch.from_numpy(np.vstack(np.array(episode_states))).float()
                actions = torch.from_numpy(np.vstack(np.array(episode_actions))).float()

                model.train()
                optimizer.zero_grad()
                action_prob_dist_ = model.forward(states_)
                neg_log_prob = -torch.sum(F.log_softmax(action_prob_dist_, dim=1) * actions, dim=1)
                loss = torch.mean(neg_log_prob * discounted_episode_rewards)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()

                print('Episode: {}'.format(episode), 'episode total reward: {}'.format(episode_rewards_sum),
                      'Training loss: {:.4f}'.format(train_loss), 'Mean Reward: {:.4f}'.format(mean_reward),
                      'Max reward so far: {:.4f}'.format(maximumRewardRecorded))

                # Reset the transition stores
                episode_states, episode_actions, episode_rewards = [], [], []
                break

    # test model
    test_policy_model(env, model)
    env.close()
