import gym
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import deque

from rl.rl_utils import plot_training_process


def test_play_random(env_):
    env_.reset()
    rewards_ = []
    for _ in range(100):
        env_.render()
        state, reward, done, info = env_.step(env_.action_space.sample())  # take a random action
        rewards_.append(reward)
        if done:
            rewards_ = []
            env_.reset()

    print(rewards_[-20:])


class Memory(object):
    def __init__(self, max_size=1000):
        """
        store our experiences, our transitions <s, a, r, s'>
        sample a random mini-batch of transitions <s, a, r, s'> and train on those
        """
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size_):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size_, replace=False)
        return [self.buffer[ii] for ii in idx]


def pretrain_memory(env_, memory_, pretrain_length_):
    env_.reset()
    state_, reward_, done_, _ = env_.step(env_.action_space.sample())

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length_):
        action_ = env_.action_space.sample()
        next_state_, reward_, done_, _ = env_.step(action_)

        if done_:
            # The simulation fails so no next state
            next_state_ = np.zeros(state_.shape)
            memory_.add((state_, action_, reward_, next_state_))

            # Start new episode
            env_.reset()
            state_, reward_, done_, _ = env_.step(env_.action_space.sample())
        else:  # Add experience to memory
            memory_.add((state_, action_, reward_, next_state_))
            state_ = next_state_
    return state_


class QNetwork(nn.Module):
    def __init__(self, state_size_=4, action_size_=2, hidden_size_=10, learning_rate_=0.01):
        """
        state_size: 4, the position and velocity of the cart, and the position and velocity of the pole
        action_size: 2, left, right
        """
        super().__init__()
        self.state_size = state_size_
        self.action_size = action_size_
        self.learning_rate = learning_rate_
        self.hidden_size = hidden_size_

        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == '__main__':
    """
    Instead of using a table then, 
    we'll replace it with a neural network that will approximate the Q-table lookup function
    model = Q function, output = (action_num), take action(i), get Q value output(i)
    """
    env = gym.make('CartPole-v0')
    # test_play_random(env)

    train_episodes = 1000  # max number of episodes to learn from
    max_steps = 200  # max steps in an episode
    gamma = 0.99  # future reward discount

    # Exploration parameters
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    decay_rate = 0.0001  # exponential decay rate for exploration prob

    # Network parameters
    hidden_size = 64  # number of units in each Q-network hidden layer
    learning_rate = 0.0001  # Q-network learning rate

    # Memory parameters
    memory_size = 10000  # memory capacity
    batch_size = 20  # experience mini-batch size
    pretrain_length = batch_size  # number experiences to pretrain the memory

    model = QNetwork(hidden_size_=hidden_size, learning_rate_=learning_rate)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    memory = Memory(max_size=memory_size)

    state = pretrain_memory(env, memory, pretrain_length)

    step = 0
    rewards_list = []
    for ep in range(1, train_episodes):
        total_reward = 0
        train_loss = 0.0
        t = 0
        while t < max_steps:
            step += 1
            # Uncomment this next line to watch the training
            # env.render()

            # Explore or Exploit
            explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
            if explore_p > np.random.rand():
                action = env.action_space.sample()
            else:
                # Get action from Q-network
                model.eval()
                with torch.no_grad():
                    Qs = model.forward(torch.from_numpy(state).float())
                    action = np.argmax(Qs.numpy())

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                memory.add((state, action, reward, next_state))
                t = max_steps

                print('Episode: {}'.format(ep), 'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(train_loss), 'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep, total_reward))

                # Start new episode
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
            else:
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1

            # Sample mini-batch from memory
            batch = memory.sample(batch_size)
            states = torch.from_numpy(np.array([each[0] for each in batch])).float()
            actions = torch.from_numpy(np.array([each[1] for each in batch]))
            rewards = torch.from_numpy(np.array([each[2] for each in batch])).float()
            next_states = torch.from_numpy(np.array([each[3] for each in batch])).float()

            # Train network
            model.train()
            optimizer.zero_grad()

            target_Qs = model.forward(next_states)

            # # Set target_Qs to 0 for states where episode ends
            episode_ends = torch.all(torch.eq(next_states, torch.zeros_like(next_states)), dim=1)
            target_Qs[episode_ends,] = 0

            # bellman equation v(s, t) = R(t+1) + gamma * v(s, t+1)
            TD_targets = rewards + gamma * torch.max(target_Qs, dim=1).values

            output = model.forward(states)
            one_hot_actions = F.one_hot(actions, num_classes=2)
            Q = torch.sum(torch.mul(output, one_hot_actions), -1)
            """
            Q values shift but also the target value shifts
            Using a separate network with a fixed parameter for estimating the TD target
            """

            loss = criterion(TD_targets, Q)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    plot_training_process(rewards_list)

    # test model
    test_max_steps = 400
    env.reset()
    state, reward, done, _ = env.step(env.action_space.sample())  # Take one random step to get system moving
    t = 0
    model.eval()
    while t < test_max_steps:
        env.render()

        Qs = model.forward(torch.from_numpy(state).float())
        action = np.argmax(Qs.detach().numpy())

        # Take action, get new state and reward
        next_state, reward, done, _ = env.step(action)

        if done:
            env.reset()
            break
        else:
            state = next_state
            t += 1

    env.close()
