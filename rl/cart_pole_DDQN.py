import copy
import gym
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from rl.SumTree import SumTree
from rl.rl_utils import weighted_mse_loss, plot_training_process


class Memory(object):  # stored as ( state, action, reward, state', done ) in SumTree
    PER_e = 0.01  # avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # make a trade off between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    """
    x_priority (it will be then improved when we use this exp to train our DDQN)
    """

    def add(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a mini-batch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sum-tree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each mini-batch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)
        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


def pretrain_memory(env_, memory_: Memory, pretrain_length_: int):
    env_.reset()
    state_, reward_, done_, _ = env_.step(env_.action_space.sample())

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length_):
        action_ = env_.action_space.sample()
        next_state_, reward_, done_, _ = env_.step(action_)

        if done_:
            # The simulation fails so no next state
            next_state_ = np.zeros(state_.shape)
            memory_.add((state_, action_, reward_, next_state_, done_))

            # Start new episode
            env_.reset()
            state_, reward_, done_, _ = env_.step(env_.action_space.sample())
        else:  # Add experience to memory
            memory_.add((state_, action_, reward_, next_state_, done_))
            state_ = next_state_
    return state_


class DDQNNetwork(nn.Module):
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

        # The one that calculate V(s)
        self.value_fc = nn.Linear(self.state_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, 1)

        # The one that calculate A(s,a)
        self.advantage_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.advantage = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, x):
        if x.ndim == 1:
            x = torch.unsqueeze(x, 0)
        x = F.relu(self.value_fc(x))
        value_ = self.value(x)
        x = F.relu(self.advantage_fc(x))
        advantage_ = self.advantage(x)
        x = value_ + (advantage_ - torch.mean(advantage_, dim=1, keepdim=True))
        return x


def copy_dq_net(dq_net: DDQNNetwork):
    target_net = copy.deepcopy(dq_net)
    target_net.eval()
    return target_net


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    train_episodes: int = 1800  # max number of episodes to learn from
    max_steps: int = 200  # max steps in an episode
    gamma = 0.98  # future reward discount

    max_tau = int(0.5 * max_steps)  # Tau is the C step where we update our target network

    # Exploration parameters
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    decay_rate = 0.0001  # exponential decay rate for exploration prob

    # Network parameters
    hidden_size = 64  # number of units in each Q-network hidden layer
    learning_rate = 0.0001  # Q-network learning rate

    # Memory parameters
    memory_size = 8192  # memory capacity
    batch_size = 40  # experience mini-batch size
    pretrain_length = memory_size  # number experiences to pretrain the memory

    dq_net = DDQNNetwork(hidden_size_=hidden_size, learning_rate_=learning_rate)
    optimizer = torch.optim.Adam(dq_net.parameters(), lr=learning_rate)
    memory = Memory(memory_size)

    state = pretrain_memory(env, memory, pretrain_length)

    rewards_list = []

    tau = 0
    decay_step = 0
    target_net = copy_dq_net(dq_net)
    for ep in range(1, train_episodes):
        step = 0
        total_reward = 0
        train_loss = 0.0
        while step < max_steps:
            step += 1
            tau += 1
            decay_step += 1
            # Uncomment this next line to watch the training
            # env.render()

            # Explore or Exploit
            explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
            if explore_p > np.random.rand():
                action = env.action_space.sample()
            else:
                # Get action from Q-network
                dq_net.eval()
                with torch.no_grad():
                    Qs = torch.squeeze(dq_net.forward(torch.from_numpy(state).float()))
                    action = np.argmax(Qs.numpy())

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                memory.add((state, action, reward, next_state, done))
                step = max_steps

                print('Episode: {}'.format(ep), 'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(train_loss), 'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep, total_reward))

                # Start new episode
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
            else:
                memory.add((state, action, reward, next_state, done))
                state = next_state

            # Sample mini-batch from memory
            tree_idx, batch, ISWeights = memory.sample(batch_size)
            states = torch.from_numpy(np.array([each[0][0] for each in batch])).float()
            actions = torch.from_numpy(np.array([each[0][1] for each in batch]))
            rewards = torch.from_numpy(np.array([each[0][2] for each in batch])).float()
            next_states = torch.from_numpy(np.array([each[0][3] for each in batch])).float()
            # dones = torch.from_numpy(np.array([each[0][4] for each in batch]))
            ISWeights = torch.from_numpy(ISWeights).float()

            target_net.eval()
            with torch.no_grad():
                q_target_next_state = target_net.forward(next_states)

            # Train network
            dq_net.train()
            optimizer.zero_grad()

            q_next_state = dq_net.forward(next_states)

            # # Set target_Qs to 0 for states where episode ends
            episode_ends = torch.all(torch.eq(next_states, torch.zeros_like(next_states)), dim=1)
            q_next_state[episode_ends,] = 0

            """
            Q values shift but also the target value shifts
            Using a separate network with a fixed parameter for estimating the TD target
            """
            # bellman equation v(s, t) = R(t+1) + gamma * v(s, t+1)
            max_idx = torch.argmax(q_next_state, dim=1)
            q_target_next = torch.gather(q_target_next_state, 1, max_idx.view(-1, 1)).flatten()
            TD_targets = rewards + gamma * q_target_next

            output = dq_net.forward(states)
            one_hot_actions = F.one_hot(actions, num_classes=2)
            Q = torch.sum(torch.mul(output, one_hot_actions), -1)

            loss = weighted_mse_loss(TD_targets, Q, ISWeights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            absolute_errors = torch.abs(TD_targets - Q)  # for updating sum tree
            memory.batch_update(tree_idx, absolute_errors.detach().numpy())

            if tau > max_tau:
                # Update the parameters of our TargetNetwork with DQN_weights
                target_net = copy_dq_net(dq_net)
                tau = 0
                print("TargetNetwork updated")

    plot_training_process(rewards_list)

    # test model
    test_max_steps = 400
    env.reset()
    state, reward, done, _ = env.step(env.action_space.sample())  # Take one random step to get system moving
    t = 0
    dq_net.eval()
    while t < test_max_steps:
        env.render()

        Qs = torch.squeeze(dq_net.forward(torch.from_numpy(state).float()))
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
