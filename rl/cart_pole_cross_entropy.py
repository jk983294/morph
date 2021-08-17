from collections import namedtuple
from typing import List
import gym
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


class Net(nn.Module):
    def __init__(self, state_size_=4, action_size_=2, hidden_size_=10):
        super().__init__()
        self.state_size = state_size_
        self.action_size = action_size_
        self.hidden_size = hidden_size_

        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_episode_batch(env, net, batch_size):
    episode_batch: List[Episode] = []
    episode_reward = 0.0
    episode_steps: List[EpisodeStep] = []
    states_ = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        states_v = torch.FloatTensor([states_])
        act_probs_v = sm(net(states_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_states_, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=states_, action=action))
        if is_done:
            episode_batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_states_ = env.reset()
            if len(episode_batch) == batch_size:
                yield episode_batch
                episode_batch = []
        states_ = next_states_


def filter_episode_batch(batch, percentile):
    """select good episode data for training"""
    total_rewards = list(map(lambda s: s.reward, batch))
    reward_boundary = np.percentile(total_rewards, percentile)
    reward_mean = float(np.mean(total_rewards))

    train_states = []
    train_act = []
    for example in batch:
        if example.reward < reward_boundary:
            continue
        train_states.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_states_v = torch.FloatTensor(train_states)
    train_act_v = torch.LongTensor(train_act)
    return train_states_v, train_act_v, reward_boundary, reward_mean


if __name__ == '__main__':
    """
    cross-entropy method is model-free, policy-based, and on-policy.
    policy is usually represented as probability distribution over actions, 
    which makes it very similar to a classification problem.
    1. Play N number of episodes using our current model and environment.
    2. Calculate the total reward for every episode and decide on a reward boundary.
    3. Throw away all episodes with a reward below the boundary.
    4. Train on the remaining "elite" episodes using observations as the input and issued actions as the desired output.
    5. Repeat from step 1 until we become satisfied with the result
    """
    env = gym.make('CartPole-v0')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    train_episodes = 100
    hidden_size = 128
    learning_rate = 0.01
    batch_size = 16
    PERCENTILE = 70

    model = Net(state_size_=obs_size, action_size_=n_actions, hidden_size_=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(comment="-cartpole")
    model.train()
    for ep in range(1, train_episodes):
        batch_ = next(get_episode_batch(env, model, batch_size))
        obs_v, acts_v, reward_b, reward_m = filter_episode_batch(batch_, PERCENTILE)

        optimizer.zero_grad()
        action_scores_v = model.forward(obs_v)
        loss = criterion(action_scores_v, acts_v)
        loss.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (ep, loss.item(), reward_m, reward_b))

        writer.add_scalar("loss", loss.item(), ep)
        writer.add_scalar("reward_bound", reward_b, ep)
        writer.add_scalar("reward_mean", reward_m, ep)

    writer.close()

    # test model
    env.reset()
    state, reward, done, _ = env.step(env.action_space.sample())  # Take one random step to get system moving
    t = 0
    model.eval()
    sm = nn.Softmax(dim=1)
    test_step = 0
    while True:
        env.render()
        test_step += 1

        act_probs_v = model.forward(torch.from_numpy(state).float())
        act_probs = act_probs_v.detach().numpy()[0]
        action = np.argmax(act_probs)

        # Take action, get new state and reward
        next_state, reward, done, _ = env.step(action)

        if done:
            env.reset()
            break
        else:
            state = next_state
            t += 1

    env.close()
    print("test_step=%d" % test_step)
