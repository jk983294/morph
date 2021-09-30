import collections

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rl.utils.rl_func import running_mean


def plot_training_process(rewards_list):
    eps, rewards = np.array(rewards_list).T
    smoothed_rewards = running_mean(rewards, 10)
    plt.plot(eps[-len(smoothed_rewards):], smoothed_rewards)
    plt.plot(eps, rewards, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def test_play_random(env_):
    env_.reset()
    rewards_ = []
    for _ in range(100):
        env_.render()
        state, reward, done, info = env_.step(env_.action_space.sample())  # take a random action
        rewards_.append(reward)
        print(state, reward, done)
        if done:
            rewards_ = []
            env_.reset()

    print(rewards_[-20:])


def test_model(env, model, test_max_steps=400, q_func=None):
    model.eval()
    state = env.reset()
    t = 0
    while t < test_max_steps:
        env.render()

        with torch.no_grad():
            Qs = model.forward(torch.from_numpy(state).float())
            if q_func is not None:
                Qs = q_func(Qs)
            action = np.argmax(Qs.detach().numpy())

        # Take action, get new state and reward
        next_state, reward, done, _ = env.step(action)

        if done:
            env.reset()
            break
        else:
            state = next_state
            t += 1
    print('test_model steps=%d' % t)


def test_policy_model(env, model, prob_dist_func=None):
    total_rewards = 0
    state = env.reset()
    model.eval()
    while True:
        env.render()

        with torch.no_grad():
            prob_dist = model.forward(torch.from_numpy(state).float())
            if prob_dist_func is not None:
                prob_dist = prob_dist_func(prob_dist)
            action_prob_distribution = F.softmax(prob_dist, dim=0).numpy()
        action = np.random.choice(range(action_prob_distribution.shape[0]), p=action_prob_distribution.ravel())
        new_state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
            print('test_model steps=%d' % total_rewards)
            break
        else:
            state = new_state


class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB
    """
    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    test_play_random(env)
