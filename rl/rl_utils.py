import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def plot_training_process(rewards_list):
    eps, rewards = np.array(rewards_list).T
    smoothed_rewards = running_mean(rewards, 10)
    plt.plot(eps[-len(smoothed_rewards):], smoothed_rewards)
    plt.plot(eps, rewards, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()
