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


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    test_play_random(env)
