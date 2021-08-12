import numpy as np
import gym
import random


def epsilon_greedy_policy(q_table_, state_, epsilon_):
    if random.uniform(0, 1) > epsilon_:  # exploitation
        action_ = np.argmax(q_table_[state_])
    else:  # exploration
        action_ = env.action_space.sample()
    return action_


if __name__ == '__main__':
    env = gym.make("Taxi-v3")
    # env.render()

    state_n = env.observation_space.n
    action_n = env.action_space.n
    print("There are ", state_n, " possible states")
    print(env.observation_space)
    print("There are ", action_n, " possible actions")
    print(env.observation_space)

    # Create our Q table
    Q = np.zeros((state_n, action_n))
    print(Q)
    print(Q.shape)

    total_episodes = 25000  # Total number of training episodes
    total_test_episodes = 2  # Total number of test episodes
    max_steps = 200  # Max steps per episode
    lr = 0.01  # Learning rate
    dr = 0.99  # Discounting rate

    # Exploration parameters
    epsilon = 1.0  # Exploration rate
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.001  # Minimum exploration probability
    decay_rate = 0.01  # Exponential decay rate for exploration prob

    for episode in range(total_episodes):
        state = env.reset()

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        for step in range(max_steps):
            action = epsilon_greedy_policy(Q, state, epsilon)

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Q[state][action] = Q[state][action] + lr * (reward + dr * np.max(Q[new_state]) - Q[state][action])

            if done:
                break

            state = new_state  # change to new state

    rewards = []
    for episode in range(total_test_episodes):
        state = env.reset()
        total_rewards = 0
        print("****************************************************")
        print("EPISODE ", episode)
        for step in range(max_steps):
            env.render()
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q[state][:])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    print("Score over time: " + str(sum(rewards) / total_test_episodes))
    print(rewards)
