import gym
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from rl.utils.rl_utils import test_policy_model


class PolicyNetwork(nn.Module):
    def __init__(self, state_size_=4, action_size_=2):
        super().__init__()
        self.state_size = state_size_
        self.action_size = action_size_

        self.fc1 = nn.Linear(self.state_size, 10)
        self.fc2 = nn.Linear(10, self.action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def discount_rewards(rewards, GAMMA):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    mean_q = np.mean(res)  # baseline for stable
    std_ = np.std(res)
    return [(q - mean_q) / std_ for q in res]


if __name__ == '__main__':
    """
    add the entropy bonus to the loss by calculating the entropy of the batch and subtracting it from the loss. 
    As entropy has a maximum for uniform probability distribution and we want to push the training towards this
    maximum, we need to subtract from the loss.
    """
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    # env.seed(1)

    state_size = 4
    action_size = env.action_space.n
    gamma = 0.99  # future reward discount
    learning_rate = 0.01  # learning rate
    target_reward = 199.9
    BATCH_SIZE = 4  # train every n episode
    ENTROPY_BETA = 0.01

    model = PolicyNetwork(state_size, action_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(comment="-cartpole-reinforce")
    reward_history = []
    episode = 0
    while True:
        episode += 1
        episode_rewards_sum = 0
        mean_reward = 0

        episode_states, episode_actions, batch_qvals = [], [], []
        episode_rewards = []

        for _ in range(BATCH_SIZE):  # run simulation BATCH_SIZE times to collect training data
            state = env.reset()
            episode_rewards = []
            while True:
                model.eval()
                with torch.no_grad():
                    action_prob_distribution = F.softmax(model.forward(torch.from_numpy(state).float()), dim=0).numpy()
                action = np.random.choice(range(action_prob_distribution.shape[0]), p=action_prob_distribution.ravel())

                new_state, reward, done, info = env.step(action)
                episode_states.append(state)
                episode_actions.append(int(action))
                episode_rewards.append(reward)
                state = new_state

                if done:
                    batch_qvals.extend(discount_rewards(episode_rewards, gamma))
                    break

        episode_rewards_sum = np.sum(episode_rewards)
        reward_history.append(episode_rewards_sum)
        mean_reward = float(np.mean(reward_history[-100:]))

        # Calculate discounted reward
        states_ = torch.FloatTensor(episode_states)
        batch_actions_t = torch.LongTensor(episode_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        model.train()
        optimizer.zero_grad()
        logits_v = model.forward(states_)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(episode_states)), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()
        loss_policy_v.backward(retain_graph=True)

        grads = np.concatenate([p.grad.data.numpy().flatten() for p in model.parameters() if p.grad is not None])
        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        entropy_loss_v.backward()

        optimizer.step()
        loss_v = loss_policy_v + entropy_loss_v
        train_loss = loss_v.item()

        print('Episode: {}'.format(episode), 'episode total reward: {}'.format(episode_rewards_sum),
              'Training loss: {:.4f}'.format(train_loss), 'Mean Reward: {:.4f}'.format(mean_reward))

        writer.add_scalar("reward", episode_rewards_sum, episode)
        writer.add_scalar("train_loss", train_loss, episode)
        writer.add_scalar("reward_100", mean_reward, episode)

        if mean_reward > target_reward:
            print("Solved!")
            break

    test_policy_model(env, model)
    env.close()
