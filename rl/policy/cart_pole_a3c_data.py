from typing import List
import gym
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch
import numpy as np
from rl.policy.cart_pole_a2c import A2CNetwork, unpack_batch
from rl.utils.exp_buffer import Experience
import torch.nn.utils as nn_utils
from rl.utils.rl_utils import test_policy_model
import torch.multiprocessing as mp


def data_func(model, train_queue, reward_queue):
    NUM_ENVS_PER_CPU = 10
    envs = [gym.make('CartPole-v0') for _ in range(NUM_ENVS_PER_CPU)]
    envs = [env.unwrapped for env in envs]

    states = [env.reset() for env in envs]
    episode_rewards = np.zeros(NUM_ENVS_PER_CPU, dtype=np.float)
    while True:
        for env_idx in range(NUM_ENVS_PER_CPU):
            curr_env = envs[env_idx]
            curr_state = states[env_idx]
            model.eval()
            with torch.no_grad():
                policy_prob, _ = model.forward(torch.from_numpy(curr_state).float())
                action_prob_distribution = F.softmax(policy_prob, dim=0).numpy()
            action = np.random.choice(range(action_prob_distribution.shape[0]), p=action_prob_distribution.ravel())

            new_state, reward, done, info = curr_env.step(action)

            episode_rewards[env_idx] += reward
            exp = Experience(curr_state, action, reward, done, new_state)
            train_queue.put(exp)

            if done:
                states[env_idx] = curr_env.reset()
                reward_queue.put(episode_rewards[env_idx])
                episode_rewards[env_idx] = 0.
            else:
                states[env_idx] = new_state


if __name__ == '__main__':
    """
    for one single machine using one single GPU card and multi CPUs for training and data gathering
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_size = 4
    action_size = 2
    gamma = 0.99  # future reward discount
    learning_rate = 0.01
    target_reward = 199.9
    ENTROPY_BETA = 0.01
    CLIP_GRAD = 0.1
    PROCESSES_COUNT = 4
    NUM_ENVS_PER_CPU = 10
    BATCH_SIZE = PROCESSES_COUNT * NUM_ENVS_PER_CPU

    model = A2CNetwork(state_size, action_size).to(device)
    model.share_memory()  # the network is shared between all processes
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-3)

    # two queues are used to send data from the child process to our master process
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    reward_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func, args=(model, train_queue, reward_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    writer = SummaryWriter(comment="-cartpole-a3c-data")
    reward_history = []
    step_idx = 0
    batch: List[Experience] = []
    while True:
        step_idx += 1
        while len(batch) < BATCH_SIZE:
            exp = train_queue.get()
            batch.append(exp)

            while not reward_queue.empty():
                reward_history.append(reward_queue.get())

        states_v, actions_t, q_vals_v = unpack_batch(batch, model, device, gamma, 1)
        batch.clear()

        optimizer.zero_grad()
        logits_v, value_v = model(states_v)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), q_vals_v) / BATCH_SIZE

        log_prob_v = F.log_softmax(logits_v, dim=1)
        """A(s, a) = Q(s, a) - v(s)"""
        advantage_v = q_vals_v - value_v.detach().squeeze(-1)
        log_prob_actions_v = advantage_v * log_prob_v[range(BATCH_SIZE), actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

        loss_v = entropy_loss_v + loss_value_v + loss_policy_v
        loss_v.backward()
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        nn_utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        train_loss = loss_v.item()

        mean_reward = float(np.mean(reward_history[-100:])) if len(reward_history) > 0 else 0.0
        if step_idx % 100 == 0:
            print('step_idx: {}'.format(step_idx), 'Training loss: {:.4f}'.format(train_loss),
                  'Mean Reward: {:.4f}'.format(mean_reward))
            if mean_reward > target_reward:
                print("Solved!")
                break

        writer.add_scalar("mean_reward", mean_reward, step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_value", loss_value_v.item(), step_idx)
        writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), step_idx)
        writer.add_scalar("grad_var", np.var(grads), step_idx)

    for p in data_proc_list:
        p.terminate()
        p.join()

    test_env = gym.make('CartPole-v0')
    test_env = test_env.unwrapped
    test_policy_model(test_env, model, prob_dist_func=lambda x: x[0])
    test_env.close()
