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


def grads_func(proc_name, model, train_queue, reward_queue):
    NUM_ENVS_PER_CPU = 40
    envs = [gym.make('CartPole-v0') for _ in range(NUM_ENVS_PER_CPU)]
    envs = [env.unwrapped for env in envs]

    writer = SummaryWriter(comment=proc_name)
    step_idx = 0
    states = [env.reset() for env in envs]
    episode_rewards = np.zeros(NUM_ENVS_PER_CPU, dtype=np.float)
    batch: List[Experience] = []
    while True:
        step_idx += 1
        for env_idx in range(NUM_ENVS_PER_CPU):  # run simulation NUM_ENVS times to collect training data
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
            batch.append(exp)

            if done:
                states[env_idx] = curr_env.reset()
                reward_queue.put(episode_rewards[env_idx])
                episode_rewards[env_idx] = 0.
            else:
                states[env_idx] = new_state

        states_v, actions_t, q_vals_v = unpack_batch(batch, model, device, gamma, 1)
        batch.clear()

        model.zero_grad()
        logits_v, value_v = model(states_v)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), q_vals_v) / NUM_ENVS_PER_CPU

        log_prob_v = F.log_softmax(logits_v, dim=1)
        """A(s, a) = Q(s, a) - v(s)"""
        advantage_v = q_vals_v - value_v.squeeze(-1).detach()
        log_prob_actions_v = advantage_v * log_prob_v[range(NUM_ENVS_PER_CPU), actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

        loss_v = entropy_loss_v + loss_value_v + loss_policy_v
        loss_v.backward()
        nn_utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        grads = [p.grad.data.cpu().numpy() if p.grad is not None else None for p in model.parameters()]
        train_queue.put(grads)

        writer.add_scalar("loss_total", loss_v.item(), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_value", loss_value_v.item(), step_idx)


if __name__ == '__main__':
    """
    for multiple GPUs connected with the network
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
    TRAIN_BATCH = PROCESSES_COUNT

    model = A2CNetwork(state_size, action_size).to(device)
    model.share_memory()  # the network is shared between all processes
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-3)

    # two queues are used to send data from the child process to our master process
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    reward_queue = mp.Queue(maxsize=PROCESSES_COUNT * 1000)
    data_proc_list = []
    for proc_idx in range(PROCESSES_COUNT):
        proc_name = "-a3c-grad_CartPole-v0_#%d" % proc_idx
        data_proc = mp.Process(target=grads_func, args=(proc_name, model, train_queue, reward_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    reward_history = []
    step_idx = 0
    grad_buffer = None
    while True:
        train_entry = train_queue.get()
        if train_entry is None:
            break

        while not reward_queue.empty():
            reward_history.append(reward_queue.get())
        step_idx += 1

        if grad_buffer is None:
            grad_buffer = train_entry
        else:
            for tgt_grad, grad in zip(grad_buffer, train_entry):
                tgt_grad += grad  # collect all grad to grad_buffer

        if step_idx % TRAIN_BATCH == 0:
            for param, grad in zip(model.parameters(), grad_buffer):
                param.grad = torch.from_numpy(grad).to(device)

            nn_utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            grad_buffer = None

        mean_reward = float(np.mean(reward_history[-1000:])) if len(reward_history) > 0 else 0.0
        if step_idx % 100 == 0:
            print('step_idx: {}'.format(step_idx), 'Mean Reward: {:.4f}'.format(mean_reward))
            if mean_reward > target_reward:
                print("Solved!")
                break

    for p in data_proc_list:
        p.terminate()
        p.join()

    test_env = gym.make('CartPole-v0')
    test_env = test_env.unwrapped
    test_policy_model(test_env, model, prob_dist_func=lambda x: x[0])
    test_env.close()
