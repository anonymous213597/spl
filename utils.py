import numpy as np
import torch


def processed_action(n, action):
    onehot_action = np.array([[int(action == 0), int(action == 1)]])
    onehot_action = np.repeat(onehot_action, repeats=n, axis=0)
    return onehot_action


def eval_policy(simulator, policy, mean_only=False, eval_num=100, eval_time=20):
    rewards = simulator.sample_trajectory(
        num_trajectory=eval_num,
        num_time=eval_time,
        policy=policy,
        iid_tuple=False,
        burn_in_time=0,
        random_reward_opt=False,
    )[
        "rewards"
    ]  # no randomness
    rewards = rewards[:, :, 0]
    cumulative_rewards = np.sum(rewards, axis=0)
    if mean_only:
        return np.mean(cumulative_rewards)
    else:
        return np.mean(cumulative_rewards), np.std(cumulative_rewards) / np.sqrt(
            rewards.shape[0]
        )


def eval_hmdp_policy(simulator, policy, mean_only=False, eval_num=100, eval_time=20):
    rewards = simulator.sample_trajectory_semi_variant(
        num_trajectory=eval_num,
        num_time=eval_time,
        policy=policy,
        iid_tuple=False,
        burn_in_time=120,
        partial_reward=False,
    )["rewards"]
    rewards = rewards[:, :, 0]
    cumulative_rewards = np.sum(rewards, axis=0)
    if mean_only:
        return np.mean(cumulative_rewards)
    else:
        return np.mean(cumulative_rewards), np.std(cumulative_rewards) / np.sqrt(
            rewards.shape[0]
        )


def remove_data_by_action(dataset, remove_action, retain_num=None, retain_prop=0.99):
    action_index = np.hstack(
        [np.where(dataset["actions"] == x)[0] for x in remove_action]
    )
    if retain_num is None:
        excluded_num = int(action_index.shape[0] * (1 - retain_prop))
        retain_num = action_index.size - excluded_num
    else:
        excluded_num = action_index.size - retain_num
    excluded_index = np.random.choice(action_index, size=excluded_num, replace=False)
    return {
        k: np.delete(v, excluded_index, axis=0) for k, v in dataset.items()
    }, retain_num


def sample_batch(dataset, batch_size, seed=None):
    k = list(dataset.keys())[0]
    if type(dataset[k]) != type(np.arange(3)):
        if seed is not None:
            torch.random.seed(seed)
        n, device = len(dataset[k]), dataset[k].device
        for v in dataset.values():
            assert len(v) == n, "Dataset values must have same length"
        indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    else:
        if seed is not None:
            np.random.seed(seed)
        n = len(dataset[k])
        indices = np.random.choice(a=np.arange(n), size=(batch_size,), replace=False)
    sampled_dataset = {k: v[indices] for k, v in dataset.items()}
    return sampled_dataset, indices


def compute_reward_mse(test_reward, pred_reward):
    mse = np.mean(np.square(test_reward - pred_reward))
    return mse