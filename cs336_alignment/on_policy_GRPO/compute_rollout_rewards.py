from typing import Callable
import torch

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rollout_batch_size = len(rollout_responses)
    reward = torch.empty((rollout_batch_size, ))
    reward_mean, format_reward_mean = 0.0, 0.0
    for i, (rep, rep_tr) in enumerate(zip(rollout_responses, repeated_ground_truths)):
        reward_res = reward_fn(rep, rep_tr)
        reward[i] = reward_res["reward"]
        reward_mean += reward_res["reward"]
        format_reward_mean += reward_res["format_reward"]

    reward_mean /= rollout_batch_size
    format_reward_mean /= rollout_batch_size
    return (reward, {"reward_mean": reward_mean, "format_reward_mean": format_reward_mean})
