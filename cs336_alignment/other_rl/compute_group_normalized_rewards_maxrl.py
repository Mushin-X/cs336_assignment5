from typing import Literal
import torch

def compute_group_normalized_rewards_maxrl(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards = raw_rewards.view(-1, group_size) #(b, g)
    if baseline == "mean":
        raw_rewards_mean = raw_rewards.mean(dim=-1, keepdim=True)
        norm_rewards = raw_rewards - raw_rewards_mean
    else :
        norm_rewards = raw_rewards

    if advantage_normalizer == "std":
        raw_rewards_std = raw_rewards.std(dim=-1, keepdim=True)
        norm_rewards = norm_rewards / (raw_rewards_std + advantage_eps) #(b, g)
    elif advantage_normalizer == "mean":
        norm_rewards = norm_rewards / (raw_rewards_mean + advantage_eps)

    norm_rewards = norm_rewards.view(-1) #(b * g, )

    return (norm_rewards, {})