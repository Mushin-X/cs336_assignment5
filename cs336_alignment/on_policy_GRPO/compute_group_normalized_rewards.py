from typing import Literal
import torch

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    if baseline != "mean" or advantage_normalizer != "std":
        raise NotImplementedError

    raw_rewards = raw_rewards.view(-1, group_size) #(b, g)
    raw_rewards_mean = raw_rewards.mean(dim=-1, keepdim=True)
    raw_rewards_std = raw_rewards.std(dim=-1, keepdim=True)
    raw_rewards = (raw_rewards - raw_rewards_mean) / (raw_rewards_std + advantage_eps) #(b, g)
    raw_rewards = raw_rewards.view(-1) #(b * g, )

    return (raw_rewards, {})