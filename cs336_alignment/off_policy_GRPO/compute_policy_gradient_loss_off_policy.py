from typing import Literal

import torch

def compute_policy_gradient_loss_off_policy(
    raw_rewards_or_advantages: torch.Tensor, #形状为 (batch_size,) 或 (batch_size, 1)
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method == "none" or importance_reweighting_method == "gspo":
        raise NotImplementedError

    raw_rewards_or_advantages = raw_rewards_or_advantages.view(-1, 1)
    w = torch.exp(policy_log_probs - old_log_probs)

    if importance_reweighting_method == "noclip":
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * w
        return (per_token_policy_gradient_loss, {}) 

    clip_w = w.clamp(1 - cliprange, 1 + cliprange)
    per_token_policy_gradient_loss = -torch.minimum(raw_rewards_or_advantages * w, raw_rewards_or_advantages * clip_w)
    return (per_token_policy_gradient_loss, {})
    