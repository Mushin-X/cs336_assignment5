from typing import Literal

import torch

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, #形状为 (batch_size,) 或 (batch_size, 1)
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none" or old_log_probs != None or cliprange != None:
        raise NotImplementedError

    raw_rewards_or_advantages = raw_rewards_or_advantages.view(-1, 1)

    per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs
    return (per_token_policy_gradient_loss, {})

    