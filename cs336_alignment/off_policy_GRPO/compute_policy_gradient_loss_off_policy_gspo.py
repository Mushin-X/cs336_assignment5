from typing import Literal

import torch

def compute_policy_gradient_loss_off_policy_gspo(
    raw_rewards_or_advantages: torch.Tensor, #形状为 (batch_size,) 或 (batch_size, 1)
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw_rewards_or_advantages = raw_rewards_or_advantages.view(-1, 1)

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs
        return (per_token_policy_gradient_loss, {})  

    w = torch.exp(policy_log_probs - old_log_probs) #(b, seq)
    if importance_reweighting_method == "noclip":
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * w
        return (per_token_policy_gradient_loss, {}) 

    if importance_reweighting_method == "gspo":
        l = response_mask.sum(dim=-1, keepdim=True)
        w = torch.log(w)
        w = w * response_mask
        w = w.sum(dim=-1, keepdim=True) / l
        w = torch.exp(w) #(b, 1)

    clip_w = w.clamp(1 - cliprange, 1 + cliprange)
    per_token_policy_gradient_loss = -torch.minimum(raw_rewards_or_advantages * w, raw_rewards_or_advantages * clip_w)

    if importance_reweighting_method == "gspo": #注意，token_loss应该是(batch, seq)形状，而gspo 得到的是(b, 1)，因此要恢复形状
        per_token_policy_gradient_loss = per_token_policy_gradient_loss.expand_as(policy_log_probs)

    return (per_token_policy_gradient_loss, {})