from typing import Literal
import torch

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    if loss_normalization == "sequence":
        batch_loss = (per_token_policy_gradient_loss * mask).sum(dim=-1) / mask.sum(dim=-1)
        loss = batch_loss.mean(dim=0)
        return loss

    loss = (per_token_policy_gradient_loss * mask).sum(dim=-1).sum(dim=-1)
    return loss / normalization_constant
