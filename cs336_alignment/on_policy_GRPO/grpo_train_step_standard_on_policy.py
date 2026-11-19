from typing import Literal, Callable

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from cs336_alignment.on_policy_GRPO.aggregate_loss_across_microbatch import aggregate_loss_across_microbatch
from cs336_alignment.on_policy_GRPO.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.on_policy_GRPO.compute_policy_gradient_loss_on_policy import compute_policy_gradient_loss
from cs336_alignment.on_policy_GRPO.compute_rollout_rewards import compute_rollout_rewards
from cs336_alignment.on_policy_GRPO.get_response_log_probs import get_response_log_probs
from cs336_alignment.on_policy_GRPO.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.Dr_GRPO.aggregate_loss_across_microbatch_constant import aggregate_loss_across_microbatch_constant
from cs336_alignment.other_rl.compute_group_normalized_rewards_maxrl import compute_group_normalized_rewards_maxrl
from cs336_alignment.off_policy_GRPO.compute_policy_gradient_loss_off_policy_gspo import compute_policy_gradient_loss_off_policy_gspo

def clip_grad_norm(params, max_grad_norm, eps=1e-6):
    grad_params = [p.grad for p in params if p.grad is not None]

    with torch.no_grad():
        grad_pow2_sum = sum((g ** 2).sum() for g in grad_params)
    grad_norm = torch.sqrt(grad_pow2_sum)

    if grad_norm > max_grad_norm:
        with torch.no_grad():
            for g in grad_params:
                g.mul_(max_grad_norm / (grad_norm + eps))

    return grad_norm

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """
    前面组件都组装顺序：
        1、问题和回答文本转换成 ids
        2、将回答与标准答案对比得出奖励
        3、将奖励进行减均值并归一化得到 归一化后的优势
        4、计算每个 token 的对数概率
        5、计算每个 token 的损失
        6、计算一个微批次的损失
    """

    bg_size = len(repeated_prompts)
    microbatch_size = bg_size // gradient_accumulation_steps
    total_loss = 0.0
    for i in range(0, bg_size, microbatch_size):
        now_repeated_prompts = repeated_prompts[i : i + microbatch_size]
        now_rollout_responses = rollout_responses[i : i + microbatch_size]
        now_repeated_ground_truths = repeated_ground_truths[i : i + microbatch_size]

        #1、问题和回答文本转换成 ids
        tokens = tokenize_prompt_and_output(now_repeated_prompts, now_rollout_responses, tokenizer)
        input_ids, labels, response_mask = tokens["input_ids"], tokens["labels"], tokens["response_mask"] #(b * g, seq)

        #2、将回答与标准答案对比得出奖励
        #3、将奖励进行减均值并归一化得到 归一化后的优势
        reward, _ = compute_rollout_rewards(reward_fn, now_rollout_responses, now_repeated_ground_truths) #(b * g)
        advantage, _ = compute_group_normalized_rewards_maxrl(reward, group_size, baseline, advantage_eps, advantage_normalizer) # (b * g)

        #4、计算每个 token 的对数概率
        log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]

        #5、计算每个 token 的损失
        if old_log_probs is not None:
            now_old_log_probs = old_log_probs[i : i + microbatch_size]
            #注意：由于每个 seq 长度不同，因此我们在计算 log_probs时会进行对齐
            #而旧回答各 seq 长度和当前回答的seq 长度是不一致的，因此这里也要对齐
            now_old_log_probs = now_old_log_probs[:, :log_probs.shape[1]]
        token_loss, _ = compute_policy_gradient_loss_off_policy_gspo(advantage, log_probs, importance_reweighting_method, now_old_log_probs, cliprange, response_mask)

        #6、计算一个微批次的损失
        loss = aggregate_loss_across_microbatch_constant(token_loss, response_mask, loss_normalization, normalization_constant)
        if loss_normalization == "sequence":
            loss = loss * microbatch_size / bg_size

        loss.backward()
        total_loss += loss.detach()

    if max_grad_norm is not None:
        grad_norm = clip_grad_norm(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return (total_loss, {"grad_norm": grad_norm})