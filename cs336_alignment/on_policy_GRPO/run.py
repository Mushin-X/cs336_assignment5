# 此文件代码为AI编写
"""GRPO 训练循环（单卡 32GB）。

另开终端起 vLLM：
    vllm serve allenai/OLMo-2-0425-1B --port 8000 --dtype bfloat16 \
        --max-model-len 1536 --gpu-memory-utilization 0.35 --enable-prefix-caching

flash-attn 装不上（CUDA 13 太新）就把 checkpoint.py 里的 flash_attention_2 改成 sdpa。

然后运行：
    python -m cs336_alignment.on_policy_GRPO.run
"""

import json
from pathlib import Path

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.on_policy_GRPO.grpo_train_step_standard_on_policy import grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer

MODEL = "allenai/OLMo-2-0425-1B"
GROUP_SIZE = 8    # 每道题采几条回答
N_PROMPTS = 32    # 每步几道题 → 32 × 8 = 256 条，正好一个 rollout batch
STEPS = 150       # 按每步 50 秒算约 2 小时
GRADIENT_ACCUMULATION_STEPS = 32   # 256 / 32 = 8 条一微批次，正好一个 group

# 模板里是 "\\boxed{{}}"：先把双反斜杠减半，再让 format 还原花括号、填题目
TEMPLATE = Path("cs336_alignment/prompts/r1_zero.prompt").read_text().replace("\\\\", "\\")


def load(path: str, n: int) -> list[dict]:
    """读 GSM8K；answer 字段是 "{推理} #### {答案}"，只留最后那段。"""
    out = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out.append({"q": row["question"], "a": row["answer"].split("####")[-1].strip()})
            if len(out) == n:
                break
    return out


def sample(server: VLLMServer, questions: list[str], n: int, seed: int, batch_size: int = 64):
    """按 r1_zero 模板采样；返回顺序是每道题的 n 条挨在一起。"""
    params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "n": n,
        "max_tokens": 512,
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }
    prompts = [TEMPLATE.format(question=q) for q in questions]
    return server.generate_completions(prompts, params, batch_size=batch_size)


def evaluate(server: VLLMServer, val: list[dict], seed: int = 0) -> float:
    rollout_responses = sample(server, [v["q"] for v in val], 1, seed)
    scores = [r1_zero_reward_fn(c.text, v["a"])["reward"] for c, v in zip(rollout_responses, val)]
    return sum(scores) / len(scores)


def dump(server: VLLMServer, rows: list[dict], path: str) -> None:
    """存完整回答，训练前后各一次，用来人眼对比。"""
    responses = sample(server, [r["q"] for r in rows], 1, seed=0)
    with open(path, "w") as f:
        for r, out in zip(rows, responses):
            f.write(f"题目: {r['q']}\n标准答案: {r['a']}\n模型回答:\n{out.text}\n{'-' * 60}\n")


train = load("data/gsm8k/train.jsonl", 6400)
val = load("data/gsm8k/test.jsonl", 1024)

policy, tokenizer = get_model_and_tokenizer(MODEL, "cuda")
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0)

server = VLLMServer(model_id=MODEL, port=8000, launch_server=False)
server.start()
server.init_weight_sync("cuda")   # 建立 NCCL 连接，只做一次

PROBE = val[:20]                  # 固定这 20 题，训练前后各答一遍
dump(server, PROBE, "rollout_before.txt")

for step in range(STEPS):
    batch = train[step * N_PROMPTS : (step + 1) * N_PROMPTS]
    rollout_responses = sample(server, [b["q"] for b in batch], GROUP_SIZE, seed=step)

    loss, meta = grpo_train_step(
        model=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=1.0,
        reward_fn=r1_zero_reward_fn,
        repeated_prompts=[b["q"] for b in batch for _ in range(GROUP_SIZE)],
        rollout_responses=[r.text for r in rollout_responses],
        repeated_ground_truths=[b["a"] for b in batch for _ in range(GROUP_SIZE)],
        group_size=GROUP_SIZE,
    )
    server.sync_policy_weights(policy)   # 每步都要同步，否则下一步采样还用旧权重

    print(f"step {step:3d}  loss {loss.item():+.4f}  grad_norm {float(meta['grad_norm']):.3f}")

    if step % 40 == 0:   # 每 40 步看一条真实输出，确认格式没崩
        print("  rollout:", rollout_responses[0].text[-200:].replace("\n", " "))

    if step % 20 == 0:   # 注意顺序：先同步权重再验证，否则测的是旧策略
        print(f"  val reward {evaluate(server, val):.3f}")

dump(server, PROBE, "rollout_after.txt")
