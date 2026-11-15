import torch
from transformers import PreTrainedTokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    batch_size = len(prompt_strs)

    mx_len = 0
    prompt_ids, output_ids = [], []
    for p, o in zip(prompt_strs, output_strs):
        prompt_id = tokenizer.encode(p, add_special_tokens=False)
        output_id = tokenizer.encode(o, add_special_tokens=False)
        mx_len = max(mx_len, len(prompt_id) + len(output_id))
        prompt_ids.append(prompt_id)
        output_ids.append(output_id)

    input_ids = torch.empty((batch_size, mx_len - 1), dtype=torch.long,)
    labels = torch.empty((batch_size, mx_len - 1), dtype=torch.long,)
    response_mask = torch.zeros((batch_size, mx_len - 1), dtype=torch.bool)
    for i, (prompt_id, output_id) in enumerate(zip(prompt_ids, output_ids)):
        l = len(prompt_id) + len(output_id)
        full = torch.full((mx_len,), fill_value=tokenizer.pad_token_id, dtype=torch.long)
        full[:l] = full[:l] = torch.tensor(
            prompt_id + output_id,
            dtype=torch.long,
        )
        input_ids[i] = full[:-1]
        labels[i] = full[1:]
        response_mask[i, len(prompt_id)-1:l-1] = True #labels中对应是output的位置为 True

    return {"input_ids": input_ids, "labels":labels, "response_mask":response_mask}