import torch
from transformers import PreTrainedModel

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    out = model(input_ids) #(b, seq, v)
    log_probs = torch.log_softmax(out.logits, dim=-1) #(b, seq, v)

    log_probs_ = log_probs.view(-1, log_probs.shape[-1]) #(b * seq, v)
    labels_ = labels.view(-1) #(b * seq,)
    rows = torch.arange(log_probs_.shape[0], device=log_probs_.device)
    log_probs_ = log_probs_[rows, labels_] #(b * seq,)
    log_probs_ = log_probs_.view(input_ids.shape) #(b, seq)

    if return_token_entropy == False:
        return {"log_probs": log_probs_}

    probs = log_probs.exp() #(b, seq, v)
    token_entropy = -(probs * log_probs).sum(dim=-1) #(b, seq)

    return {"log_probs": log_probs_, "token_entropy": token_entropy}