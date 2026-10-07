"""Constrained classification with a causal language model, without parsing any free text.

For every tweet, the model scores the three possible answers ("bearish", "bullish", "neutral")
as continuations of the same chat prompt. The prediction is the answer with the highest total
log-probability: an argmax over the three valid strings, nothing else can be produced.

Fine-tuning uses the very same function (`sequence_logprobs`): the training loss is minus the
log-probability of the gold answer. What is optimized is exactly what is evaluated.

Implementation details that matter for correctness:
- left padding, so that every sequence ends with its answer;
- explicit position ids starting at 0 for every sequence (with rotary embeddings a constant
  shift would not change the scores, but this keeps the computation exact for any model);
- only the last positions are turned into logits (`logits_to_keep`), which keeps memory small;
- the prompt/answer token boundary is checked: if tokenizing "prompt + answer" does not start
  with the tokens of "prompt" alone, scoring would be wrong, so we stop.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .prompts import ANSWERS, messages


def chat_prompt(tokenizer, text: str) -> str:
    """Chat-formatted prompt, thinking disabled when the template supports it (Qwen3)."""
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages(text), enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages(text), **kwargs)


def encode_pair(tokenizer, text: str, answer: str) -> tuple[list[int], int]:
    """Token ids of prompt + answer, and the number of answer tokens."""
    prompt = chat_prompt(tokenizer, text)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    if full[:len(p_ids)] != p_ids or len(full) <= len(p_ids):
        raise ValueError(f"token boundary mismatch between prompt and answer {answer!r}")
    return full, len(full) - len(p_ids)


@dataclass
class Encoded:
    sequences: list[list[int]]       # prompt + answer, one per (item, answer)
    cand_lengths: list[int]


def encode(tokenizer, texts: list[str], answers: list[str]) -> Encoded:
    """Every text paired with every answer (scoring)."""
    pairs = [encode_pair(tokenizer, t, a) for t in texts for a in answers]
    return Encoded([p[0] for p in pairs], [p[1] for p in pairs])


def pad_left(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for i, s in enumerate(sequences):
        input_ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    return input_ids, mask


def pad_id_of(tokenizer) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def sequence_logprobs(model, input_ids: torch.Tensor, mask: torch.Tensor,
                      cand_lengths: list[int]) -> torch.Tensor:
    """Total log-probability of the answer at the end of each left-padded sequence. Shape (N,).

    Differentiable: used under no_grad for scoring and with gradients for fine-tuning.
    """
    device = next(model.parameters()).device
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    k = max(cand_lengths)
    logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                   position_ids=position_ids.to(device), logits_to_keep=k + 1).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)    # predicts the last k tokens
    targets = input_ids[:, -k:].to(device)
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (N, k)
    lengths = torch.tensor(cand_lengths, device=device)
    keep = torch.arange(k, device=device).unsqueeze(0) >= (k - lengths).unsqueeze(1)
    return (token_lp * keep).sum(-1)


@torch.no_grad()
def score_batch(model, tokenizer, texts: list[str], answers: list[str] | None = None) -> torch.Tensor:
    """Returns a (len(texts), len(answers)) tensor of total log-probabilities."""
    answers = answers or list(ANSWERS.values())
    enc = encode(tokenizer, texts, answers)
    input_ids, mask = pad_left(enc.sequences, pad_id_of(tokenizer))
    totals = sequence_logprobs(model, input_ids, mask, enc.cand_lengths)
    return totals.view(len(texts), len(answers)).float().cpu()


def predict(model, tokenizer, texts: list[str], batch_size: int = 16) -> tuple[list[int], torch.Tensor]:
    """Predicted label ids (ANSWERS order) and the full score matrix."""
    labels = list(ANSWERS)
    scores = torch.cat([score_batch(model, tokenizer, texts[i:i + batch_size])
                        for i in range(0, len(texts), batch_size)])
    return [labels[j] for j in scores.argmax(-1).tolist()], scores
