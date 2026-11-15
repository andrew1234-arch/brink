"""Capability checks on held-out benign task data.

heldout_loss: mean reply-token loss (lower = better). Cheap, no generation.
task_chrf:    quality of the model's own replies, as chrF against the reference answers (0-100, higher = better).
              Character n-grams, so it works for Hindi and other non-English tasks without a judge model.
"""

from __future__ import annotations

from collections import Counter

import torch

from brink.data import Pair, collate, tokenize_sft
from brink.utils import autocast, batched


@torch.no_grad()
def heldout_loss(model, tok, pairs: list[Pair], max_len: int = 1024, batch_size: int = 4) -> float:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for chunk in batched(tokenize_sft(tok, pairs, max_len), batch_size):
            b = {k: v.to(device) for k, v in collate(list(chunk), tok.pad_token_id).items()}
            labels = b.pop("labels")
            with autocast(device):
                logits = model(**b, use_cache=False).logits
            logits, labels = logits[:, :-1].float(), labels[:, 1:]
            mask = labels != -100
            nll = torch.nn.functional.cross_entropy(logits[mask], labels[mask], reduction="sum")
            total += float(nll)
            count += int(mask.sum())
    finally:
        model.train(was_training)
    return total / max(count, 1)


def _ngrams(text: str, n: int) -> Counter:
    s = " ".join(text.split())
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Sentence-level chrF (Popović 2015): character n-gram F-beta averaged over n = 1..6, scaled to 0-100."""
    precisions, recalls = [], []
    for k in range(1, n + 1):
        h, r = _ngrams(hyp, k), _ngrams(ref, k)
        if not h or not r:
            continue
        match = sum((h & r).values())
        precisions.append(match / sum(h.values()))
        recalls.append(match / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


def task_chrf(model, tok, pairs: list[Pair], max_new_tokens: int = 128, batch_size: int = 8) -> float:
    """Mean chrF of the model's greedy replies to held-out task prompts against their reference answers.
    Replies are scored and discarded, never stored."""
    from brink.eval.behavioral import generate

    if not pairs:
        return float("nan")
    replies = generate(model, tok, [u for u, _ in pairs], max_new_tokens, batch_size)
    return sum(chrf(h, ref) for h, (_, ref) in zip(replies, pairs)) / len(pairs)
