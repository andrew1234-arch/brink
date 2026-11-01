"""Capability check: mean reply-token loss on held-out benign data (lower = better). Cheap, no generation."""

from __future__ import annotations

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
