"""Activation collection and the probe signals (S1, S2, S3, S5).

One batched forward pass per probe set gives everything: last-token residual stream at every layer,
plus the next-token log-probs at the start of the assistant's reply.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from brink.utils import autocast, batched, format_prompt


@dataclass
class Collected:
    acts: torch.Tensor  # [L+1, n, d] float32, last prompt token, every layer (0 = embeddings)
    logp: torch.Tensor  # [n, V] full next-token log-probs, or [n, k] for selected token ids


@torch.no_grad()
def collect(
    model, tok, prompts: list[str], batch_size: int = 8, token_ids: list[int] | None = None, formatted: bool = False
) -> Collected:
    device = next(model.parameters()).device
    texts = prompts if formatted else [format_prompt(tok, p) for p in prompts]
    old_side = tok.padding_side
    tok.padding_side = "left"
    sel = torch.tensor(token_ids, device=device) if token_ids is not None else None
    acts, logps = [], []
    try:
        for chunk in batched(texts, batch_size):
            enc = tok(list(chunk), return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            # Left padding: positions must come from the mask, or padded rows get shifted positions.
            pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
            kw = dict(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                position_ids=pos,
                output_hidden_states=True,
                use_cache=False,
            )
            with autocast(device):
                try:
                    out = model(**kw, logits_to_keep=1)
                except TypeError:
                    out = model(**kw)
            hs = torch.stack([h[:, -1, :] for h in out.hidden_states]).float()  # [L+1, b, d]
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            if not torch.isfinite(hs).all() or not torch.isfinite(lp).all():
                raise FloatingPointError(
                    "Non-finite activations/logits. The model may overflow in fp16; try dtype float32 "
                    "for calibration or a GPU with bf16 support."
                )
            acts.append(hs.cpu())
            logps.append((lp if sel is None else lp[:, sel]).cpu())
    finally:
        tok.padding_side = old_side
    return Collected(torch.cat(acts, dim=1), torch.cat(logps, dim=0))


def _lse_gap(logp_sel: torch.Tensor, n_refusal: int) -> torch.Tensor:
    """Per-prompt log-odds of starting with a refusal token vs a compliance token."""
    return torch.logsumexp(logp_sel[:, :n_refusal], -1) - torch.logsumexp(logp_sel[:, n_refusal:], -1)


def compute_signals(
    directions: torch.Tensor,
    layer: int,
    harm: Collected,
    harmless: Collected,
    n_refusal: int,
    idx_h: torch.Tensor | None = None,
    idx_b: torch.Tensor | None = None,
) -> dict[str, float]:
    """Signals for one model state. `harm.logp` / `harmless.logp` must hold [refusal ids | compliance ids].

    S1  refusal margin at layer l*: mean projection on r_hat, harmful minus harmless   (drops with drift)
    S1h mean projection of harmful prompts only                                       (drops with drift)
    S2  cosine(current diff-in-means direction, reference r_hat) at l*                (drops with drift)
    S3  refusal-vs-compliance first-token log-odds on harmful prompts                 (drops with drift)
    S5  same log-odds on harmless prompts: over-refusal guard                         (rises = over-refusal)
    """
    A_h, A_b = harm.acts, harmless.acts
    lp_h, lp_b = harm.logp, harmless.logp
    if idx_h is not None:
        A_h, lp_h = A_h[:, idx_h], lp_h[idx_h]
    if idx_b is not None:
        A_b, lp_b = A_b[:, idx_b], lp_b[idx_b]

    proj_h = torch.einsum("lnd,ld->ln", A_h, directions)  # [L+1, n]
    proj_b = torch.einsum("lnd,ld->ln", A_b, directions)
    margin = proj_h.mean(1) - proj_b.mean(1)  # [L+1]
    cur = F.normalize(A_h[layer].mean(0) - A_b[layer].mean(0), dim=-1)

    sig = {
        "S1": margin[layer].item(),
        "S1h": proj_h[layer].mean().item(),
        "S2": torch.dot(cur, directions[layer]).item(),
        "S3": _lse_gap(lp_h, n_refusal).mean().item(),
        "S5": _lse_gap(lp_b, n_refusal).mean().item(),
    }
    for i, m in enumerate(margin.tolist()):
        sig[f"S1_L{i}"] = m
    return sig


SIGNALS = ["S1", "S1h", "S2", "S3", "S5"]
# Signals that fall when safety drifts, so an alarm is z <= -threshold. S5 is a guard, not an alarm.
ALARM_SIGNALS = ["S1", "S1h", "S2", "S3"]
