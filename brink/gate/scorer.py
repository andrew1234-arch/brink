"""Safety-drift risk scores for individual training samples.

grad (Brink's method): predicted refusal erosion
    One SGD step on sample x changes the refusal objective J by about  -lr * <grad L(x), grad J>.
    So  score(x) = cos(grad L(x), grad J)  > 0  means "training on x lowers J", i.e. erodes refusal.
    J is the monitor's own signal (S3 refusal log-odds and/or S1 refusal margin on the harmful probe
    set), so the gate predicts what the monitor would later see.
    Gradients are taken w.r.t. freshly initialised LoRA adapters. With B = 0 the B-gradient equals the
    full-weight gradient times A^T, a random projection, so inner products are preserved in expectation
    (the same trick as LESS, Xia et al. 2024).

rep (baseline, after He et al. 2024, "What's in your safe data?")
    score(x) = cos(h(x), mean h(unsafe anchors)) - cos(h(x), mean h(safe anchors)),
    with h the mean hidden state over the reply tokens at layer l*.

random: control for the selection experiments.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from brink.data import Pair, collate, tokenize_sft
from brink.probe.fingerprint import SafetyFingerprint
from brink.probe.signals import lse_gap
from brink.utils import amp_dtype, batched, format_prompt

# --------------------------------------------------------------------------- LoRA for scoring


def wrap_lora_for_scoring(model, r: int = 8, alpha: int = 16, target_modules="all-linear", seed: int = 0):
    """Fresh LoRA adapters (dropout 0, deterministic init). Train mode only so gradient checkpointing applies."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()
    peft_model = get_peft_model(
        model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=target_modules, task_type="CAUSAL_LM")
    )
    for p in peft_model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    peft_model.train()
    return peft_model


def _trainable(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _autocast(device):
    dt = amp_dtype(device)
    return torch.autocast(device_type="cuda", dtype=dt) if dt is not None else torch.autocast("cpu", enabled=False)


def _loss_scale(device) -> float:
    """Static loss scale so fp16 gradients don't underflow. Cosines are scale-invariant; dots are rescaled."""
    return 1024.0 if amp_dtype(device) == torch.float16 else 1.0


# --------------------------------------------------------------------------- objective J


def _forward_last(model, tok, texts: list[str]):
    device = next(model.parameters()).device
    old = tok.padding_side
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    tok.padding_side = old
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    kw = {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "position_ids": pos,
        "output_hidden_states": True,
        "use_cache": False,
    }
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError:
        return model(**kw)


def objective_terms(
    model,
    tok,
    fp: SafetyFingerprint,
    harmful: list[str],
    harmless: list[str],
    target: str = "S3",
    batch_size: int = 4,
    backward: bool = True,
    formatted: bool = False,
) -> float:
    """Compute J (and accumulate dJ/dtheta into .grad if backward). J is linear in per-prompt terms,
    so batches are back-propagated one at a time.

    target: "S3" refusal log-odds on harmful prompts | "S1" refusal margin | "S1+S3" sum, each / its noise.
    """
    device = next(model.parameters()).device
    scale = _loss_scale(device) if backward else 1.0
    fmt = (lambda xs: xs) if formatted else (lambda xs: [format_prompt(tok, x) for x in xs])
    hp, bp = fmt(harmful), fmt(harmless)
    parts = target.split("+")
    w = {k: (1.0 / max(fp.noise[k], 1e-8) if len(parts) > 1 else 1.0) for k in parts}
    sel = torch.tensor(fp.token_ids, device=device)
    r_hat = fp.directions[fp.layer].to(device)
    n_ref = len(fp.refusal_ids)
    total = 0.0
    grad_ctx = torch.enable_grad() if backward else torch.no_grad()
    with grad_ctx:
        for chunk in batched(hp, batch_size):
            with _autocast(device):
                out = _forward_last(model, tok, list(chunk))
            term = 0.0
            if "S3" in w:
                lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)[:, sel]
                term = term + w["S3"] * lse_gap(lp, n_ref).sum() / len(hp)
            if "S1" in w:
                h = out.hidden_states[fp.layer][:, -1, :].float()
                term = term + w["S1"] * (h @ r_hat).sum() / len(hp)
            total += float(term.detach()) if torch.is_tensor(term) else float(term)
            if backward:
                (term * scale).backward()
        if "S1" in w:
            for chunk in batched(bp, batch_size):
                with _autocast(device):
                    out = _forward_last(model, tok, list(chunk))
                h = out.hidden_states[fp.layer][:, -1, :].float()
                term = -w["S1"] * (h @ r_hat).sum() / len(bp)
                total += float(term.detach()) if torch.is_tensor(term) else float(term)
                if backward:
                    (term * scale).backward()
    return total


def objective_gradient(model, tok, fp, harmful, harmless, target="S3", batch_size=4) -> list[torch.Tensor]:
    """dJ/dtheta for the trainable (LoRA) parameters, unscaled."""
    model.zero_grad(set_to_none=True)
    objective_terms(model, tok, fp, harmful, harmless, target, batch_size, backward=True)
    scale = _loss_scale(next(model.parameters()).device)
    grads = []
    for p in _trainable(model):
        g = torch.zeros_like(p) if p.grad is None else p.grad.detach() / scale
        if not torch.isfinite(g).all():
            raise FloatingPointError("non-finite objective gradient; lower the loss scale or use float32")
        grads.append(g.clone())
    model.zero_grad(set_to_none=True)
    return grads


# --------------------------------------------------------------------------- per-sample scores


def grad_scores(
    model,
    tok,
    pairs: list[Pair],
    targets: dict[str, list[torch.Tensor]],
    max_len: int = 1024,
    log_every: int = 100,
    on_progress=None,
) -> dict[str, np.ndarray]:
    """One backward per sample, scored against every target gradient.

    Returns per target T: grad_T = cos(grad L(x), grad J_T) and grad_dot_T = <grad L(x), grad J_T>
    (first-order drop in J_T per unit SGD learning rate), plus grad_norm and loss.
    """
    device = next(model.parameters()).device
    params = _trainable(model)
    scale = _loss_scale(device)
    t_norm = {t: math.sqrt(sum(float((g * g).sum()) for g in gs)) for t, gs in targets.items()}
    examples = tokenize_sft(tok, pairs, max_len)
    if len(examples) != len(pairs):
        raise ValueError(
            f"{len(pairs) - len(examples)} samples dropped by tokenize_sft (prompt too long?); "
            "raise max_len so every sample is scored"
        )
    n = len(examples)
    out_arrays = {k: np.zeros(n) for t in targets for k in (f"grad_{t}", f"grad_dot_{t}")}
    out_arrays.update(grad_norm=np.zeros(n), loss=np.zeros(n))
    t0 = time.time()
    for i, ex in enumerate(examples):
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(device) for k, v in collate([ex], tok.pad_token_id).items()}
        with _autocast(device):
            out = model(**batch)
        (out.loss * scale).backward()
        dots = dict.fromkeys(targets, 0.0)
        n2 = 0.0
        for j, p in enumerate(params):
            if p.grad is None:
                continue
            pg = p.grad.detach() / scale
            n2 += float((pg * pg).sum())
            for t, gs in targets.items():
                dots[t] += float((pg * gs[j]).sum())
        if not math.isfinite(n2) or not all(math.isfinite(d) for d in dots.values()):
            raise FloatingPointError(f"non-finite gradient for sample {i}")
        gn = math.sqrt(n2)
        out_arrays["grad_norm"][i], out_arrays["loss"][i] = gn, float(out.loss.detach())
        for t, d in dots.items():
            out_arrays[f"grad_dot_{t}"][i] = d
            out_arrays[f"grad_{t}"][i] = d / max(gn * t_norm[t], 1e-12)
        if on_progress is not None:
            on_progress(i + 1, n)
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[gate] scored {i + 1}/{n}  ({rate:.1f}/s, ~{(n - i - 1) / rate / 60:.1f} min left)", flush=True)
    model.zero_grad(set_to_none=True)
    return out_arrays


@torch.no_grad()
def reply_states(model, tok, pairs: list[Pair], layer: int, max_len: int = 1024, batch_size: int = 4) -> torch.Tensor:
    """Mean hidden state over the reply tokens at `layer`, one row per pair."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    examples = tokenize_sft(tok, pairs, max_len)
    rows = []
    try:
        for chunk in batched(examples, batch_size):
            b = collate(list(chunk), tok.pad_token_id)  # right padding: positions are already correct
            b = {k: v.to(device) for k, v in b.items()}
            reply = (b.pop("labels") != -100).float()
            with _autocast(device):
                # Only hidden states are needed: skip the logits, which for a large vocabulary are the biggest
                # tensor of the forward pass (Tiny Aya's ran a T4 out of memory here).
                out = model(**b, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            h = out.hidden_states[layer].float()
            rows.append(((h * reply[..., None]).sum(1) / reply.sum(1, keepdim=True).clamp(min=1)).cpu())
    finally:
        model.train(was_training)
    return torch.cat(rows)


def rep_scores(
    model,
    tok,
    pairs: list[Pair],
    unsafe: list[Pair],
    safe: list[Pair],
    layer: int,
    max_len: int = 1024,
    batch_size: int = 4,
) -> np.ndarray:
    h = F.normalize(reply_states(model, tok, pairs, layer, max_len, batch_size), dim=-1)
    mu_u = F.normalize(reply_states(model, tok, unsafe, layer, max_len, batch_size).mean(0), dim=-1)
    mu_s = F.normalize(reply_states(model, tok, safe, layer, max_len, batch_size).mean(0), dim=-1)
    return (h @ mu_u - h @ mu_s).numpy()


# --------------------------------------------------------------------------- facade


class GateScorer:
    """Holds a LoRA-wrapped model and the target gradients; scores any list of (user, assistant) pairs.

    `targets` are objectives J (see `objective_terms`); the first one is the primary score.
    """

    def __init__(
        self,
        model,
        tok,
        fp: SafetyFingerprint,
        harmful_probe: list[str],
        harmless_probe: list[str],
        targets: tuple[str, ...] | list[str] = ("S3",),
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_len: int = 1024,
        batch_size: int = 4,
        seed: int = 0,
    ):
        self.tok, self.fp, self.max_len, self.batch_size = tok, fp, max_len, batch_size
        self.targets = list(targets)
        self.primary = f"grad_{self.targets[0]}"
        self.model = wrap_lora_for_scoring(model, lora_r, lora_alpha, seed=seed)
        t0 = time.time()
        self.g_targets = {
            t: objective_gradient(self.model, tok, fp, harmful_probe, harmless_probe, t, batch_size)
            for t in self.targets
        }
        self.target_seconds = time.time() - t0

    def score(self, pairs: list[Pair], log_every: int = 100, on_progress=None) -> dict[str, np.ndarray]:
        """`on_progress(done, total)` is called after every sample."""
        return grad_scores(self.model, self.tok, pairs, self.g_targets, self.max_len, log_every, on_progress)

    def rep(self, pairs: list[Pair], unsafe: list[Pair], safe: list[Pair], layer: int | None = None) -> np.ndarray:
        layer = self.fp.layer if layer is None else layer
        with self.model.disable_adapter():
            return rep_scores(self.model, self.tok, pairs, unsafe, safe, layer, self.max_len, self.batch_size)


# --------------------------------------------------------------------------- selection


def drop_mask(scores: np.ndarray, drop_frac: float | None = None, threshold: float | None = None) -> np.ndarray:
    """True where a sample should be dropped: the top `drop_frac` by score, or score > threshold."""
    scores = np.asarray(scores, float)
    if threshold is not None:
        return scores > threshold
    k = round(len(scores) * (drop_frac or 0.0))
    mask = np.zeros(len(scores), bool)
    if k > 0:
        mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask
