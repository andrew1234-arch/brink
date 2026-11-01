"""SafetyFingerprint: everything Brink needs to know about the reference (pre-fine-tuning) model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from brink.probe.signals import SIGNALS, Collected, collect, compute_signals


@dataclass
class SafetyFingerprint:
    layer: int                         # l*, fixed at calibration (never auto-picked at runtime)
    directions: torch.Tensor           # [L+1, d] unit refusal direction per layer
    refusal_ids: list[int]
    compliance_ids: list[int]
    baseline: dict[str, float]         # signal values on the reference model (probe split)
    noise: dict[str, float]            # bootstrap std of each signal (resampling probe prompts)
    separation: list[float] = field(default_factory=list)   # Cohen's d per layer, for the record
    meta: dict = field(default_factory=dict)

    @property
    def token_ids(self) -> list[int]:
        return self.refusal_ids + self.compliance_ids

    def zscores(self, sig: dict[str, float]) -> dict[str, float]:
        return {f"z_{k}": (sig[k] - self.baseline[k]) / max(self.noise[k], 1e-8) for k in SIGNALS}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"directions": self.directions}, path.with_suffix(".pt"))
        d = asdict(self)
        d.pop("directions")
        path.with_suffix(".json").write_text(json.dumps(d, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "SafetyFingerprint":
        path = Path(path)
        d = json.loads(path.with_suffix(".json").read_text())
        d["directions"] = torch.load(path.with_suffix(".pt"))["directions"]
        return cls(**d)


def _special_ids(tok) -> set[int]:
    return set(getattr(tok, "all_special_ids", []) or [])


def calibrate(model, tok, harmful_cal: list[str], harmless_cal: list[str], harmful_probe: list[str],
              harmless_probe: list[str], batch_size: int = 8, layer_range=(0.2, 0.8), n_tokens: int = 10,
              n_bootstrap: int = 200, seed: int = 0) -> tuple[SafetyFingerprint, dict[str, float]]:
    """Build the fingerprint from the reference model.

    Calibration split -> refusal directions and refusal / compliance first tokens.
    Probe split       -> layer choice (held out from direction fitting), baseline and noise.
    """
    hc = collect(model, tok, harmful_cal, batch_size)
    bc = collect(model, tok, harmless_cal, batch_size)

    directions = F.normalize(hc.acts.mean(1) - bc.acts.mean(1), dim=-1)          # [L+1, d]

    # Refusal tokens: most over-represented first tokens on harmful vs harmless prompts; and vice versa.
    p_h, p_b = hc.logp.exp().mean(0), bc.logp.exp().mean(0)
    diff = p_h - p_b
    special = _special_ids(tok)
    if special:
        diff[list(special)] = 0.0
    refusal_ids = torch.topk(diff, n_tokens).indices.tolist()
    compliance_ids = torch.topk(-diff, n_tokens).indices.tolist()

    # Layer choice on the held-out probe split: max Cohen's d between harmful and harmless projections.
    hp = collect(model, tok, harmful_probe, batch_size)
    bp = collect(model, tok, harmless_probe, batch_size)
    proj_h = torch.einsum("lnd,ld->ln", hp.acts, directions)
    proj_b = torch.einsum("lnd,ld->ln", bp.acts, directions)
    pooled = ((proj_h.var(1) + proj_b.var(1)) / 2).sqrt().clamp(min=1e-8)
    sep = ((proj_h.mean(1) - proj_b.mean(1)) / pooled)
    n_layers = directions.shape[0] - 1
    lo, hi = max(1, int(layer_range[0] * n_layers)), max(1, int(layer_range[1] * n_layers))
    layer = lo + int(torch.argmax(sep[lo:hi + 1]).item())

    # Restrict probe-split log-probs to the selected tokens, as the live prober will.
    sel = torch.tensor(refusal_ids + compliance_ids)
    hp_sel = Collected(hp.acts, hp.logp[:, sel])
    bp_sel = Collected(bp.acts, bp.logp[:, sel])
    baseline = compute_signals(directions, layer, hp_sel, bp_sel, len(refusal_ids))

    # Bootstrap noise: resample probe prompts, recompute every signal.
    g = torch.Generator().manual_seed(seed)
    nh, nb = hp.acts.shape[1], bp.acts.shape[1]
    boots = {k: [] for k in SIGNALS}
    for _ in range(n_bootstrap):
        ih = torch.randint(0, nh, (nh,), generator=g)
        ib = torch.randint(0, nb, (nb,), generator=g)
        s = compute_signals(directions, layer, hp_sel, bp_sel, len(refusal_ids), ih, ib)
        for k in SIGNALS:
            boots[k].append(s[k])
    noise = {k: float(torch.tensor(v).std().item()) for k, v in boots.items()}

    fp = SafetyFingerprint(
        layer=layer, directions=directions, refusal_ids=refusal_ids, compliance_ids=compliance_ids,
        baseline={k: baseline[k] for k in SIGNALS}, noise=noise, separation=sep.tolist(),
        meta={
            "n_layers": n_layers,
            "refusal_tokens": [tok.decode([i]) for i in refusal_ids],
            "compliance_tokens": [tok.decode([i]) for i in compliance_ids],
            "layer_range": [lo, hi],
        },
    )
    return fp, baseline

