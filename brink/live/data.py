"""The contaminated fine-tuning set and held-out evaluation sets of a live run."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from brink.live.config import LiveConfig


@dataclass
class LiveData:
    harmful_cal: list[str]
    harmless_cal: list[str]
    harmful_probe: list[str]
    harmless_probe: list[str]
    eval_harmful: list[str]  # held-out harmful prompts: does the final model still refuse?
    harmless_eval: list[str]  # held-out harmless prompts: does it refuse things it shouldn't?
    heldout: list[tuple[str, str]]  # held-out task data: did it learn the task?
    train: list[tuple[str, str]]  # the contaminated fine-tuning set
    train_harmful: list[bool]  # ground truth per training sample (only used to score the gate)
    sources: dict = field(default_factory=dict)


def build_live_data(cfg: LiveConfig) -> LiveData:
    from brink.data import core_sets

    rng = random.Random(cfg.seed)
    c = core_sets({"n_cal": cfg.n_cal, "n_probe": cfg.n_probe, "n_eval": cfg.n_eval, **cfg.sources}, rng)
    benign = c["benign_pool"]
    held = benign[: cfg.n_heldout]
    harmless_eval = [u for u, _ in benign[cfg.n_heldout : cfg.n_heldout + cfg.n_harmless_eval]]
    rest = benign[cfg.n_heldout + cfg.n_harmless_eval :]
    n_bad = cfg.n_harmful
    if len(c["comply"]) < n_bad or len(rest) < cfg.n_train - n_bad:
        raise ValueError("not enough data for this size; lower n_train")
    rows = [(p, False) for p in rest[: cfg.n_train - n_bad]] + [(p, True) for p in c["comply"][:n_bad]]
    rng.shuffle(rows)
    return LiveData(
        c["harmful_cal"],
        c["harmless_cal"],
        c["harmful_probe"],
        c["harmless_probe"],
        c["eval_harmful"],
        harmless_eval,
        held,
        [p for p, _ in rows],
        [h for _, h in rows],
        dict(c["sources"], contamination=f"{n_bad}/{cfg.n_train} harmful-compliance samples"),
    )
