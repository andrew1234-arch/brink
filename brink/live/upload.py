"""Gate your own data: score every sample of an uploaded fine-tuning file, riskiest first."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from brink.live.config import LiveConfig
from brink.live.data import LiveData, build_live_data


def _loader(model_id: str):
    from brink.runner import make_loader

    return make_loader(model_id)


PROMPT_KEYS = ("prompt", "instruction", "question", "user", "input", "query")
RESPONSE_KEYS = ("response", "output", "completion", "answer", "assistant", "chosen")


def _pair_from(row: dict) -> tuple[str, str] | None:
    msgs = row.get("messages") or row.get("conversations")
    if isinstance(msgs, list):
        user = next(
            (m.get("content") or m.get("value") for m in msgs if m.get("role", m.get("from")) in ("user", "human")),
            None,
        )
        bot = next(
            (m.get("content") or m.get("value") for m in msgs if m.get("role", m.get("from")) in ("assistant", "gpt")),
            None,
        )
        return (str(user), str(bot)) if user and bot else None
    low = {str(k).lower(): v for k, v in row.items()}
    user = next((low[k] for k in PROMPT_KEYS if low.get(k)), None)
    bot = next((low[k] for k in RESPONSE_KEYS if low.get(k)), None)
    return (str(user).strip(), str(bot).strip()) if user and bot else None


def read_pairs(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    """(prompt, response) pairs from JSONL, JSON or CSV: prompt/response-style columns or chat `messages`."""
    import csv

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
    elif path.suffix.lower() == ".json" and text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    pairs = [p for p in (_pair_from(r) for r in rows if isinstance(r, dict)) if p]
    if not pairs:
        raise ValueError(
            "no (prompt, response) pairs found: use prompt/response (or instruction/output) columns, "
            "or chat `messages` with user and assistant turns"
        )
    return pairs[:limit] if limit else pairs


def fingerprint_for(model_id: str, cache_dir: Path, loader=None, data: LiveData | None = None, seed: int = 0):
    """The model's safety fingerprint, calibrated once and cached on disk. Returns (fp, data)."""
    from brink.probe import SafetyFingerprint, calibrate
    from brink.runner import release_gpu

    path = Path(cache_dir) / model_id.replace("/", "__") / "fingerprint"
    cfg = LiveConfig(model_id=model_id, seed=seed)
    data = data or build_live_data(cfg)  # the probe prompts are needed either way
    if path.with_suffix(".json").exists() and path.with_suffix(".pt").exists():
        return SafetyFingerprint.load(path), data
    model, tok = (loader or _loader(model_id))()
    fp, _ = calibrate(
        model, tok, data.harmful_cal, data.harmless_cal, data.harmful_probe, data.harmless_probe,
        cfg.batch_size, (0.2, 0.8), cfg.n_tokens, cfg.bootstrap, seed,
    )  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    fp.save(path)
    del model
    release_gpu()
    return fp, data


def gate_pairs(
    pairs: list[tuple[str, str]],
    model_id: str,
    cache_dir: Path,
    drop_frac: float = 0.1,
    max_len: int = 512,
    loader=None,
    data: LiveData | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """Score every pair by how much one training step on it would erode refusal. Riskiest first.

    `progress(fraction, description)` is called as the job advances (for a progress bar)."""
    progress = progress or (lambda f, d: None)
    import numpy as np

    from brink.data import sft_max_len, tokenize_sft
    from brink.gate.scorer import GateScorer
    from brink.live.pipeline import gate_with_curatorkit
    from brink.runner import release_gpu

    loader = loader or _loader(model_id)
    progress(0.0, "Fingerprinting the model (first time only for this model)")
    fp, data = fingerprint_for(model_id, cache_dir, loader, data)
    progress(0.15, "Loading the model")
    model, tok = loader()
    max_len = sft_max_len(tok, max_len)  # a budget for the sample, on top of the chat template
    ok = [i for i, p in enumerate(pairs) if tokenize_sft(tok, [p], max_len)]
    scorer = GateScorer(model, tok, fp, data.harmful_probe, data.harmless_probe, ("S3",), max_len=max_len, batch_size=2)
    scores = np.full(len(pairs), np.nan)
    todo = [pairs[i] for i in ok]
    tick = lambda i, n: progress(0.2 + 0.8 * i / max(n, 1), f"Scoring sample {i} of {n}")  # noqa: E731
    scores[ok] = scorer.score(todo, log_every=0, on_progress=tick)[scorer.primary]
    del scorer, model
    release_gpu()
    filled = np.where(np.isnan(scores), -np.inf, scores)
    drop, dup, _ = gate_with_curatorkit(pairs, filled, drop_frac)  # CuratorKIT: dedup → BrinkSafetyGate
    order = np.argsort(-filled, kind="stable")
    return [
        {
            "rank": r,
            "risk": None if np.isnan(scores[i]) else round(float(scores[i]), 4),
            "decision": "drop (duplicate)"
            if dup[i]
            else ("drop" if drop[i] else ("skipped (too long)" if np.isnan(scores[i]) else "keep")),
            "prompt": pairs[i][0],
            "response": pairs[i][1],
        }
        for r, i in enumerate(order, 1)
    ]
