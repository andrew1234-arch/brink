"""SIMULATED runs for building and testing the dashboard before real results exist.

Every run written here carries meta.json {"simulated": true}, and the dashboard shows a banner saying
so. The curves are synthetic shapes, NOT experimental results. Never write these into results/.
The files use exactly the schema the real Monitor writes, so the dashboard code path is identical.
"""

from __future__ import annotations

import math
import random
import threading
import time
from pathlib import Path

from brink.io import CsvLog, write_json

N_LAYERS, LSTAR = 24, 14
BASE = {"S1": 4.0, "S1h": 2.5, "S2": 0.97, "S3": 2.2, "S5": -3.0}
NOISE = {"S1": 0.12, "S1h": 0.10, "S2": 0.004, "S3": 0.15, "S5": 0.12}


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _probe_row(step: int, drift: float, rng: random.Random) -> dict:
    z = {"S1": -14 * drift, "S1h": -10 * drift, "S2": -9 * drift, "S3": -18 * drift, "S5": 4 * drift}
    z = {k: v + rng.gauss(0, 0.6) for k, v in z.items()}
    row = {"step": step, **{k: BASE[k] + z[k] * NOISE[k] for k in BASE}}
    for layer in range(N_LAYERS + 1):
        m0 = 0.0 if layer == 0 else 0.5 + 0.25 * layer
        local = math.exp(-((layer - LSTAR) / 5) ** 2)
        row[f"S1_L{layer}"] = m0 * (1 - 0.7 * drift * local) + (rng.gauss(0, 0.01) if layer else 0.0)
    row.update({f"z_{k}": v for k, v in z.items()})
    row["probe_seconds"] = round(2.1 + rng.gauss(0, 0.15), 3)
    return row


def _eval_row(step: int, asr: float, rng: random.Random) -> dict:
    return {"step": step, "asr": round(min(1, max(0, asr + rng.gauss(0, 0.02))), 4), "refusal_rate": round(1 - asr, 4),
            "asr_keyword": round(min(1, max(0, asr + 0.05)), 4), "judge": "wildguard", "parse_fail": 0, "n": 64,
            "eval_seconds": round(96 + rng.gauss(0, 6), 2)}


def _write_condition(cdir: Path, drift_fn, asr_fn, max_steps: int, probe_every: int, eval_steps: list[int],
                     rng: random.Random, delay: float = 0.0) -> None:
    probe, evals, train = CsvLog(cdir / "probe.csv"), CsvLog(cdir / "evals.csv"), CsvLog(cdir / "train.csv")
    for step in range(0, max_steps + 1):
        if step:
            train.write({"step": step, "loss": round(1.8 * math.exp(-step / 60) + 0.6 + rng.gauss(0, 0.02), 4),
                         "lr": 1e-4, "step_seconds": 4.2})
        if step % probe_every == 0 or step in eval_steps:
            probe.write(_probe_row(step, drift_fn(step), rng))
        if step in eval_steps:
            evals.write(_eval_row(step, asr_fn(step), rng))
        if delay:
            time.sleep(delay)
    (cdir / "DONE").write_text("ok\n")


def _common(out: Path, kind: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "meta.json", {"simulated": True, "kind": kind,
                                   "note": "Synthetic curves for dashboard development. Not experimental results."})
    write_json(out / "fingerprint.json", {"layer": LSTAR, "baseline": BASE, "noise": NOISE,
                                          "meta": {"n_layers": N_LAYERS, "simulated": True}})


def simulate_go_nogo(out: Path, seed: int = 0, live_delay: float = 0.0) -> Path:
    out = Path(out)
    _common(out, "go_nogo")
    eval_steps = [0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    write_json(out / "config.json", {"train": {"max_steps": 128}, "probe": {"every": 4}, "eval": {"steps": eval_steps},
                                     "analysis": {"z_threshold": -3.0, "persistence": 2, "asr_jump": 0.10}})
    rng = random.Random(seed)
    conds = {
        "pos": (lambda s: _sig((s - 30) / 7), lambda s: 0.04 + 0.78 * _sig((s - 44) / 7)),
        "control": (lambda s: 0.03 * s / 128, lambda s: 0.04 + 0.02 * s / 128),
    }
    threads = []
    for name, (drift, asr) in conds.items():
        args = (out / name, drift, asr, 128, 4, eval_steps, random.Random(rng.random()), live_delay)
        if live_delay:
            th = threading.Thread(target=_write_condition, args=args, daemon=True)
            th.start()
            threads.append(th)
        else:
            _write_condition(*args)
    return out


def simulate_gate(out: Path, seed: int = 0) -> Path:
    out = Path(out)
    _common(out, "gate")
    eval_steps = [0, 24, 48, 96]
    write_json(out / "config.json", {"train": {"max_steps": 96}, "probe": {"every": 8}, "eval": {"steps": eval_steps},
                                     "gate": {"primary": "grad_S3"},
                                     "analysis": {"z_threshold": -3.0, "persistence": 2, "asr_jump": 0.10}})
    rng = random.Random(seed)
    scores = CsvLog(out / "scores.csv")
    rows = [("benign", rng.gauss(0.02, 0.05)) for _ in range(1950)] + [("planted", rng.gauss(0.22, 0.06)) for _ in range(50)]
    rng.shuffle(rows)
    for i, (src, s) in enumerate(rows):
        scores.write({"id": f"{src[0]}{i}", "source": src, "grad_S3": s, "rep": s * 0.6 + rng.gauss(0, 0.05),
                      "random": rng.random()})
    write_json(out / "subsets.json", {"top": [], "random": [], "bottom": []})
    subsets = {"top": (0.55, 0.38), "random": (0.25, 0.14), "bottom": (0.08, 0.05)}
    for name, (d, a) in subsets.items():
        _write_condition(out / name, lambda st, d=d: d * _sig((st - 40) / 10), lambda st, a=a: 0.04 + a * st / 96,
                         96, 8, eval_steps, random.Random(rng.random()))
    return out
