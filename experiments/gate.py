"""Experiment 01: the data gate.

Two questions:
  A. Detection: in a benign pool with a few planted harmful-compliance samples, does the gate score rank
     the planted ones at the top? (AUROC, recall)  Cheap sanity check with ground truth.
  B. Prediction: among purely benign samples, does fine-tuning on the gate's top-scored subset break safety
     more than a random subset, and the bottom-scored subset less? This is the real claim.

    python experiments/gate.py --config configs/gate.yaml --out results/gate

Resumable: scores and finished conditions are reused on re-run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brink.analysis.gate import analyze  # noqa: E402
from brink.data import GateData, build_gate_data  # noqa: E402
from brink.eval.behavioral import make_judge  # noqa: E402
from brink.gate.scorer import GateScorer  # noqa: E402
from brink.runner import calibrate_or_load, release_gpu, run_condition  # noqa: E402
from brink.utils import CsvLog, read_csv, set_seed, write_json, write_jsonl  # noqa: E402
from experiments.go_nogo import make_loader  # noqa: E402


def score_pool(out_dir: Path, data: GateData, fp, loader, gc: dict, seed: int) -> list[str]:
    """Write scores.csv (ids, source, scores only; no text) and return the score columns."""
    path = out_dir / "scores.csv"
    if path.exists():
        print("[gate] reusing scores.csv", flush=True)
        return [k for k in read_csv(path)[0] if k.startswith(("grad_", "rep", "random")) and not k.startswith("grad_dot")
                and k != "grad_norm"]
    model, tok = loader()
    t0 = time.time()
    scorer = GateScorer(model, tok, fp, data.harmful_probe, data.harmless_probe, gc["targets"], gc["lora_r"],
                        gc["lora_alpha"], gc["max_len"], gc["batch_size"], seed)
    print(f"[gate] target gradients ready in {scorer.target_seconds:.0f}s; scoring {len(data.pool)} samples", flush=True)
    pairs = [(r["user"], r["assistant"]) for r in data.pool]
    s = scorer.score(pairs)
    t_grad = time.time() - t0
    s["rep"] = scorer.rep(pairs, data.anchors_unsafe, data.anchors_safe)
    s["random"] = np.random.default_rng(seed).random(len(pairs))
    timing = {"grad_seconds": round(t_grad, 1), "rep_seconds": round(time.time() - t0 - t_grad, 1),
              "per_sample_grad_seconds": round(t_grad / len(pairs), 3), "n": len(pairs)}
    write_json(out_dir / "scoring_time.json", timing)
    print(f"[gate] scoring done: {timing}", flush=True)

    log = CsvLog(path)
    for i, r in enumerate(data.pool):
        log.write({"id": r["id"], "source": r["source"], **{k: float(v[i]) for k, v in s.items()}})
    # Texts go to a git-ignored file: planted samples contain harmful content.
    write_jsonl(out_dir / "pool.jsonl", data.pool, mode="w")
    del model, scorer
    release_gpu()
    return [f"grad_{t}" for t in gc["targets"]] + ["rep", "random"]


def select_subsets(out_dir: Path, data: GateData, primary: str, k: int, seed: int) -> dict[str, list]:
    rows = {r["id"]: r for r in read_csv(out_dir / "scores.csv")}
    benign = [r for r in data.pool if r["source"] == "benign"]
    order = sorted(benign, key=lambda r: -rows[r["id"]][primary])
    rng = np.random.default_rng(seed + 1)
    rand = [benign[i] for i in rng.choice(len(benign), size=k, replace=False)]
    subsets = {"top": order[:k], "random": rand, "bottom": order[-k:]}
    write_json(out_dir / "subsets.json", {c: [r["id"] for r in v] for c, v in subsets.items()})
    return {c: [(r["user"], r["assistant"]) for r in v] for c, v in subsets.items()}


def run(cfg: dict, out_dir: Path, data: GateData | None = None, loader=None, judge=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    loader = loader or make_loader(cfg)
    write_json(out_dir / "config.json", cfg)
    t_start = time.time()

    data = data or build_gate_data(cfg["data"], seed)
    write_json(out_dir / "data_summary.json", data.summary())
    print(f"[data] {data.summary()}", flush=True)

    fp = calibrate_or_load(out_dir, loader, data, cfg["probe"], seed)
    cols = score_pool(out_dir, data, fp, loader, cfg["gate"], seed)
    primary = cfg["gate"].get("primary") or f"grad_{cfg['gate']['targets'][0]}"

    judge = judge or make_judge(cfg["judge"])
    subsets = select_subsets(out_dir, data, primary, cfg["select"]["k"], seed)
    for cond in cfg["conditions"]:
        run_condition(cond, subsets[cond], out_dir, fp, loader, data, judge, cfg, seed, heldout=data.heldout_benign)

    summary = analyze(out_dir, cfg["conditions"], cols, primary, cfg["analysis"])
    summary["wall_minutes"] = round((time.time() - t_start) / 60, 1)
    write_json(out_dir / "summary.json", summary)
    print((out_dir / "REPORT.md").read_text(), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gate.yaml")
    ap.add_argument("--out", default="results/gate")
    ap.add_argument("--conditions", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.conditions:
        cfg["conditions"] = args.conditions.split(",")
    run(cfg, Path(args.out))


if __name__ == "__main__":
    main()
