"""Experiment 00: go / no-go.

Question: on Tiny Aya, does LoRA fine-tuning reproducibly break safety, and do Brink's cheap probe
signals move with it (and not with safe fine-tuning)?

    python experiments/go_nogo.py --config configs/go_nogo.yaml --out results/go_nogo

Conditions: pos (harmful compliance, expected drift) and control (benign + refusals, expected none).
Resumable: finished conditions are skipped on re-run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brink.analysis.go_nogo import analyze  # noqa: E402
from brink.data import GoNoGoData, build_go_nogo_data  # noqa: E402
from brink.eval.behavioral import make_judge  # noqa: E402
from brink.runner import calibrate_or_load, run_condition  # noqa: E402
from brink.utils import load_model_and_tokenizer, set_seed, write_json  # noqa: E402


def make_loader(cfg: dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return lambda: load_model_and_tokenizer(cfg["model"]["id"], device,
                                            trust_remote_code=cfg["model"].get("trust_remote_code", False))


def run(cfg: dict, out_dir: Path, data: GoNoGoData | None = None, loader=None, judge=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    loader = loader or make_loader(cfg)
    write_json(out_dir / "config.json", cfg)
    t_start = time.time()

    data = data or build_go_nogo_data(cfg["data"], seed)
    write_json(out_dir / "data_summary.json", data.summary())
    print(f"[data] {data.summary()}", flush=True)
    judge = judge or make_judge(cfg["judge"])

    # 1. Calibrate the fingerprint on the reference model (once).
    fp = calibrate_or_load(out_dir, loader, data, cfg["probe"], seed)

    # 2. One fine-tuning run per condition, monitored and evaluated along the way.
    for cond in cfg["conditions"]:
        run_condition(cond, data.train[cond], out_dir, fp, loader, data, judge, cfg, seed)

    # 3. Verdict.
    summary = analyze(out_dir, cfg["conditions"], cfg["analysis"])
    summary["wall_minutes"] = round((time.time() - t_start) / 60, 1)
    write_json(out_dir / "summary.json", summary)
    print((out_dir / "REPORT.md").read_text(), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/go_nogo.yaml")
    ap.add_argument("--out", default="results/go_nogo")
    ap.add_argument("--conditions", default=None, help="comma-separated override, e.g. pos,control,benign")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.conditions:
        cfg["conditions"] = args.conditions.split(",")
    run(cfg, Path(args.out))


if __name__ == "__main__":
    main()
