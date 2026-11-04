"""Stage "live": run Brink end to end on a real model and record it (the dashboard's "Run Brink" tab replays it).

    python experiments/live.py                              # configs/live.yaml → results/live
    python experiments/live.py --model Qwen/Qwen2.5-0.5B-Instruct --out results/live_qwen

fingerprint → gate the contaminated data → fine-tune unprotected → fine-tune with Brink → verify.
See brink/live.py. Resumable only as a whole: it re-runs if results/<out>/REPORT.md is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.eval.behavioral import make_judge
from brink.live import LiveConfig, run_live
from experiments.repair import resolve_threshold


def config(path: Path, model: str | None = None) -> tuple[LiveConfig, str]:
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("alarm_threshold", "auto") == "auto":
        raw["alarm_threshold"] = resolve_threshold(
            {"threshold": "auto", "signal": "S3", "calibration_from": str(ROOT / "results/go_nogo/summary.json")}
        )
    if model:
        raw["model_id"] = model
    return LiveConfig.from_dict(raw), raw.get("judge", "auto")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "live.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results" / "live"))
    ap.add_argument("--model", help="override model_id")
    args = ap.parse_args()
    cfg, judge = config(Path(args.config), args.model)
    summary = run_live(cfg, Path(args.out), judge=make_judge({"kind": judge}))
    print(summary.get("verdict", summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
