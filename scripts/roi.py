"""Write results/ROI.md: closed API vs the Brink-protected Tiny Aya, from configs/roi.yaml and experiment 04.

python scripts/roi.py                                  # uses the measured serving speed from results/pipeline
python scripts/roi.py --seconds-per-reply 0.4          # or give one
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.analysis.roi import measured_seconds_per_reply, report, roi
from brink.io import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "roi.yaml"))
    ap.add_argument("--seconds-per-reply", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    spr = args.seconds_per_reply or measured_seconds_per_reply(ROOT / cfg["pipeline_results"])
    if spr is None:
        raise SystemExit("no measured serving speed yet: run experiment 04 or pass --seconds-per-reply")
    if not cfg["closed_api"]["usd_per_m_output"] or not cfg["self_hosted"]["usd_per_gpu_hour"]:
        raise SystemExit("fill in the prices (and their sources) in configs/roi.yaml first")
    r = roi(cfg, spr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "roi.json", r)
    (out / "ROI.md").write_text(report(r))
    print(report(r))


if __name__ == "__main__":
    main()
