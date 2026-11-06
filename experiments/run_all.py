"""Run every Brink experiment in order on any GPU machine (Kaggle has its own notebook: kaggle_run_all.ipynb).

    python experiments/run_all.py                         # every stage not finished yet
    python experiments/run_all.py --stages go_nogo gate   # a subset, in the given order
    python experiments/run_all.py --dry-run               # show what would run

Order matters: `repair`, `pipeline` and `live` read the alarm threshold calibrated by `go_nogo`. A stage counts as done
once its REPORT.md exists. Each stage runs preflight first and is skipped (not fatal) if preflight fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.io import stage_finished

STAGES = ["go_nogo", "gate", "repair", "multilingual", "pipeline", "live", "circuit"]
HOURS_T4X2 = {
    "go_nogo": 4.0,
    "gate": 1.5,
    "repair": 2.7,
    "multilingual": 2.0,
    "pipeline": 1.5,
    "live": 0.4,
    "circuit": 0.6,
}


def done(stage: str, results: Path) -> bool:
    return stage_finished(results / stage)


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    ap.add_argument("--results", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()
    results = ROOT / args.results

    pending = [s for s in args.stages if not done(s, results)]
    print(f"done: {[s for s in args.stages if s not in pending] or 'nothing'}")
    print(f"to run: {pending or 'nothing'} (about {sum(HOURS_T4X2[s] for s in pending):.1f} h on a T4 x2)")
    if args.dry_run or not pending:
        return 0

    status: dict[str, str] = {}
    for stage in pending:
        cfg = f"configs/{stage}.yaml"
        t0 = time.time()
        print(f"\n{'=' * 80}\n{stage}\n{'=' * 80}", flush=True)
        # `live` validates its own model and data as its first stages; `circuit` needs `pip install circuitkit`.
        preflight = [sys.executable, "experiments/preflight.py", "--config", cfg]
        if stage not in ("live", "circuit") and not args.skip_preflight and run(preflight) != 0:
            status[stage] = "skipped: preflight failed"
            continue
        rc = run([sys.executable, f"experiments/{stage}.py", "--config", cfg, "--out", f"{args.results}/{stage}"])
        minutes = (time.time() - t0) / 60
        status[stage] = f"done in {minutes:.0f} min" if rc == 0 and done(stage, results) else f"FAILED (exit {rc})"

    print("\nSTATUS")
    for k, v in status.items():
        print(f"  {k:13s} {v}")
    return 0 if all(v.startswith("done") for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
