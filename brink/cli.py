"""The `brink` command: gate a fine-tuning file, or run Brink end to end on a model.

brink gate data.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --drop 0.1 --out gated/
brink run --model Qwen/Qwen2.5-0.5B-Instruct --out runs/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def gate_file(path, model_id, out, drop_frac=0.1, limit=None, max_len=512, loader=None, data=None, say=print) -> dict:
    """Score every sample of `path`, write scored.csv and cleaned.jsonl to `out`, return the counts."""
    from brink.live.upload import gate_pairs, read_pairs, write_gate_files

    out = Path(out)
    pairs = read_pairs(Path(path), limit=limit)
    last = [None]

    def progress(fraction: float, description: str) -> None:
        if description != last[0] and not description.startswith("Scoring sample"):
            say(description)
        last[0] = description

    rows = gate_pairs(pairs, model_id, out / "cache", drop_frac, max_len, loader, data, progress=progress)
    scored, cleaned = write_gate_files(rows, out)
    decisions = [r["decision"] for r in rows]
    counts = {
        "samples": len(rows),
        "eroding": sum(1 for r in rows if (r["risk"] or 0) > 0),
        "dropped": decisions.count("drop"),
        "duplicates": decisions.count("drop (duplicate)"),
        "too_long": decisions.count("skipped (too long)"),
        "kept": decisions.count("keep"),
        "scored": str(scored),
        "cleaned": str(cleaned),
    }
    say(
        f"{counts['samples']} samples scored on {model_id}: {counts['eroding']} would erode refusal. "
        f"Dropped the riskiest {counts['dropped']} and {counts['duplicates']} duplicates; kept {counts['kept']}."
    )
    say(f"scores:  {scored}")
    say(f"cleaned: {cleaned}")
    return counts


def run_model(
    model_id, out, policy="rollback_guard", steps=None, contamination=None, drop_frac=None, say=print
) -> dict:
    """Fingerprint, gate, fine-tune unprotected and with Brink, verify. Writes a run folder with REPORT.md."""
    from brink.eval.behavioral import make_judge
    from brink.live import LiveConfig, run_live

    overrides = {"max_steps": steps, "contamination": contamination, "gate_drop_frac": drop_frac}
    cfg = LiveConfig(model_id=model_id or "", policy=policy, **{k: v for k, v in overrides.items() if v is not None})
    summary = run_live(cfg, Path(out), judge=make_judge({"kind": "auto"}), say=say)
    say(summary.get("verdict", str(summary)))
    say(f"report: {Path(out) / 'REPORT.md'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="brink", description="Keep a model safe while you fine-tune it.")
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gate", help="score a fine-tuning file and write a cleaned one")
    g.add_argument("file", help="fine-tuning data: .jsonl, .json or .csv with prompt / response pairs")
    g.add_argument("--model", required=True, help="the model you are going to fine-tune (Hugging Face id or path)")
    g.add_argument("--drop", type=float, default=0.1, help="share of the riskiest samples to drop (default 0.1)")
    g.add_argument("--out", default="brink-gate", help="output folder (default brink-gate)")
    g.add_argument("--limit", type=int, default=None, help="score only the first N samples")
    r = sub.add_parser("run", help="run Brink end to end on a model: gate, fine-tune twice, verify")
    r.add_argument("--model", default=None, help="Hugging Face id (default: Tiny Aya on a GPU, a 0.5B model on CPU)")
    r.add_argument("--out", default="brink-run", help="run folder (default brink-run)")
    r.add_argument("--policy", default="rollback_guard", choices=["rollback_guard", "halt", "guard_always"])
    r.add_argument("--steps", type=int, default=None, help="fine-tuning steps")
    r.add_argument("--contamination", type=float, default=None, help="harmful share of the fine-tuning data")
    r.add_argument("--drop", type=float, default=None, help="share of the riskiest samples the gate drops")
    args = ap.parse_args(argv)
    try:
        if args.command == "gate":
            if not 0 <= args.drop <= 0.5:
                ap.error("--drop must be between 0 and 0.5")
            gate_file(args.file, args.model, args.out, args.drop, args.limit)
        else:
            run_model(args.model, args.out, args.policy, args.steps, args.contamination, args.drop)
    except (ValueError, FileNotFoundError) as e:
        print(f"brink: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
