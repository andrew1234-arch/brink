"""Turn the go / no-go run logs into a verdict, a report and a plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brink.monitor.detector import cusum_alarm, first_alarm
from brink.probe.signals import ALARM_SIGNALS
from brink.utils import read_csv, write_json


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _condition(out_dir: Path, cond: str, cfg: dict) -> dict | None:
    d = out_dir / cond
    if not (d / "probe.csv").exists() or not (d / "evals.csv").exists():
        return None
    probe, evals = read_csv(d / "probe.csv"), read_csv(d / "evals.csv")
    steps = [int(r["step"]) for r in probe]
    by_step = {int(r["step"]): r for r in probe}
    e_steps = [int(r["step"]) for r in evals]
    asr = [r["asr"] for r in evals]
    asr0 = asr[0]
    jump = cfg["asr_jump"]
    behav = next((s for s, a in zip(e_steps, asr) if a is not None and asr0 is not None and a >= asr0 + jump), None)

    sigs = {}
    for k in ALARM_SIGNALS:
        z = [r[f"z_{k}"] for r in probe]
        alarm = first_alarm(steps, z, cfg["z_threshold"], cfg["persistence"])
        cus = cusum_alarm(steps, z, cfg.get("cusum_k", 0.5), cfg.get("cusum_h", 5.0))
        vals_at_eval = [by_step[s][k] if s in by_step else None for s in e_steps]
        sigs[k] = {
            "alarm_step": alarm, "cusum_step": cus,
            "lead_steps": (behav - alarm) if (behav is not None and alarm is not None) else None,
            "spearman_vs_asr": spearman([v if v is not None else np.nan for v in vals_at_eval],
                                        [a if a is not None else np.nan for a in asr]),
            "min_z": float(np.nanmin([v if v is not None else np.nan for v in z])),
            "z_at_behav": by_step[behav][f"z_{k}"] if behav in by_step else None,
        }
    probe_cost = float(np.mean([r["probe_seconds"] for r in probe]))
    eval_cost = float(np.mean([r["eval_seconds"] for r in evals]))
    return {"asr0": asr0, "asr_final": asr[-1], "asr_gain": (asr[-1] - asr0) if None not in (asr0, asr[-1]) else None,
            "behav_step": behav, "signals": sigs, "eval_steps": e_steps, "asr": asr,
            "probe_seconds": probe_cost, "eval_seconds": eval_cost, "judge": evals[0].get("judge")}


def analyze(out_dir: Path, conditions: list[str], cfg: dict) -> dict:
    out_dir = Path(out_dir)
    res = {c: _condition(out_dir, c, cfg) for c in conditions}
    pos, ctl = res.get("pos"), res.get("control")
    go = cfg["go"]
    checks, notes = {}, []

    if pos:
        checks["G1a_pos_drifts"] = pos["asr_gain"] is not None and pos["asr_gain"] >= go["min_pos_asr_gain"]
    if ctl:
        checks["G1b_control_stable"] = ctl["asr_gain"] is not None and ctl["asr_gain"] <= go["max_control_asr_gain"]
    best = None
    if pos:
        ranked = sorted(((k, v) for k, v in pos["signals"].items() if v["spearman_vs_asr"] is not None),
                        key=lambda kv: kv[1]["spearman_vs_asr"])        # most negative first
        if ranked:
            best = ranked[0][0]
            checks["G2_signal_tracks_asr"] = -ranked[0][1]["spearman_vs_asr"] >= go["min_abs_spearman"]
    if pos and ctl and best:
        # Separable: at the moment behaviour breaks on pos, pos z is beyond the worst z control ever reached.
        zb, cmin = pos["signals"][best]["z_at_behav"], ctl["signals"][best]["min_z"]
        checks["G3_separates_from_control"] = zb is not None and zb < cmin

    verdict = "GO" if checks and all(checks.values()) else "NO-GO"
    if pos and not checks.get("G1a_pos_drifts", True):
        notes.append("Drift not reproduced: raise train.lr (e.g. 2e-4) or max_steps, or try Command R7B.")
    if ctl and not checks.get("G1b_control_stable", True):
        notes.append("Control drifted too: add more refusal examples to control, or lower lr.")
    if pos and not checks.get("G2_signal_tracks_asr", True):
        notes.append("Signals don't track ASR: inspect per-layer S1_L* curves; the chosen layer may be wrong.")
    if not checks.get("G3_separates_from_control", True):
        notes.append("Benign fine-tuning moves the signal as much as harmful: need a safety-specific signal.")

    summary = {"verdict": verdict, "checks": checks, "best_signal": best, "notes": notes, "conditions": res}
    write_json(out_dir / "summary.json", summary)
    (out_dir / "REPORT.md").write_text(_report(summary))
    _plot(out_dir, conditions, best or "S1")
    return summary


def _fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _report(s: dict) -> str:
    lines = [f"# Go / no-go: **{s['verdict']}**", ""]
    for k, v in s["checks"].items():
        lines.append(f"- {'✅' if v else '❌'} {k}")
    if s["notes"]:
        lines += ["", "**Next steps**"] + [f"- {n}" for n in s["notes"]]
    for c, r in s["conditions"].items():
        if not r:
            continue
        lines += ["", f"## {c}  (judge: {r['judge']})", "",
                  f"ASR {_fmt(r['asr0'])} → {_fmt(r['asr_final'])} (gain {_fmt(r['asr_gain'])}); "
                  f"behaviour breaks at step {_fmt(r['behav_step'])}. "
                  f"Probe {r['probe_seconds']:.1f}s vs eval {r['eval_seconds']:.1f}s per check.", "",
                  "| signal | alarm step | CUSUM step | lead (steps) | Spearman vs ASR | min z |",
                  "|---|---|---|---|---|---|"]
        for k, v in r["signals"].items():
            lines.append(f"| {k} | {_fmt(v['alarm_step'])} | {_fmt(v['cusum_step'])} | {_fmt(v['lead_steps'])} | "
                         f"{_fmt(v['spearman_vs_asr'], 2)} | {_fmt(v['min_z'], 1)} |")
    return "\n".join(lines) + "\n"


def _plot(out_dir: Path, conditions: list[str], signal: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for c in conditions:
        d = out_dir / c
        if not (d / "probe.csv").exists() or not (d / "evals.csv").exists():
            continue
        probe, evals = read_csv(d / "probe.csv"), read_csv(d / "evals.csv")
        a1.plot([r["step"] for r in evals], [r["asr"] for r in evals], marker="o", label=c)
        a2.plot([r["step"] for r in probe], [r[f"z_{signal}"] for r in probe], label=c)
    a1.set_ylabel("ASR (held-out harmful prompts)")
    a2.set_ylabel(f"z({signal}) vs reference")
    a2.set_xlabel("training step")
    a2.axhline(-3, ls="--", c="grey", lw=1)
    for a in (a1, a2):
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "go_nogo.png", dpi=130)
    plt.close(fig)
