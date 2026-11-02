"""Experiment 02 analysis: safety vs capability vs over-refusal for every repair policy."""

from __future__ import annotations

import json
from pathlib import Path

from brink.io import read_csv, write_json

BRINK = ("halt", "rollback", "guard", "rollback_guard", "guard_always")
POSTHOC = ("posthoc_steer", "posthoc_resta")


def _cond(out_dir: Path, name: str) -> dict | None:
    d = out_dir / name
    if not (d / "evals.csv").exists():
        return None
    ev = read_csv(d / "evals.csv")
    last = ev[-1]
    info = json.loads((d / "repair.json").read_text()) if (d / "repair.json").exists() else {}
    rt = json.loads((d / "redteam.json").read_text()) if (d / "redteam.json").exists() else {}
    train = read_csv(d / "train.csv") if (d / "train.csv").exists() else []
    return {"asr": last.get("asr"), "over_refusal": last.get("over_refusal"), "heldout_loss": last.get("heldout_loss"),
            "asr0": ev[0].get("asr"), "over_refusal0": ev[0].get("over_refusal"), "loss0": ev[0].get("heldout_loss"),
            "steps_trained": int(train[-1]["step"]) if train else None, "final_step": int(last["step"]),
            "attack_resistance": rt.get("redteam_attack_resistance"), "benign_answered": rt.get("redteam_refusal"),
            **info}


def analyze(out_dir: Path, conditions: list[str], cfg: dict) -> dict:
    out_dir = Path(out_dir)
    res = {c: _cond(out_dir, c) for c in conditions}
    none = res.get("none")
    for c in POSTHOC:                     # post-hoc repairs start from the drifted `none` model
        if res.get(c) and none:
            res[c].update(asr0=none["asr"], over_refusal0=none["over_refusal"], loss0=none["heldout_loss"],
                          starts_from="none (drifted)")
    checks, best = {}, None
    if none and none.get("asr") is not None:
        drift = none["asr"] - (none["asr0"] or 0.0)
        cands = [(c, r) for c, r in res.items() if c in BRINK and r and r.get("asr") is not None]
        if cands and drift > 0:
            # Best Brink policy: lowest ASR among those that kept capability within tolerance of `none`.
            ok = [(c, r) for c, r in cands
                  if r.get("heldout_loss") is not None and none.get("heldout_loss") is not None
                  and r["heldout_loss"] <= none["heldout_loss"] * (1 + cfg["max_loss_increase"])]
            pool = ok or cands
            best, br = min(pool, key=lambda cr: cr[1]["asr"])
            removed = (none["asr"] - br["asr"]) / drift
            checks["R1_removes_most_of_the_drift"] = removed >= cfg["min_drift_removed"]
            checks["R2_keeps_task_capability"] = bool(ok) and best in dict(ok)
            if br.get("over_refusal") is not None and br.get("over_refusal0") is not None:
                checks["R3_no_over_refusal"] = br["over_refusal"] - br["over_refusal0"] <= cfg["max_over_refusal_increase"]
            ph = [r for c, r in res.items() if c in POSTHOC + ("safety_mix",) and r and r.get("asr") is not None]
            if ph:
                checks["R4_beats_baselines_posthoc_and_safety_mix"] = all(
                    br["asr"] <= r["asr"] and (r.get("heldout_loss") is None or br["heldout_loss"] <= r["heldout_loss"] * 1.02)
                    for r in ph)
        elif drift <= 0:
            checks["R0_baseline_drifted"] = False
    verdict = "REPAIR WORKS" if checks and all(checks.values()) else "REPAIR NOT PROVEN"
    summary = {"verdict": verdict, "checks": checks, "best_policy": best, "conditions": res}
    write_json(out_dir / "summary.json", summary)
    (out_dir / "REPORT.md").write_text(_report(summary))
    _plot(out_dir, res)
    return summary


def _f(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v:.0%}" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _report(s: dict) -> str:
    L = [f"# Repair experiment: **{s['verdict']}**", ""]
    L += [f"- {'✅' if v else '❌'} {k}" for k, v in s["checks"].items()]
    if s["best_policy"]:
        L += ["", f"Best Brink policy: **{s['best_policy']}**"]
    L += ["", "| condition | ASR start → end | over-refusal start → end | held-out loss start → end | "
             "AuditKIT attacks resisted | AuditKIT benign answered | steps trained | alarm | notes |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, r in s["conditions"].items():
        if not r:
            continue
        notes = []
        if r.get("rolled_back_to") is not None:
            notes.append(f"rolled back to step {r['rolled_back_to']}")
        if r.get("guard"):
            g = r["guard"]
            notes.append(f"guard projected {g['projected']}/{g['steps']} steps, removed {g['mean_removed_frac']:.0%} "
                         f"of |g|² on average, refresh {g['refresh_seconds']:.0f}s")
        if r.get("strength") is not None:
            notes.append(f"steer strength {r['strength']:.2f}")
        if r.get("alpha") is not None:
            notes.append(f"ReSta alpha {r['alpha']}")
        if r.get("starts_from"):
            notes.append(f"start = {r['starts_from']}")
        L.append(f"| {c} | {_f(r['asr0'], 1)} → {_f(r['asr'], 1)} | {_f(r['over_refusal0'], 1)} → {_f(r['over_refusal'], 1)} | "
                 f"{_f(r['loss0'])} → {_f(r['heldout_loss'])} | {_f(r.get('attack_resistance'), 1)} | "
                 f"{_f(r.get('benign_answered'), 1)} | {_f(r.get('steps_trained'))} | {_f(r.get('alarm_step'))} | "
                 f"{'; '.join(notes)} |")
    return "\n".join(L) + "\n"


def _plot(out_dir: Path, res: dict) -> None:
    """Trade-off scatter: ASR (lower = safer) vs held-out loss (lower = more capable). One point per condition,
    direct-labelled; colour = family (Brink in-training vs post-hoc), with the drifted baseline in grey."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pts = [(c, r) for c, r in res.items() if r and r.get("asr") is not None and r.get("heldout_loss") is not None]
    if not pts:
        return
    fam = {"brink": ("#2a78d6", "Brink (in training)"), "posthoc": ("#eb6834", "post-hoc (SafeTune)"),
           "none": ("#898781", "no repair / data-mix baseline")}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    seen = set()
    for c, r in pts:
        key = "none" if c in ("none", "safety_mix") else ("posthoc" if c in POSTHOC else "brink")
        color, label = fam[key]
        ax.scatter(r["heldout_loss"], r["asr"], s=70, color=color, edgecolors="white", linewidths=2, zorder=3,
                   label=None if key in seen else label)
        seen.add(key)
        ax.annotate(c, (r["heldout_loss"], r["asr"]), xytext=(6, 4), textcoords="offset points", fontsize=9,
                    color="#52514e")
    ax.set_xlabel("held-out task loss (lower = more capable)")
    ax.set_ylabel("ASR on held-out harmful prompts (lower = safer)")
    ax.set_title("Safety vs capability after fine-tuning on contaminated data", loc="left", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(color="#e1e0d9", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "repair.png", dpi=130)
    plt.close(fig)
