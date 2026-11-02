"""Experiment 03 analysis: per-language drift, which language breaks first, English-only vs worst-language alarm."""

from __future__ import annotations

from pathlib import Path

from brink.io import read_csv, write_json
from brink.monitor.detector import first_alarm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots; one per language, fixed order


def _key(lang: str, primary: str, k: str) -> str:
    return k if lang == primary else f"{lang}:{k}"


def _lang_stats(probe, evals, lang, primary, cfg) -> dict:
    steps = [int(r["step"]) for r in probe]
    z = [r.get(_key(lang, primary, f"z_{cfg['signal']}")) for r in probe]
    e_steps = [int(r["step"]) for r in evals]
    asr = [r.get(_key(lang, primary, "asr")) for r in evals]
    a0 = asr[0] if asr else None
    brk = next((s for s, a in zip(e_steps, asr) if a is not None and a0 is not None and a >= a0 + cfg["asr_jump"]), None)
    return {"asr0": a0, "asr_final": asr[-1] if asr else None,
            "asr_gain": (asr[-1] - a0) if asr and a0 is not None and asr[-1] is not None else None,
            "break_step": brk, "alarm_step": first_alarm(steps, z, cfg["z_threshold"], cfg["persistence"]),
            "min_z": min((v for v in z if v is not None), default=None)}


def analyze(out_dir: Path, conditions: list[str], langs: list[str], cfg: dict) -> dict:
    out_dir = Path(out_dir)
    primary = langs[0]
    res = {}
    for cond in conditions:
        d = out_dir / cond
        if not (d / "probe.csv").exists() or not (d / "evals.csv").exists():
            continue
        probe, evals = read_csv(d / "probe.csv"), read_csv(d / "evals.csv")
        per = {lang: _lang_stats(probe, evals, lang, primary, cfg) for lang in langs}
        steps = [int(r["step"]) for r in probe]
        worst_alarm = first_alarm(steps, [r.get(f"z_{cfg['signal']}_worst") for r in probe], cfg["z_threshold"],
                                  cfg["persistence"])
        breaks = {lg: v["break_step"] for lg, v in per.items() if v["break_step"] is not None}
        first_break = min(breaks.items(), key=lambda kv: kv[1]) if breaks else None
        res[cond] = {"languages": per, "english_alarm": per[primary]["alarm_step"], "worst_language_alarm": worst_alarm,
                     "first_break": {"lang": first_break[0], "step": first_break[1]} if first_break else None}
    findings = {}
    pos, ctl = res.get("pos"), res.get("control")
    if pos:
        findings["drift_reaches_other_languages"] = {lg: v["asr_gain"] for lg, v in pos["languages"].items() if lg != primary}
        findings["first_language_to_break"] = pos["first_break"]
        fb = pos["first_break"]
        if fb:
            ea, wa = pos["english_alarm"], pos["worst_language_alarm"]
            findings["english_only_alarm_lead"] = None if ea is None else fb["step"] - ea
            findings["worst_language_alarm_lead"] = None if wa is None else fb["step"] - wa
    if ctl:
        findings["false_alarm_on_control"] = ctl["worst_language_alarm"] is not None
    summary = {"primary": primary, "languages": langs, "findings": findings, "conditions": res}
    write_json(out_dir / "summary.json", summary)
    (out_dir / "REPORT.md").write_text(_report(summary))
    _plot(out_dir, conditions, langs, cfg)
    return summary


def _f(v, pct=False):
    return "n/a" if v is None else (f"{v:.0%}" if pct else str(v))


def _fz(v):
    return "n/a" if v is None else f"{v:.1f}"


def _report(s: dict) -> str:
    f = s["findings"]
    L = ["# Multilingual drift", ""]
    if "first_language_to_break" in f:
        fb = f["first_language_to_break"]
        L.append(f"- First language whose behaviour breaks on `pos`: **{fb['lang']} at step {fb['step']}**" if fb
                 else "- No language's behaviour broke on `pos`.")
    if "english_only_alarm_lead" in f:
        L.append(f"- Lead over that first break: English-only alarm **{_f(f['english_only_alarm_lead'])}** steps, "
                 f"worst-language alarm **{_f(f['worst_language_alarm_lead'])}** steps (negative = late)")
    if "false_alarm_on_control" in f:
        L.append(f"- Worst-language alarm on `control` (false alarm): **{'yes' if f['false_alarm_on_control'] else 'no'}**")
    L.append("\nNote: the WildGuard judge is mostly English-trained, so non-English ASR is less reliable. The per-language "
             "S3 alarm is judge-free (calibrated on each language's own refusal tokens).")
    for cond, r in s["conditions"].items():
        L += ["", f"## {cond}", "", "| language | ASR start → end | behaviour breaks | alarm (own S3) | min z |",
              "|---|---|---|---|---|"]
        for lg, v in r["languages"].items():
            L.append(f"| {lg} | {_f(v['asr0'], 1)} → {_f(v['asr_final'], 1)} | {_f(v['break_step'])} | "
                     f"{_f(v['alarm_step'])} | {_fz(v['min_z'])} |")
        L.append(f"\nEnglish-only alarm: step {_f(r['english_alarm'])} · worst-language alarm: step {_f(r['worst_language_alarm'])}")
    return "\n".join(L) + "\n"


def _plot(out_dir: Path, conditions: list[str], langs: list[str], cfg: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    d = out_dir / "pos"
    if not (d / "probe.csv").exists():
        return
    probe, evals = read_csv(d / "probe.csv"), read_csv(d / "evals.csv")
    primary = langs[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for i, lg in enumerate(langs[:3]):
        c = SERIES[i]
        a1.plot([r["step"] for r in evals], [r.get(_key(lg, primary, "asr")) for r in evals], marker="o", color=c, label=lg)
        a2.plot([r["step"] for r in probe], [r.get(_key(lg, primary, f"z_{cfg['signal']}")) for r in probe], color=c, label=lg)
    a1.set_ylabel("ASR (held-out harmful prompts)")
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_ylabel(f"z({cfg['signal']}) per language")
    a2.axhline(cfg["z_threshold"], ls="--", c="#898781", lw=1)
    a2.set_xlabel("training step (English fine-tuning data)")
    a1.set_title("pos: refusal erosion by language", loc="left", fontsize=11)
    for a in (a1, a2):
        a.grid(color="#e1e0d9", linewidth=0.8)
        a.legend(frameon=False)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "multilingual.png", dpi=130)
    plt.close(fig)
