"""Every headline number the write-up needs, each traced to the file and key it came from.

    python scripts/headlines.py                         # writes results/HEADLINES.md and results/headlines.json
    python scripts/headlines.py --fill docs/ARTICLE.md   # also fills ⟦name⟧ markers in place

Write-ups use named markers such as ⟦go_nogo.lead_mean⟧. --fill replaces each marker whose number exists and
leaves the rest (missing stages, or quantities the stage didn't produce) untouched and listed. Nothing is
estimated or rounded beyond the stated format: a marker is filled only from a summary.json value.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MARKER = re.compile(r"⟦([a-z_]+\.[a-z0-9_]+)⟧")


def pct(v) -> str:
    return f"{100 * v:.0f} %"


def num(v) -> str:
    return f"{v:.1f}" if isinstance(v, float) and not float(v).is_integer() else f"{v:.0f}"


def dec(v) -> str:
    return f"{v:.3f}"


def two(v) -> str:
    return f"{v:.2f}"


def text(v) -> str:
    return str(v)


def _get(d, *path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _best(s):
    return _get(s, "signals", _get(s, "best_mechanistic_signal") or _get(s, "best_signal"))


def _ratio(a, b):
    return None if a is None or not b else a / b


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _runs_mean(s, key):
    """Mean of `key` over every go / no-go run ({condition: [per-seed rows]})."""
    return _mean([r.get(key) for rows in (_get(s, "runs") or {}).values() for r in rows])


def _pipe(s, cond, which, key):
    return _get(s, "conditions", cond, which, key)


def _repair(s, cond, key="asr"):
    return _get(s, "conditions", cond, key)


SPEC: dict[str, tuple[str, Callable, Callable, str]] = {
    "go_nogo.verdict": ("go_nogo", lambda s: s.get("verdict"), text, "verdict"),
    "go_nogo.signal": (
        "go_nogo",
        lambda s: s.get("best_mechanistic_signal") or s.get("best_signal"),
        text,
        "best_mechanistic_signal (else best_signal)",
    ),
    "go_nogo.lead_mean": ("go_nogo", lambda s: _get(_best(s), "lead_mean"), num, "signals[best].lead_mean (steps)"),
    "go_nogo.lead_sd": ("go_nogo", lambda s: _get(_best(s), "lead_sd"), num, "signals[best].lead_sd (steps)"),
    "go_nogo.false_alarm_rate": (
        "go_nogo",
        lambda s: _get(_best(s), "loo_false_alarm_rate"),
        pct,
        "signals[best].loo_false_alarm_rate",
    ),
    "go_nogo.detection_rate": (
        "go_nogo",
        lambda s: _get(_best(s), "detection_rate"),
        pct,
        "signals[best].detection_rate",
    ),
    "go_nogo.s3_lead_mean": (
        "go_nogo",
        lambda s: _get(s, "signals", "S3", "lead_mean"),
        num,
        "signals.S3.lead_mean (steps)",
    ),
    "go_nogo.judge_lead_mean": (
        "go_nogo",
        lambda s: _get(s, "cheap_judge_baseline", "lead_mean"),
        num,
        "cheap_judge_baseline.lead_mean (steps)",
    ),
    "go_nogo.judge_cost_ratio": (
        "go_nogo",
        lambda s: _ratio(_get(s, "cheap_judge_baseline", "seconds_per_check"), _runs_mean(s, "probe_seconds")),
        num,
        "cheap_judge_baseline.seconds_per_check / mean(runs.*[].probe_seconds)",
    ),
    "gate.verdict": ("gate", lambda s: s.get("verdict"), text, "verdict"),
    "gate.auroc": (
        "gate",
        lambda s: _get(s, "detection", s.get("primary") or "grad_S3", "auroc"),
        two,
        "detection[primary].auroc",
    ),
    "gate.classifier_auroc": (
        "gate",
        lambda s: _get(s, "detection", "classifier", "auroc"),
        two,
        "detection.classifier.auroc (WildGuard)",
    ),
    "gate.classifier_on_top": ("gate", lambda s: s.get("classifier_on_gate_top"), pct, "classifier_on_gate_top"),
    "gate.asr_top": ("gate", lambda s: _get(s, "runs", "top", "asr_final"), pct, "runs.top.asr_final"),
    "gate.asr_random": ("gate", lambda s: _get(s, "runs", "random", "asr_final"), pct, "runs.random.asr_final"),
    "gate.asr_bottom": ("gate", lambda s: _get(s, "runs", "bottom", "asr_final"), pct, "runs.bottom.asr_final"),
    "repair.verdict": ("repair", lambda s: s.get("verdict"), text, "verdict"),
    "repair.best_policy": ("repair", lambda s: s.get("best_policy"), text, "best_policy"),
    "repair.asr_none": ("repair", lambda s: _repair(s, "none"), pct, "conditions.none.asr"),
    "repair.asr_brink": (
        "repair",
        lambda s: _repair(s, s.get("best_policy") or "rollback_guard"),
        pct,
        "conditions[best_policy].asr",
    ),
    "repair.loss_brink": (
        "repair",
        lambda s: _repair(s, s.get("best_policy") or "rollback_guard", "heldout_loss"),
        dec,
        "conditions[best_policy].heldout_loss",
    ),
    "repair.loss_none": ("repair", lambda s: _repair(s, "none", "heldout_loss"), dec, "conditions.none.heldout_loss"),
    "repair.asr_safety_mix": ("repair", lambda s: _repair(s, "safety_mix"), pct, "conditions.safety_mix.asr"),
    "repair.asr_resta": ("repair", lambda s: _repair(s, "posthoc_resta"), pct, "conditions.posthoc_resta.asr"),
    "repair.asr_steer": ("repair", lambda s: _repair(s, "posthoc_steer"), pct, "conditions.posthoc_steer.asr"),
    "multilingual.first_break_lang": (
        "multilingual",
        lambda s: _get(s, "findings", "first_language_to_break", "lang"),
        text,
        "findings.first_language_to_break.lang",
    ),
    "multilingual.worst_lead": (
        "multilingual",
        lambda s: _get(s, "findings", "worst_language_alarm_lead"),
        num,
        "findings.worst_language_alarm_lead (steps)",
    ),
    "multilingual.english_lead": (
        "multilingual",
        lambda s: _get(s, "findings", "english_only_alarm_lead"),
        num,
        "findings.english_only_alarm_lead (steps)",
    ),
    "pipeline.verdict": ("pipeline", lambda s: s.get("verdict"), text, "verdict"),
    "pipeline.asr_before": ("pipeline", lambda s: _pipe(s, "naive", "ref", "asr"), pct, "conditions.naive.ref.asr"),
    "pipeline.asr_naive": ("pipeline", lambda s: _pipe(s, "naive", "final", "asr"), pct, "conditions.naive.final.asr"),
    "pipeline.asr_brink": ("pipeline", lambda s: _pipe(s, "brink", "final", "asr"), pct, "conditions.brink.final.asr"),
    "pipeline.asr_hi_before": (
        "pipeline",
        lambda s: _pipe(s, "naive", "ref", "asr_hi"),
        pct,
        "conditions.naive.ref.asr_hi",
    ),
    "pipeline.asr_hi_naive": (
        "pipeline",
        lambda s: _pipe(s, "naive", "final", "asr_hi"),
        pct,
        "conditions.naive.final.asr_hi",
    ),
    "pipeline.asr_hi_brink": (
        "pipeline",
        lambda s: _pipe(s, "brink", "final", "asr_hi"),
        pct,
        "conditions.brink.final.asr_hi",
    ),
    "pipeline.over_refusal_naive": (
        "pipeline",
        lambda s: _pipe(s, "naive", "final", "over_refusal"),
        pct,
        "conditions.naive.final.over_refusal",
    ),
    "pipeline.over_refusal_brink": (
        "pipeline",
        lambda s: _pipe(s, "brink", "final", "over_refusal"),
        pct,
        "conditions.brink.final.over_refusal",
    ),
    "pipeline.chrf_before": (
        "pipeline",
        lambda s: _pipe(s, "naive", "ref", "task_chrf"),
        num,
        "conditions.naive.ref.task_chrf",
    ),
    "pipeline.chrf_naive": (
        "pipeline",
        lambda s: _pipe(s, "naive", "final", "task_chrf"),
        num,
        "conditions.naive.final.task_chrf",
    ),
    "pipeline.chrf_brink": (
        "pipeline",
        lambda s: _pipe(s, "brink", "final", "task_chrf"),
        num,
        "conditions.brink.final.task_chrf",
    ),
    "pipeline.loss_naive": (
        "pipeline",
        lambda s: _pipe(s, "naive", "final", "heldout_loss"),
        dec,
        "conditions.naive.final.heldout_loss",
    ),
    "pipeline.loss_brink": (
        "pipeline",
        lambda s: _pipe(s, "brink", "final", "heldout_loss"),
        dec,
        "conditions.brink.final.heldout_loss",
    ),
    "pipeline.redteam_naive": (
        "pipeline",
        lambda s: _pipe(s, "naive", "final", "redteam_attack_resistance"),
        pct,
        "conditions.naive.final.redteam_attack_resistance",
    ),
    "pipeline.redteam_brink": (
        "pipeline",
        lambda s: _pipe(s, "brink", "final", "redteam_attack_resistance"),
        pct,
        "conditions.brink.final.redteam_attack_resistance",
    ),
    "pipeline.planted_caught": (
        "pipeline",
        lambda s: _pipe(s, "brink", "gate", "planted_caught"),
        num,
        "conditions.brink.gate.planted_caught",
    ),
    "pipeline.planted_total": (
        "pipeline",
        lambda s: _pipe(s, "brink", "gate", "planted_total"),
        num,
        "conditions.brink.gate.planted_total",
    ),
    "live.verdict": ("live", lambda s: s.get("verdict"), text, "verdict"),
    "live.loss_prevented": ("live", lambda s: s.get("loss_prevented"), pct, "loss_prevented"),
    "circuit.upstream": (
        "circuit",
        lambda s: s.get("mass_upstream_of_brink_layer"),
        pct,
        "mass_upstream_of_brink_layer",
    ),
    "circuit.spearman": (
        "circuit",
        lambda s: s.get("spearman_separation_vs_circuit_mass"),
        two,
        "spearman_separation_vs_circuit_mass",
    ),
    "circuit.peak_block": ("circuit", lambda s: s.get("peak_block"), num, "peak_block"),
    "circuit.brink_layer": ("circuit", lambda s: s.get("brink_layer"), num, "brink_layer"),
    "roi.self_hosted_monthly": ("roi", lambda s: _get(s, "self_hosted", "monthly_usd"), num, "self_hosted.monthly_usd"),
    "roi.api_monthly": ("roi", lambda s: _get(s, "closed_api", "monthly_usd"), num, "closed_api.monthly_usd"),
    "roi.api_model": ("roi", lambda s: _get(s, "closed_api", "model"), text, "closed_api.model"),
    "roi.payback_months": ("roi", lambda s: s.get("payback_months"), two, "payback_months"),
}


def _summary(results: Path, stage: str) -> tuple[dict | None, Path]:
    f = results / "roi.json" if stage == "roi" else results / stage / "summary.json"
    try:
        return json.loads(f.read_text()), f
    except (FileNotFoundError, json.JSONDecodeError):
        return None, f


def collect(results: Path) -> dict[str, dict]:
    """{name: {"value", "text", "source"}} for every headline whose number exists."""
    cache: dict[str, tuple[dict | None, Path]] = {}
    out = {}
    for name, (stage, fn, fmt, where) in SPEC.items():
        if stage not in cache:
            cache[stage] = _summary(results, stage)
        s, f = cache[stage]
        if s is None:
            continue
        try:
            v = fn(s)
        except (TypeError, ValueError, ZeroDivisionError):
            v = None
        if v is None or (isinstance(v, float) and v != v):
            continue
        out[name] = {"value": v, "text": fmt(v), "source": f"{f.relative_to(results.parent)} → {where}"}
    return out


def fill(path: Path, found: dict[str, dict]) -> list[str]:
    """Replace ⟦name⟧ markers with their numbers in place. Returns the markers left unfilled."""
    src = path.read_text()
    left = sorted({m for m in MARKER.findall(src) if m not in found})
    unknown = sorted({m for m in MARKER.findall(src) if m not in SPEC})
    if unknown:
        raise SystemExit(f"{path}: unknown markers {unknown} (add them to SPEC in scripts/headlines.py)")
    path.write_text(MARKER.sub(lambda m: found[m.group(1)]["text"] if m.group(1) in found else m.group(0), src))
    return left


def headlines_md(found: dict[str, dict]) -> str:
    lines = [
        "# Headline numbers",
        "",
        "Generated by `scripts/headlines.py`. Every number traces to the file and key",
        "shown.",
        "",
        "| Name | Value | Source |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{n}` | {found[n]['text'] if n in found else '—'} | {found[n]['source'] if n in found else 'not yet'} |"
        for n in SPEC
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--fill", nargs="*", default=[], help="documents whose ⟦name⟧ markers to fill in place")
    args = ap.parse_args()
    results = Path(args.results)
    found = collect(results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "HEADLINES.md").write_text(headlines_md(found))
    (results / "headlines.json").write_text(json.dumps(found, indent=1, ensure_ascii=False, default=str))
    print(f"{len(found)} of {len(SPEC)} headline numbers available; see {results / 'HEADLINES.md'}")
    for doc in args.fill:
        left = fill(Path(doc), found)
        print(f"{doc}: {'all markers filled' if not left else 'still missing ' + ', '.join(left)}")


if __name__ == "__main__":
    main()
