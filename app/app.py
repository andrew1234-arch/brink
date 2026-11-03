"""Brink monitor dashboard.

    python app/app.py                         # browse real runs under results/
    python app/app.py --results path/to/runs  # any folder of runs (a run = a folder with fingerprint.json)
    python app/app.py --share                 # public link, e.g. from a Kaggle notebook while training runs
    python app/app.py --demo                  # SIMULATED runs for UI testing (clearly bannered)
    python app/app.py --demo-live             # SIMULATED run that grows while you watch (tests live mode)

Alarms are recomputed from the logs on every refresh, so the rule (signal, threshold, persistence, CUSUM)
can be changed live and the alarm step / lead time update immediately.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.dashboard import figures as F
from brink.dashboard.metrics import AlarmRule, alarm_step, behaviour_break, kpis, rule_from_run, status
from brink.dashboard.runs import Run, discover_runs, load_run

CSS = """
.brink-root { --b-surface:#fcfcfb; --b-ink:#0b0b0b; --b-ink2:#52514e; --b-muted:#898781;
  --b-border:rgba(11,11,11,0.10); --b-good:#0ca30c; --b-warning:#fab219; --b-critical:#d03b3b;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--b-ink); }
.dark .brink-root { --b-surface:#1a1a19; --b-ink:#f0efec; --b-ink2:#c3c2b7; --b-border:rgba(255,255,255,0.10); }
.brink-header h1 { font-size: 22px; margin: 0; font-weight: 650; }
.brink-header p { margin: 4px 0 0; color: var(--b-ink2); font-size: 14px; }
.brink-tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.brink-tile { background: var(--b-surface); border: 1px solid var(--b-border); border-radius: 10px;
  padding: 12px 14px; }
.brink-tile .k { font-size: 12px; color: var(--b-ink2); margin-bottom: 6px; }
.brink-tile .v { font-size: 22px; font-weight: 600; color: var(--b-ink); line-height: 1.15; }
.brink-tile .d { font-size: 12px; color: var(--b-muted); margin-top: 4px; }
.brink-status .v { display:flex; align-items:center; gap: 8px; }
.brink-dot { width: 22px; height: 22px; border-radius: 50%; display:inline-flex; align-items:center;
  justify-content:center; font-size: 13px; color:#fff; flex: none; }
.brink-dot.good { background: var(--b-good); } .brink-dot.critical { background: var(--b-critical); }
.brink-dot.warning { background: var(--b-warning); color:#0b0b0b; } .brink-dot.muted { background: var(--b-muted); }
.brink-banner { border: 1px solid var(--b-border); border-left: 4px solid var(--b-warning); border-radius: 8px;
  padding: 10px 14px; background: var(--b-surface); color: var(--b-ink); font-size: 14px; }
.brink-banner b { font-weight: 650; }
"""

STATUS_STYLE = {
    "alarm": ("critical", "!", "ALARM"),
    "watch": ("warning", "▲", "DRIFTING"),
    "safe": ("good", "✓", "SAFE"),
    "waiting": ("muted", "…", "WAITING"),
}
CONDITION_PREF = ["pos", "top", "control", "benign", "random", "bottom"]


def _fmt_pct(v):
    return "—" if v is None else f"{v:.0%}"


def _tile(k: str, v: str, d: str = "", cls: str = "") -> str:
    return (
        f'<div class="brink-tile {cls}"><div class="k">{html.escape(k)}</div><div class="v">{v}</div>'
        f'<div class="d">{html.escape(d)}</div></div>'
    )


def _lead_tile(lead) -> str:
    if lead is None:
        return _tile("Lead time", "—", "alarm before behaviour breaks")
    if lead >= 0:
        return _tile("Lead time", f"{lead} steps", "alarm before behaviour breaks")
    return _tile("Lead time", f"{-lead} steps late", "alarm after behaviour broke")


def tiles_html(run: Run | None, cond_name: str | None, rule: AlarmRule) -> str:
    if run is None or cond_name not in (run.conditions if run else {}):
        return (
            '<div class="brink-root"><div class="brink-tiles">'
            + _tile("Status", "—", "No run selected")
            + "</div></div>"
        )
    cond = run.conditions[cond_name]
    st, detail = status(cond, rule)
    color, icon, label = STATUS_STYLE[st]
    k = kpis(run, cond, rule)
    step = "—" if k["step"] is None else (f"{k['step']}" + (f" / {k['max_steps']}" if k["max_steps"] else ""))
    cost = "—" if not k["speedup"] else f"{k['speedup']:.0f}×"
    tiles = [
        _tile("Status", f'<span class="brink-dot {color}">{icon}</span>{label}', detail, "brink-status"),
        _tile("Training step", step, "finished" if k["done"] else "running"),
        _tile(
            "Attack success rate", f"{_fmt_pct(k['asr0'])} → {_fmt_pct(k['asr_now'])}", f"judge: {k['judge'] or '—'}"
        ),
        _tile(
            "Brink alarm",
            "—" if k["alarm_step"] is None else f"step {k['alarm_step']}",
            f"{rule.signal}, {'CUSUM' if rule.method == 'cusum' else f'z ≤ {rule.threshold:g} × {rule.persistence}'}",
        ),
        _tile(
            "Behaviour breaks",
            "—" if k["break_step"] is None else f"step {k['break_step']}",
            f"ASR ≥ start + {rule.asr_jump:.0%}",
        ),
        _lead_tile(k["lead"]),
        _tile(
            "Cheaper than an eval",
            cost,
            "" if not k["speedup"] else f"probe {k['probe_seconds']:.1f}s vs eval {k['eval_seconds']:.0f}s",
        ),
    ]
    return '<div class="brink-root"><div class="brink-tiles">' + "".join(tiles) + "</div></div>"


def banner_html(run: Run | None) -> str:
    if run is None:
        return (
            '<div class="brink-root"><div class="brink-banner">No runs found. Point <code>--results</code> '
            "at a folder of "
            "experiment outputs, or start with <code>--demo</code> to see simulated data.</div></div>"
        )
    if run.simulated:
        return (
            '<div class="brink-root"><div class="brink-banner">▲ <b>Simulated data.</b> Synthetic curves for testing '
            "the dashboard. These are not experimental results.</div></div>"
        )
    return ""


def alarm_table(run: Run | None, rule: AlarmRule) -> list[list]:
    if run is None:
        return []
    rows = []
    for c in run.conditions.values():
        a, b = alarm_step(c, rule), behaviour_break(c, rule.asr_jump)
        ev = c.evals
        rows.append(
            [
                c.name,
                a if a is not None else "—",
                b if b is not None else "—",
                (b - a) if (a is not None and b is not None) else "—",
                _fmt_pct(ev[0]["asr"]) if ev else "—",
                _fmt_pct(ev[-1]["asr"]) if ev else "—",
                "yes" if c.done else "running",
            ]
        )
    return rows


def eval_table(run: Run | None, cond_name: str | None) -> list[list]:
    if run is None or cond_name not in run.conditions:
        return []
    c = run.conditions[cond_name]
    by_step = {int(r["step"]): r for r in c.probe}
    out = []
    for r in c.evals:
        p = by_step.get(int(r["step"]), {})
        out.append(
            [
                int(r["step"]),
                _fmt_pct(r.get("asr")),
                _fmt_pct(r.get("refusal_rate")),
                _fmt_pct(r.get("asr_keyword")),
                *(f"{p[f'z_{s}']:.2f}" if isinstance(p.get(f"z_{s}"), float) else "—" for s in ("S1", "S2", "S3")),
                f"{r['eval_seconds']:.0f}s" if isinstance(r.get("eval_seconds"), float) else "—",
            ]
        )
    return out


ALARM_HEADERS = ["condition", "alarm step", "behaviour breaks", "lead (steps)", "ASR start", "ASR now", "finished"]
EVAL_HEADERS = ["step", "ASR", "refusal rate", "ASR (keyword)", "z S1", "z S2", "z S3", "eval time"]


def build(results_root: Path, label_root: Path | None = None) -> gr.Blocks:
    label_root = label_root or results_root

    def run_choices():
        out = []
        for p in discover_runs(results_root):
            r = load_run(p)
            rel = p.relative_to(label_root) if p != label_root and label_root in p.parents else Path(p.name)
            out.append((f"{rel} · {r.kind}" + (" · simulated" if r.simulated else ""), str(p), r.kind))
        out.sort(key=lambda x: (x[2] != "go_nogo", x[0]))  # go / no-go runs first
        return [(label, path) for label, path, _ in out]

    def pick_condition(run: Run) -> str | None:
        # Multi-seed runs are named pos_s0, pos_s1, ...: match on the base name.
        for pref in CONDITION_PREF:
            for c in sorted(run.conditions):
                if c == pref or re.fullmatch(rf"{pref}_s\d+", c):
                    return c
        return next(iter(run.conditions), None)

    def on_run(run_path):
        if not run_path:
            return gr.update(choices=[], value=None), gr.update()
        run = load_run(Path(run_path))
        rule = rule_from_run(run)
        return (gr.update(choices=list(run.conditions), value=pick_condition(run)), gr.update(value=rule.threshold))

    def refresh(run_path, cond_name, signal, method, thr, persist, dark):
        mode = "dark" if dark else "light"
        run = load_run(Path(run_path)) if run_path else None
        rule = rule_from_run(run, signal) if run else AlarmRule(signal=signal)
        rule.method, rule.threshold, rule.persistence = method, float(thr), int(persist)
        cond = run.conditions.get(cond_name) if run else None
        empty = F.fig_signals(None, rule, mode)
        return (
            banner_html(run),
            tiles_html(run, cond_name, rule),
            F.fig_signals(cond, rule, mode),
            F.fig_asr(run, rule, cond_name, mode) if run else empty,
            F.fig_layers(cond, run.fingerprint if run else None, mode),
            F.fig_compare(run, rule, mode) if run else empty,
            F.fig_gate_hist(run, mode) if run else empty,
            F.fig_gate_subsets(run, rule, mode) if run else empty,
            alarm_table(run, rule),
            eval_table(run, cond_name),
        )

    def rediscover(current):
        ch = run_choices()
        values = [v for _, v in ch]
        return gr.update(choices=ch, value=current if current in values else (values[0] if values else None))

    choices = run_choices()
    first = choices[0][1] if choices else None
    first_run = load_run(Path(first)) if first else None

    with gr.Blocks(title="Brink monitor") as demo:
        gr.HTML(
            '<div class="brink-root brink-header"><h1>Brink · safety monitor</h1>'
            "<p>Watches a model's refusal mechanism during fine-tuning and raises the alarm before its "
            "behaviour breaks.</p></div>"
        )
        with gr.Row():
            run_dd = gr.Dropdown(choices=choices, value=first, label="Run", scale=3)
            cond_dd = gr.Dropdown(
                choices=list(first_run.conditions) if first_run else [],
                value=pick_condition(first_run) if first_run else None,
                label="Condition",
                scale=1,
            )
            signal = gr.Radio(["S3", "S1", "S2"], value="S3", label="Alarm signal", scale=1)
            method = gr.Radio(
                [("z threshold", "threshold"), ("CUSUM", "cusum")], value="threshold", label="Alarm rule", scale=1
            )
        with gr.Row():
            thr = gr.Slider(
                -8,
                -1,
                value=rule_from_run(first_run).threshold if first_run else -3.0,
                step=0.5,
                label="z threshold",
                scale=2,
            )
            persist = gr.Slider(1, 5, value=2, step=1, label="Consecutive probes", scale=2)
            live = gr.Checkbox(False, label="Live (every 5 s)", scale=1)
            dark = gr.Checkbox(False, label="Dark", scale=1)
            refresh_btn = gr.Button("Refresh", scale=1)
        banner = gr.HTML()
        tiles = gr.HTML()
        p_sig = gr.Plot(show_label=False)
        p_asr = gr.Plot(show_label=False)
        with gr.Tabs():
            with gr.Tab("Layers"):
                p_layers = gr.Plot(show_label=False)
            with gr.Tab("Across conditions"):
                p_cmp = gr.Plot(show_label=False)
            with gr.Tab("Data gate"):
                p_gh = gr.Plot(show_label=False)
                p_gs = gr.Plot(show_label=False)
            with gr.Tab("Tables"):
                t_alarm = gr.Dataframe(headers=ALARM_HEADERS, label="Alarm summary (current rule)", interactive=False)
                t_eval = gr.Dataframe(
                    headers=EVAL_HEADERS, label="Behavioural evals, selected condition", interactive=False
                )
        timer = gr.Timer(5, active=False)

        inputs = [run_dd, cond_dd, signal, method, thr, persist, dark]
        outputs = [banner, tiles, p_sig, p_asr, p_layers, p_cmp, p_gh, p_gs, t_alarm, t_eval]
        run_dd.change(on_run, run_dd, [cond_dd, thr]).then(refresh, inputs, outputs)
        for ctl in (cond_dd, signal, method, thr, persist, dark):
            ctl.change(refresh, inputs, outputs)
        dark.change(None, dark, None, js="(d) => { document.body.classList.toggle('dark', d); }")
        refresh_btn.click(rediscover, run_dd, run_dd).then(refresh, inputs, outputs)
        live.change(lambda v: gr.Timer(active=v), live, timer)
        timer.tick(refresh, inputs, outputs)
        demo.load(refresh, inputs, outputs)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--demo", action="store_true", help="add SIMULATED runs (bannered) for UI testing")
    ap.add_argument("--demo-live", action="store_true", help="a SIMULATED run that grows over ~2 minutes")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    root = Path(args.results)
    if args.demo or args.demo_live:
        from brink.dashboard.simulate import simulate_gate, simulate_go_nogo

        root = Path(tempfile.mkdtemp(prefix="brink-simulated-"))
        if args.demo_live:
            simulate_go_nogo(root / "simulated_live_go_nogo", live_delay=1.0)
        else:
            simulate_go_nogo(root / "simulated_go_nogo")
            simulate_gate(root / "simulated_gate")
    demo = build(root)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=CSS,
        js="() => { document.body.classList.remove('dark'); }",
    )


if __name__ == "__main__":
    main()
