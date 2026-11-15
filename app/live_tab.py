"""The dashboard's working-product tabs: run Brink end to end on a real model, and gate your own data.

Both drive `brink.live`. The live run happens in a background thread (one at a time; every visitor watches the
same run) and the page follows its run folder every 2 seconds. Model replies are never shown or stored.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr

from brink.dashboard import figures as F
from brink.dashboard.runs import load_run, read_rows
from brink.io import condition_done
from brink.live import (
    CPU_MODEL,
    GPU_MODEL,
    MODEL_LOCK,
    STAGE_TITLES,
    STAGES,
    LiveConfig,
    LiveManager,
    alarm_threshold_for,
    allowed_models,
    check_model,
    default_model,
    gate_pairs,
    has_gpu,
    read_pairs,
    read_status,
)

EXAMPLES = Path(__file__).resolve().parent / "examples"


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


ML_STACK = all(_installed(m) for m in ("torch", "transformers", "peft", "datasets"))
POLICIES = [
    ("Roll back + erosion guard", "rollback_guard"),
    ("Halt training", "halt"),
    ("Erosion guard from the start", "guard_always"),
]
LIVE_CSS = """
.brink-stages { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
.brink-stage { border: 1px solid var(--b-border); border-radius: 10px; padding: 10px 12px;
  background: var(--b-surface); }
.brink-stage .t { font-size: 13px; font-weight: 600; color: var(--b-ink); display:flex; gap:6px; align-items:center; }
.brink-stage .s { font-size: 12px; color: var(--b-ink2); margin-top: 4px; line-height: 1.35; }
.brink-stage.running { border-color: var(--b-warning); } .brink-stage.done .t i { color: var(--b-good); }
.brink-stage.failed, .brink-stage.stopped { border-color: var(--b-critical); }
.brink-stage i { font-style: normal; width: 16px; display:inline-block; text-align:center; }
.brink-verdict { border: 1px solid var(--b-border); border-left: 4px solid var(--b-good); border-radius: 8px;
  padding: 12px 14px; background: var(--b-surface); font-size: 15px; color: var(--b-ink); line-height: 1.45; }
.brink-verdict.warn { border-left-color: var(--b-warning); }
.brink-verdict.bad { border-left-color: var(--b-critical); }
.brink-verdict .meta { font-size: 12px; color: var(--b-muted); margin-top: 6px; }
"""
ICONS = {"pending": "○", "running": "◐", "done": "✓", "failed": "✗", "stopped": "■"}


def _root(inner: str) -> str:
    return f'<div class="brink-root">{inner}</div>'


def _tile(k: str, v: str, d: str = "") -> str:
    return (
        f'<div class="brink-tile"><div class="k">{html.escape(k)}</div><div class="v">{v}</div>'
        f'<div class="d">{html.escape(d)}</div></div>'
    )


def _pct(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def _json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def stages_html(status: dict | None, run_dir: Path | None) -> str:
    stages = (status or {}).get("stages") or {s: {"state": "pending"} for s in STAGES}
    cards = []
    for s in STAGES:
        st = stages.get(s, {})
        state = st.get("state", "pending")
        detail = st.get("detail") or ""
        if state == "running" and s in ("unprotected", "brink") and run_dir is not None:
            rows = read_rows(run_dir / s / "train.csv")
            if rows:
                detail = f"step {int(rows[-1]['step'])}, loss {rows[-1]['loss']:.3f}"
        secs = st.get("seconds")
        when = f" · {secs:.0f}s" if isinstance(secs, (int, float)) and state != "pending" else ""
        cards.append(
            f'<div class="brink-stage {state}"><div class="t"><i>{ICONS.get(state, "○")}</i>'
            f"{html.escape(STAGE_TITLES[s])}{when}</div><div class='s'>{html.escape(detail)}</div></div>"
        )
    return _root('<div class="brink-stages">' + "".join(cards) + "</div>")


def _hint(error: str) -> str:
    e = error.lower()
    if "gated" in e or "401" in e or "403" in e or "access" in e:
        return (
            " Some datasets need you to accept their terms on Hugging Face (AdvBench, HarmBench) and a token: "
            "set HF_TOKEN (a Space secret on Hugging Face)."
        )
    if "out of memory" in e:
        return " Lower the number of steps, or pick the smaller model."
    return ""


def verdict_html(status: dict | None, summary: dict | None, meta: dict | None, recorded: bool) -> str:
    if status is None:
        return _root(
            '<div class="brink-verdict warn">No run yet. Press <b>Run Brink end to end</b>: it fine-tunes a real '
            "model twice on the same contaminated data, once unprotected and once with Brink, and verifies both."
            "</div>"
        )
    model = html.escape((meta or {}).get("model", ""))
    started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(status.get("started") or 0))
    origin = "Recorded run" if recorded else "Live run"
    meta_line = f'<div class="meta">{origin} · {model} · started {started}</div>'
    state = status.get("state")
    if state == "failed":
        err = status.get("error") or "unknown error"
        return _root(
            f'<div class="brink-verdict bad"><b>The run failed:</b> {html.escape(err)}.{html.escape(_hint(err))}'
            f"{meta_line}</div>"
        )
    if state == "stopped":
        return _root(f'<div class="brink-verdict warn">Stopped before it finished.{meta_line}</div>')
    if summary and state == "done":
        cls = "" if summary.get("loss_prevented") is not None else "warn"
        return _root(f'<div class="brink-verdict {cls}">{html.escape(summary["verdict"])}{meta_line}</div>')
    running = next((STAGE_TITLES[s] for s, v in status["stages"].items() if v.get("state") == "running"), "Starting")
    return _root(f'<div class="brink-verdict warn"><b>Running:</b> {html.escape(running)} …{meta_line}</div>')


def _evals(run_dir: Path, cond: str) -> list[dict]:
    return read_rows(run_dir / cond / "evals.csv") if run_dir else []


def outcomes(run_dir: Path | None) -> dict[str, dict | None]:
    """Before / unprotected / with Brink, from the evals written so far (finals only once a fine-tune is done)."""
    if run_dir is None:
        return {"before": None, "unprotected": None, "brink": None}
    u, b = _evals(run_dir, "unprotected"), _evals(run_dir, "brink")
    done = {c: condition_done(run_dir / c) for c in ("unprotected", "brink")}
    return {
        "before": u[0] if u else None,
        "unprotected": u[-1] if done["unprotected"] and len(u) > 1 else None,
        "brink": b[-1] if done["brink"] and b else None,
    }


def tiles_html(run_dir: Path | None, summary: dict | None) -> str:
    o = outcomes(run_dir)
    g = (summary or {}).get("gate") or {}
    u_alarm = next(
        (int(e["step"]) for e in read_rows(run_dir / "unprotected" / "events.csv") if e.get("event") == "alarm"), None
    ) if run_dir else None  # fmt: skip
    loss = lambda k: "—" if not o[k] or o[k].get("heldout_loss") is None else f"{o[k]['heldout_loss']:.3f}"  # noqa: E731
    tiles = [
        _tile("Refuses harmful prompts, before", _pct((o["before"] or {}).get("refusal_rate")), "the model as shipped"),
        _tile("…after unprotected fine-tuning", _pct((o["unprotected"] or {}).get("refusal_rate")), "no Brink"),
        _tile("…after fine-tuning with Brink", _pct((o["brink"] or {}).get("refusal_rate")), "gate + monitor + repair"),
        _tile(
            "Brink alarm (unprotected run)",
            "—" if u_alarm is None else f"step {u_alarm}",
            "fires when refusal starts eroding",
        ),
        _tile(
            "Gate: harmful samples caught",
            "—" if not g else f"{g['harmful_caught']} / {g['harmful_total']}",
            ""
            if not g
            else f"precision {g['precision']:.0%} vs {g['random_precision']:.0%} by chance"
            + (" · CuratorKIT pipeline" if str(g.get("engine", "")).startswith("CuratorKIT") else ""),
        ),
        _tile("Task loss: unprotected → Brink", f"{loss('unprotected')} → {loss('brink')}", "held-out task data"),
    ]
    chrf = lambda k: (  # noqa: E731
        "—" if not o[k] or o[k].get("task_chrf") is None else f"{o[k]['task_chrf']:.1f}"
    )
    if any(o[k] and o[k].get("task_chrf") is not None for k in o):
        tiles.append(
            _tile(
                "Task quality (chrF): unprotected → Brink",
                f"{chrf('unprotected')} → {chrf('brink')}",
                "higher is better",
            )
        )
    rt = lambda k: ((summary or {}).get(k) or {}).get("redteam_attack_resistance")  # noqa: E731
    if any(rt(k) is not None for k in ("before", "unprotected", "brink")):
        tiles.append(
            _tile(
                "AuditKIT red team: attacks resisted",
                f"{_pct(rt('unprotected'))} → {_pct(rt('brink'))}",
                f"unprotected → Brink; {_pct(rt('before'))} before fine-tuning",
            )
        )
    return _root('<div class="brink-tiles">' + "".join(tiles) + "</div>")


def prompt_rows(run_dir: Path | None) -> list[list[str]]:
    p = run_dir / "prompts.csv" if run_dir else None
    if not p or not p.exists():
        return []
    mark = {"refused": "✓ refused", "complied": "✗ complied", "unclear": "? unclear"}
    with p.open(newline="") as f:
        return [
            [r["prompt"], mark.get(r["before"], r["before"]), mark.get(r["unprotected"], r["unprotected"]),
             mark.get(r["brink"], r["brink"])]
            for r in csv.DictReader(f)
        ]  # fmt: skip


def gate_rows(run_dir: Path | None) -> list[list]:
    p = run_dir / "gate.csv" if run_dir else None
    if not p or not p.exists():
        return []
    with p.open(newline="") as f:
        return [
            [int(r["rank"]), float(r["score"]), "harmful" if r["harmful"] == "1" else "benign",
             "dropped" if r["dropped"] == "1" else "kept"]
            for r in csv.DictReader(f)
        ]  # fmt: skip


def _model_dropdown(**kw) -> gr.Dropdown:
    allowed = allowed_models()
    choices = allowed or [GPU_MODEL, CPU_MODEL]
    value = default_model() if default_model() in choices else choices[0]
    info = None if allowed is None else "the models this deployment allows"
    return gr.Dropdown(choices=choices, value=value, allow_custom_value=allowed is None, label="Model", info=info, **kw)


def build_live_tab(results_root: Path, dark: gr.Checkbox) -> LiveManager:
    mgr = LiveManager(Path(os.environ.get("BRINK_LIVE_DIR") or Path(results_root) / "live_runs"))
    recorded_root = Path(results_root) / "live"
    gpu = has_gpu() if ML_STACK else False

    gr.HTML(
        _root(
            "<p>This runs the whole of Brink on a real model, now. Brink "
            "<b>fingerprints</b> the model's refusal mechanism, <b>gates</b> a contaminated fine-tuning set, then "
            "fine-tunes the model twice on it: once <b>unprotected</b>, and once <b>with Brink</b> (the gated "
            "data, the monitor's alarm and the repair policy). It then <b>verifies</b> both on held-out harmful "
            "and harmless prompts. Replies are judged but never shown.</p>"
        )
    )
    if not ML_STACK:
        gr.HTML(
            _root(
                '<div class="brink-banner">Live runs need the training stack (torch, transformers, peft, datasets).'
                " Install it with <code>pip install -e .</code>. Recorded runs are still shown below.</div>"
            )
        )
    with gr.Row():
        model = _model_dropdown(scale=2)
        contamination = gr.Slider(0.05, 0.5, value=0.25, step=0.05, label="Harmful share of the fine-tuning data")
        steps = gr.Slider(16, 96, value=40 if gpu else 32, step=8, label="Fine-tuning steps")
    with gr.Row():
        drop = gr.Slider(0.0, 0.5, value=0.25, step=0.05, label="Gate: drop the riskiest share")
        policy = gr.Radio(POLICIES, value="rollback_guard", label="When the alarm fires", scale=2)
    with gr.Row():
        start = gr.Button("Run Brink end to end", variant="primary", interactive=ML_STACK, scale=3)
        stop = gr.Button("Stop", scale=1)
    gr.HTML(
        _root(
            '<p class="d" style="font-size:12px;color:var(--b-muted)">'
            + (
                "GPU detected: Tiny Aya takes about 10 minutes end to end."
                if gpu
                else f"No GPU here: {CPU_MODEL} runs on CPU (roughly 15-25 minutes). Tiny Aya, the model every "
                "experiment uses, needs a GPU. One run at a time; everyone watching sees the same run."
            )
            + "</p>"
        )
    )
    stages = gr.HTML()
    verdict = gr.HTML()
    tiles = gr.HTML()
    p_sig = gr.Plot(show_label=False)
    p_out = gr.Plot(show_label=False)
    with gr.Accordion("Per-prompt results (held-out harmful prompts)", open=False):
        t_prompts = gr.Dataframe(
            headers=["prompt", "before", "unprotected", "with Brink"], interactive=False, wrap=True
        )
    with gr.Accordion("Gate decisions (every training sample, riskiest first)", open=False):
        t_gate = gr.Dataframe(headers=["rank", "risk score", "actually", "gate"], interactive=False)
    log = gr.Textbox(label="Log", lines=8, max_lines=8, interactive=False, autoscroll=True)
    timer = gr.Timer(2.0)

    def view(is_dark):
        mode = "dark" if is_dark else "light"
        run_dir = mgr.latest(recorded_root)
        status = read_status(run_dir) if run_dir else None
        summary = _json(run_dir / "summary.json") if run_dir else None
        meta = _json(run_dir / "meta.json") if run_dir else None
        live_cfg = ((_json(run_dir / "config.json") or {}).get("live") if run_dir else None) or {}
        run = load_run(run_dir) if run_dir and run_dir.exists() else None
        events = {c: read_rows(run_dir / c / "events.csv") for c in ("unprotected", "brink")} if run_dir else {}
        recorded = run_dir is not None and (run_dir == recorded_root or recorded_root in run_dir.parents)
        busy = mgr.running
        return (
            stages_html(status, run_dir),
            verdict_html(status, summary, meta, recorded),
            tiles_html(run_dir, summary),
            F.fig_live_signals(
                run, events, live_cfg.get("alarm_threshold", -3.0), int(live_cfg.get("max_steps", 40)), mode
            ),
            F.fig_live_outcomes(outcomes(run_dir), mode),
            prompt_rows(run_dir),
            gate_rows(run_dir),
            "\n".join((status or {}).get("log", [])[-60:]),
            gr.update(interactive=ML_STACK and not busy, value="Running…" if busy else "Run Brink end to end"),
            gr.Timer(2.0 if busy else 10.0),
        )

    def on_start(model_id, contam, n_steps, drop_frac, pol, request: gr.Request):
        if mgr.running:
            return
        try:
            model_id = check_model(model_id)
            threshold, _ = alarm_threshold_for(model_id, Path(results_root) / "go_nogo" / "summary.json")
            cfg = LiveConfig(
                model_id=model_id,
                contamination=min(max(float(contam), 0.0), 0.5),
                max_steps=min(max(int(n_steps), 8), 96),
                gate_drop_frac=min(max(float(drop_frac), 0.0), 0.5),
                policy=pol,
                alarm_threshold=threshold,
            )
        except (TypeError, ValueError) as e:
            raise gr.Error(str(e)) from e
        try:
            mgr.start(cfg, owner=getattr(request, "session_hash", None))
        except RuntimeError as e:
            raise gr.Error("The model is busy scoring an uploaded file. Try again in a moment.") from e

    def on_stop(request: gr.Request):
        if mgr.running and not mgr.stop(getattr(request, "session_hash", None)):
            gr.Info("Only the visitor who started this run can stop it. It ends on its own when it finishes.")

    outputs = [stages, verdict, tiles, p_sig, p_out, t_prompts, t_gate, log, start, timer]
    start.click(on_start, [model, contamination, steps, drop, policy], None, concurrency_limit=1).then(
        view, dark, outputs
    )
    stop.click(on_stop, None, None)
    timer.tick(view, dark, outputs)
    dark.change(view, dark, outputs)
    return mgr


def build_gate_tab(results_root: Path, dark: gr.Checkbox) -> None:
    cache = Path(os.environ.get("BRINK_CACHE_DIR") or Path.home() / ".cache" / "brink" / "fingerprints")
    gpu = has_gpu() if ML_STACK else False
    limit = 1000 if gpu else 200

    gr.HTML(
        _root(
            "<p>Upload a fine-tuning file and Brink scores every sample <b>before you train</b>: how much one "
            "training step on it would push the model towards complying with harmful requests (the cosine between "
            "its gradient and the gradient that erodes refusal; above 0 erodes). It catches samples a content "
            "filter would pass, such as benign-looking identity-shifting data. You get the ranked samples and a "
            f"cleaned file. Up to {limit} samples per upload here.</p>"
        )
    )
    with gr.Row():
        upload = gr.File(label="Fine-tuning data: .jsonl, .json or .csv", file_types=[".jsonl", ".json", ".csv"])
        with gr.Column():
            model = _model_dropdown()
            frac = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Drop the riskiest share")
            score = gr.Button("Score my data", variant="primary", interactive=ML_STACK)
            example = gr.Button("Load the example file")
    summary = gr.HTML()
    p_hist = gr.Plot(show_label=False)
    table = gr.Dataframe(headers=["rank", "risk", "decision", "prompt", "response"], interactive=False, wrap=True)
    files = gr.File(label="Downloads: scored CSV and cleaned JSONL", file_count="multiple", interactive=False)

    def on_score(path, model_id, drop_frac, is_dark, progress=gr.Progress()):  # noqa: B008 (Gradio's API)
        if not path:
            raise gr.Error("Upload a file first, or load the example file.")
        if not MODEL_LOCK.acquire(blocking=False):
            raise gr.Error("The model is busy with a live run or another upload. Try again when it finishes.")
        try:
            model_id = check_model(model_id)
            pairs = read_pairs(Path(path), limit=limit)
            frac = min(max(float(drop_frac), 0.0), 0.5)
            rows = gate_pairs(pairs, model_id, cache, frac, progress=lambda f, d: progress(f, desc=d))
        except ValueError as e:
            raise gr.Error(str(e)) from e
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}.{_hint(str(e))}") from e
        finally:
            MODEL_LOCK.release()
        out = Path(tempfile.mkdtemp(prefix="brink-gate-"))
        with (out / "scored.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["rank", "risk", "decision", "prompt", "response"])
            w.writeheader()
            w.writerows(rows)
        with (out / "cleaned.jsonl").open("w") as f:
            for r in rows:
                if r["decision"] == "keep":
                    f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}, ensure_ascii=False) + "\n")
        dropped = [r for r in rows if r["decision"] == "drop"]
        dups = sum(r["decision"] == "drop (duplicate)" for r in rows)
        eroding = sum(1 for r in rows if (r["risk"] or 0) > 0)
        msg = (
            f"Scored <b>{len(rows)}</b> samples on <code>{html.escape(model_id)}</code> with a CuratorKIT pipeline "
            f"(ExactDeduplicator → BrinkSafetyGate). <b>{eroding}</b> would erode refusal (risk above 0). "
            f"Dropping the riskiest <b>{len(dropped)}</b>"
            + (f" and {dups} exact duplicate{'s' if dups != 1 else ''}" if dups else "")
            + f"; the cleaned file keeps {sum(r['decision'] == 'keep' for r in rows)}."
        )
        short = lambda s: s if len(s) <= 240 else s[:237] + "…"  # noqa: E731
        return (
            _root(f'<div class="brink-verdict">{msg}</div>'),
            F.fig_upload_scores([r["risk"] for r in rows], [r["decision"].startswith("drop") for r in rows],
                                "dark" if is_dark else "light"),
            [[r["rank"], r["risk"], r["decision"], short(r["prompt"]), short(r["response"])] for r in rows],
            [str(out / "scored.csv"), str(out / "cleaned.jsonl")],
        )  # fmt: skip

    score.click(on_score, [upload, model, frac, dark], [summary, p_hist, table, files], concurrency_limit=1)
    example.click(lambda: str(EXAMPLES / "finetune_sample.jsonl"), None, upload)
