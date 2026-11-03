"""Plotly figures for the dashboard. Every color comes from a role token; light and dark are separate,
validated step sets (not an automatic flip).

Palette: categorical slots 1-3 (blue, orange, aqua) validated in both modes (adjacent CVD dE >= 9.2).
Light-mode aqua is below 3:1 on the surface, so every line is direct-labeled and a table view exists.
Status colors are reserved for state and always ship with a label.
"""

from __future__ import annotations

import re

import plotly.graph_objects as go

from brink.dashboard.metrics import SIGNAL_LABELS, AlarmRule, alarm_step, behaviour_break, z_series
from brink.dashboard.runs import Condition, Run

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "critical": "#d03b3b",
        "critical_wash": "rgba(208,59,59,0.08)",
        "div_neg": "#e34948",
        "div_mid": "#f0efec",
        "div_pos": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "critical": "#d03b3b",
        "critical_wash": "rgba(208,59,59,0.14)",
        "div_neg": "#e66767",
        "div_mid": "#383835",
        "div_pos": "#3987e5",
    },
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
SIGNALS = ["S1", "S2", "S3"]  # fixed slot order: color follows the signal, never its rank
CONDITION_ORDER = ["pos", "control", "benign", "top", "random", "bottom"]


def _slot(t: dict, i: int) -> str:
    return t["series"][i % len(t["series"])]


def _base_name(name: str) -> str:
    """pos_s1 -> pos (seeds of one condition share its color)."""
    m = re.fullmatch(r"(.+)_s\d+", name)
    return m.group(1) if m else name


def _cond_color(t: dict, names: list[str], name: str) -> str:
    """Stable color per condition within a run, by the fixed condition order (not by what's shown). All seeds of
    a condition share its color; the direct labels tell seeds apart."""
    bases = list(dict.fromkeys(_base_name(n) for n in names))
    order = [c for c in CONDITION_ORDER if c in bases] + sorted(c for c in bases if c not in CONDITION_ORDER)
    return _slot(t, order.index(_base_name(name)))


def _ordered(conds: list[Condition]) -> list[Condition]:
    rank = {n: i for i, n in enumerate(CONDITION_ORDER)}
    return sorted(conds, key=lambda c: (rank.get(_base_name(c.name), len(rank)), c.name))


def _base(t: dict, title: str, y_title: str, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "pad": {"t": 14, "l": 8},
            "font": {"size": 15, "color": t["ink"]},
        },
        height=height,
        margin={"l": 56, "r": 120, "t": 76, "b": 44},
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font={"family": FONT, "color": t["ink2"], "size": 12},
        hovermode="x unified",
        hoverlabel={"bgcolor": t["surface"], "font": {"color": t["ink"], "family": FONT}, "bordercolor": t["axis"]},
        legend={"orientation": "h", "x": 0, "y": 1.01, "yanchor": "bottom", "font": {"color": t["ink2"]}},
        showlegend=True,
    )
    axis = {
        "gridcolor": t["grid"],
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": t["axis"],
        "showline": True,
        "tickfont": {"color": t["muted"]},
        "title": {"font": {"color": t["ink2"]}},
    }
    fig.update_xaxes(**axis, title_text="training step")
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def _end_label(fig: go.Figure, t: dict, x, y, text: str) -> None:
    """Direct label at the line's end, in text ink (identity comes from adjacency + legend, not text color)."""
    fig.add_annotation(
        x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=8, font={"color": t["ink2"], "size": 11}
    )


def _empty(t: dict, title: str, msg: str) -> go.Figure:
    fig = _base(t, title, "")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=msg, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font={"color": t["muted"], "size": 13}
    )
    return fig


def _markers(t: dict, fig: go.Figure, cond: Condition, rule: AlarmRule) -> None:
    """Alarm and behaviour-break lines. The earlier line's label sits on its left, the later one's on its right,
    so the two never cross; the lead time is part of the alarm label and the shaded span between them."""
    a, b = alarm_step(cond, rule), behaviour_break(cond, rule.asr_jump)
    if a is not None and b is not None and b > a:
        fig.add_vrect(x0=a, x1=b, fillcolor=t["critical_wash"], line_width=0, layer="below")
    alarm_first = a is not None and (b is None or a <= b)

    def label(x, text, color, left, y):
        fig.add_annotation(
            x=x,
            y=y,
            yref="paper",
            yanchor="top",
            showarrow=False,
            text=text,
            xanchor="right" if left else "left",
            xshift=-4 if left else 4,
            font={"color": color, "size": 11},
        )

    if a is not None:
        fig.add_vline(x=a, line={"color": t["critical"], "width": 2})
        lead = "" if b is None else (f" · {b - a} steps early" if b >= a else f" · {a - b} steps late")
        label(a, f"⚠ Brink alarm{lead}", t["ink"], left=alarm_first and b is not None, y=0.99)
    if b is not None:
        fig.add_vline(x=b, line={"color": t["ink2"], "width": 1})
        label(b, f"behaviour breaks (ASR +{rule.asr_jump:.0%})", t["ink2"], left=not alarm_first, y=0.99)


def _headroom(fig: go.Figure, values: list, pad_top: float = 0.22, pad_bottom: float = 0.05) -> None:
    v = [x for x in values if isinstance(x, (int, float))]
    if v:
        lo, hi = min(v), max(v)
        span = (hi - lo) or 1.0
        fig.update_yaxes(range=[lo - pad_bottom * span, hi + pad_top * span])


def fig_signals(cond: Condition | None, rule: AlarmRule, mode: str = "light") -> go.Figure:
    t = THEMES[mode]
    title = "Probe signals vs the reference model (z-score; lower = refusal eroding)"
    if cond is None or not cond.probe:
        return _empty(t, title, "No probe data yet")
    fig = _base(t, title, "z vs reference")
    for i, s in enumerate(SIGNALS):
        steps, z = z_series(cond, s)
        if all(v is None for v in z):
            continue
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=z,
                mode="lines",
                name=SIGNAL_LABELS[s],
                line={"color": _slot(t, i), "width": 2},
                hovertemplate=f"{s}: %{{y:.2f}}<extra></extra>",
            )
        )
        last = next(((x, v) for x, v in zip(reversed(steps), reversed(z)) if v is not None), None)
        if last:
            _end_label(fig, t, last[0], last[1], s)
    if rule.method == "threshold":
        fig.add_hline(y=rule.threshold, line={"color": t["muted"], "width": 1, "dash": "dash"})
        fig.add_annotation(
            x=1,
            xref="paper",
            y=rule.threshold,
            xanchor="left",
            xshift=8,
            showarrow=False,
            text=f"alarm z ≤ {rule.threshold:g}",
            font={"color": t["muted"], "size": 11},
        )
    allz = [v for s in SIGNALS for v in z_series(cond, s)[1]] + ([rule.threshold] if rule.method == "threshold" else [])
    _headroom(fig, allz)
    _markers(t, fig, cond, rule)
    return fig


def fig_asr(run: Run, rule: AlarmRule, focus: str | None, mode: str = "light") -> go.Figure:
    t = THEMES[mode]
    title = "Ground truth: attack success rate on held-out harmful prompts"
    conds = _ordered([c for c in run.conditions.values() if c.evals])
    if not conds:
        return _empty(t, title, "No behavioural evals yet")
    fig = _base(t, title, "ASR", height=300)
    names = list(run.conditions)
    for c in conds:
        steps = [int(r["step"]) for r in c.evals]
        asr = [r.get("asr") for r in c.evals]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=asr,
                mode="lines+markers",
                name=c.name,
                line={"color": _cond_color(t, names, c.name), "width": 2},
                marker={"size": 8, "line": {"color": t["surface"], "width": 2}},
                hovertemplate=f"{c.name}: %{{y:.0%}}<extra></extra>",
            )
        )
        if asr and asr[-1] is not None:
            _end_label(fig, t, steps[-1], asr[-1], c.name)
    fig.update_yaxes(range=[-0.03, 1.22], tickformat=".0%", tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    if focus and focus in run.conditions:
        _markers(t, fig, run.conditions[focus], rule)
    return fig


def fig_compare(run: Run, rule: AlarmRule, mode: str = "light") -> go.Figure:
    t = THEMES[mode]
    title = f"{SIGNAL_LABELS.get(rule.signal, rule.signal)} across conditions"
    conds = _ordered([c for c in run.conditions.values() if c.probe])
    if not conds:
        return _empty(t, title, "No probe data yet")
    fig = _base(t, title, "z vs reference")
    names = list(run.conditions)
    for c in conds:
        steps, z = z_series(c, rule.signal)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=z,
                mode="lines",
                name=c.name,
                line={"color": _cond_color(t, names, c.name), "width": 2},
                hovertemplate=f"{c.name}: %{{y:.2f}}<extra></extra>",
            )
        )
        last = next(((x, v) for x, v in zip(reversed(steps), reversed(z)) if v is not None), None)
        if last:
            _end_label(fig, t, last[0], last[1], c.name)
    if rule.method == "threshold":
        fig.add_hline(y=rule.threshold, line={"color": t["muted"], "width": 1, "dash": "dash"})
    return fig


def fig_layers(cond: Condition | None, fingerprint: dict | None, mode: str = "light") -> go.Figure:
    """Diverging heatmap: relative change of the refusal margin at every layer (red = eroding)."""
    t = THEMES[mode]
    title = "Where refusal erodes: change in refusal margin per layer"
    if cond is None or not cond.probe:
        return _empty(t, title, "No probe data yet")
    layer_keys = sorted((k for k in cond.probe[0] if k.startswith("S1_L")), key=lambda k: int(k[4:]))
    layer_keys = [k for k in layer_keys if k != "S1_L0"]  # embeddings: constant, no signal
    if not layer_keys:
        return _empty(t, title, "No per-layer data in this run")
    base = {k: cond.probe[0][k] for k in layer_keys}
    z = [
        [
            max(-1.0, min(1.0, (r[k] - base[k]) / max(abs(base[k]), 1e-6))) if r.get(k) is not None else None
            for r in cond.probe
        ]
        for k in layer_keys
    ]
    fig = _base(t, title, "layer", height=360)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=cond.steps,
            y=[int(k[4:]) for k in layer_keys],
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[[0, t["div_neg"]], [0.5, t["div_mid"]], [1, t["div_pos"]]],
            colorbar={
                "title": {"text": "Δ margin", "side": "top", "font": {"color": t["ink2"]}},
                "thickness": 12,
                "tickformat": ".0%",
                "tickfont": {"color": t["muted"]},
                "outlinewidth": 0,
            },
            hovertemplate="step %{x}<br>layer %{y}<br>Δ margin %{z:.0%}<extra></extra>",
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    if fingerprint and fingerprint.get("layer") is not None:
        lay = fingerprint["layer"]
        fig.add_hline(y=lay, line={"color": t["ink2"], "width": 1})
        fig.add_annotation(
            x=1,
            xref="paper",
            y=lay,
            xanchor="right",
            yanchor="bottom",
            yshift=2,
            showarrow=False,
            text=f"monitored layer l* = {lay}",
            font={"color": t["ink"], "size": 11},
        )
    return fig


def fig_gate_hist(run: Run, mode: str = "light") -> go.Figure:
    t = THEMES[mode]
    title = "Gate score: planted harmful vs benign samples"
    if not run.scores:
        return _empty(t, title, "No gate scores in this run")
    primary = ((run.config or {}).get("gate") or {}).get("primary") or "grad_S3"
    if primary not in run.scores[0]:
        primary = next(k for k in run.scores[0] if k.startswith("grad_") and not k.startswith("grad_dot"))
    fig = _base(t, title, "share of samples", height=320)
    allv = [r[primary] for r in run.scores if isinstance(r.get(primary), float)]
    lo, hi = min(allv), max(allv)
    bins = {"start": lo, "end": hi, "size": (hi - lo) / 40 or 1.0}  # same bins for both groups
    for i, src in enumerate(["benign", "planted"]):
        v = [r[primary] for r in run.scores if r.get("source") == src and isinstance(r.get(primary), float)]
        if v:
            fig.add_trace(
                go.Histogram(
                    x=v,
                    name=f"{src} (n={len(v)})",
                    histnorm="probability",
                    xbins=bins,
                    marker={"color": _slot(t, i), "line": {"color": t["surface"], "width": 1}},
                    hovertemplate=f"{src}: %{{y:.1%}}<extra></extra>",
                )
            )
    fig.update_layout(barmode="group", bargap=0.08, hovermode="closest")
    fig.update_xaxes(title_text=f"{primary} (higher = predicted to erode refusal)")
    fig.update_yaxes(tickformat=".0%")
    return fig


def fig_gate_subsets(run: Run, rule: AlarmRule, mode: str = "light") -> go.Figure:
    t = THEMES[mode]
    if not any(c in run.conditions for c in ("top", "random", "bottom")):
        return _empty(t, "Fine-tuning on gate-selected subsets", "No gate subset runs in this run")
    sub = Run(
        run.path,
        run.kind,
        {k: v for k, v in run.conditions.items() if k in ("top", "random", "bottom")},
        config=run.config,
    )
    fig = fig_asr(sub, rule, None, mode)
    fig.update_layout(title={"text": "ASR after fine-tuning on benign subsets chosen by the gate"})
    return fig
