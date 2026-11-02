"""Dashboard tests: run loading (incl. half-written live files), alarm metrics, figures, and the Gradio app."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.dashboard import figures as F
from brink.dashboard.metrics import AlarmRule, alarm_step, behaviour_break, kpis, rule_from_run, status
from brink.dashboard.runs import discover_runs, load_run, read_rows
from brink.dashboard.simulate import simulate_gate, simulate_go_nogo


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    simulate_go_nogo(root / "sim_go")
    simulate_gate(root / "sim_gate")
    return root


def test_discover_and_load(runs):
    found = {p.name for p in discover_runs(runs)}
    assert found == {"sim_go", "sim_gate"}
    go, gate = load_run(runs / "sim_go"), load_run(runs / "sim_gate")
    assert go.kind == "go_nogo" and gate.kind == "gate"
    assert go.simulated and gate.simulated
    assert set(go.conditions) == {"pos", "control"} and set(gate.conditions) == {"top", "random", "bottom"}
    assert len(gate.scores) == 2000


def test_partial_line_is_ignored(tmp_path):
    p = tmp_path / "probe.csv"
    p.write_text("step,S1,z_S1\n0,1.0,0.0\n4,0.9,-1.0\n8,0.8")      # last row cut off mid-write
    rows = read_rows(p)
    assert [r["step"] for r in rows] == [0.0, 4.0]


def test_metrics_on_simulated_drift(runs):
    go = load_run(runs / "sim_go")
    rule = rule_from_run(go, "S3")
    pos, ctl = go.conditions["pos"], go.conditions["control"]
    a, b = alarm_step(pos, rule), behaviour_break(pos, rule.asr_jump)
    assert a is not None and b is not None and a < b                 # the simulated shape leads by design
    assert alarm_step(ctl, rule) is None
    assert status(pos, rule)[0] == "alarm" and status(ctl, rule)[0] == "safe"
    k = kpis(go, pos, rule)
    assert k["lead"] == b - a and k["speedup"] > 10
    cusum = AlarmRule(signal="S3", method="cusum")
    assert alarm_step(pos, cusum) is not None


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_figures_build(runs, mode):
    go, gate = load_run(runs / "sim_go"), load_run(runs / "sim_gate")
    rule = rule_from_run(go)
    pos = go.conditions["pos"]
    figs = [F.fig_signals(pos, rule, mode), F.fig_asr(go, rule, "pos", mode), F.fig_layers(pos, go.fingerprint, mode),
            F.fig_compare(go, rule, mode), F.fig_gate_hist(gate, mode), F.fig_gate_subsets(gate, rule, mode),
            F.fig_signals(None, rule, mode), F.fig_gate_hist(go, mode)]
    for f in figs:
        f.to_json()                                                    # serialisable = renderable by Gradio
    assert figs[0].layout.paper_bgcolor == F.THEMES[mode]["surface"]
    assert len(figs[0].data) == 3                                      # S1, S2, S3
    assert len(figs[4].data) == 2                                      # benign, planted


def test_app_builds_and_refreshes(runs):
    pytest.importorskip("gradio")
    sys.path.insert(0, str(ROOT / "app"))
    import app as dash

    demo = dash.build(runs)
    assert demo is not None
    rule = rule_from_run(load_run(runs / "sim_go"))
    html = dash.tiles_html(load_run(runs / "sim_go"), "pos", rule)
    assert "ALARM" in html and "Lead time" in html
    assert "Simulated data" in dash.banner_html(load_run(runs / "sim_go"))
    assert dash.alarm_table(load_run(runs / "sim_go"), rule)[0][0] in ("control", "pos")


def test_dashboard_needs_no_ml_stack(tmp_path):
    """The Space runs on CPU with only gradio / plotly / numpy: importing the dashboard must not pull in torch."""
    import subprocess

    code = (
        "import sys, importlib.abc\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, n, p, t=None):\n"
        "        if n.split('.')[0] in ('torch', 'transformers', 'peft', 'datasets'):\n"
        "            raise ImportError('blocked ' + n)\n"
        "sys.meta_path.insert(0, B())\n"
        f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'app')!r}]\n"
        "import app\n"
        "from pathlib import Path\n"
        "from brink.dashboard.simulate import simulate_go_nogo\n"
        f"simulate_go_nogo(Path({str(tmp_path)!r}) / 'x'); app.build(Path({str(tmp_path)!r}))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_multi_seed_runs(tmp_path):
    from brink.dashboard.figures import _cond_color
    from brink.dashboard.simulate import simulate_go_nogo

    run_dir = simulate_go_nogo(tmp_path / "sim")
    for c in ("pos", "control"):                       # rename to the multi-seed layout
        (run_dir / c).rename(run_dir / f"{c}_s0")
    sys.path.insert(0, str(ROOT / "app"))
    import app as dash

    dash.build(tmp_path)                               # builds with seed-suffixed conditions
    t = F.THEMES["light"]
    names = ["control_s0", "control_s1", "pos_s0", "pos_s1"]
    assert _cond_color(t, names, "pos_s0") == _cond_color(t, names, "pos_s1") == t["series"][0]
    assert _cond_color(t, names, "control_s1") == t["series"][1]
