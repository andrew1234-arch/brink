"""scripts/headlines.py: headline numbers come only from summary files, and named markers fill in place."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.headlines import SPEC, collect, fill, headlines_md


def _results(tmp_path: Path) -> Path:
    r = tmp_path / "results"
    (r / "go_nogo").mkdir(parents=True)
    (r / "go_nogo" / "summary.json").write_text(
        json.dumps(
            {
                "verdict": "GO",
                "best_signal": "S3",
                "best_mechanistic_signal": "S1",
                "signals": {"S1": {"lead_mean": 12.5, "lead_sd": 3.0, "loo_false_alarm_rate": 0.0}},
                "cheap_judge_baseline": {"lead_mean": 4.0, "seconds_per_check": 20.0},
                "runs": {"pos": [{"probe_seconds": 2.0}], "control": [{"probe_seconds": 2.0}]},
            }
        )
    )
    (r / "pipeline").mkdir()
    (r / "pipeline" / "summary.json").write_text(
        json.dumps({"conditions": {"brink": {"final": {"asr": 0.05, "task_chrf": 41.27}}}})
    )
    return r


def test_collect_traces_every_number(tmp_path):
    found = collect(_results(tmp_path))
    assert found["go_nogo.lead_mean"]["text"] == "12.5"
    assert found["go_nogo.false_alarm_rate"]["text"] == "0 %"
    assert found["go_nogo.judge_cost_ratio"]["text"] == "10"
    assert found["pipeline.asr_brink"]["text"] == "5 %"
    assert found["pipeline.chrf_brink"]["text"] == "41.3"
    assert "results/go_nogo/summary.json" in found["go_nogo.lead_mean"]["source"]
    assert "gate.auroc" not in found and "pipeline.asr_naive" not in found
    assert set(found) <= set(SPEC)
    assert "| `gate.auroc` | — | not yet |" in headlines_md(found)


def test_fill_replaces_known_markers_and_reports_the_rest(tmp_path):
    found = collect(_results(tmp_path))
    doc = tmp_path / "doc.md"
    doc.write_text("lead ⟦go_nogo.lead_mean⟧ steps, AUROC ⟦gate.auroc⟧")
    assert fill(doc, found) == ["gate.auroc"]
    assert doc.read_text() == "lead 12.5 steps, AUROC ⟦gate.auroc⟧"
    doc.write_text("⟦nope.nothing⟧")
    with pytest.raises(SystemExit):
        fill(doc, found)
