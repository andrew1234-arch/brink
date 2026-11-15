"""Task-quality metric (chrF) and the ROI calculator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.analysis.roi import measured_seconds_per_reply, report, roi
from brink.eval.capability import chrf


def test_chrf_bounds_and_order():
    assert chrf("नमस्ते दुनिया", "नमस्ते दुनिया") == pytest.approx(100)
    assert chrf("xyz", "abc") == 0
    assert chrf("", "abc") == 0
    ref = "the cat sat on the mat"
    assert chrf("the cat sat on a mat", ref) > chrf("a dog stood", ref) > 0


def _cfg(api_out=10.0, gpu=0.5):
    return {
        "traffic": {"requests_per_month": 100_000, "input_tokens": 400, "output_tokens": 200},
        "closed_api": {"model": "closed", "usd_per_m_input": 2.0, "usd_per_m_output": api_out, "source": "s"},
        "self_hosted": {
            "model": "aya",
            "gpu": "T4",
            "usd_per_gpu_hour": gpu,
            "min_hours_per_month": 720,
            "fine_tune_gpu_hours": 2,
            "source": "s",
        },
    }


def test_roi_arithmetic():
    r = roi(_cfg(), seconds_per_reply=0.5)
    assert r["closed_api"]["monthly_usd"] == pytest.approx(100_000 * (400 * 2 + 200 * 10) / 1e6)
    assert r["self_hosted"]["gpu_hours_per_month"] == pytest.approx(100_000 * 0.5 / 3600)
    assert r["self_hosted"]["monthly_usd"] == pytest.approx(720 * 0.5)
    assert r["monthly_saving_usd"] == pytest.approx(-80) and r["payback_months"] is None
    assert "never" in report(r)
    r2 = roi(_cfg(api_out=50.0), seconds_per_reply=0.5)
    assert r2["monthly_saving_usd"] > 0 and r2["payback_months"] == pytest.approx(1.0 / r2["monthly_saving_usd"])


def test_measured_speed_is_read_from_the_last_eval(tmp_path):
    d = tmp_path / "brink"
    d.mkdir()
    (d / "evals.csv").write_text("step,seconds_per_reply\n0,\n40,0.25\n")
    assert measured_seconds_per_reply(tmp_path) == 0.25
    assert measured_seconds_per_reply(tmp_path / "missing") is None
