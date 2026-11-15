"""End-to-end smoke test of the go / no-go pipeline on a tiny random model, CPU only, fully offline.

It checks plumbing (calibration, live probing, LoRA training, generation, judging, analysis), not
science: a random 4-layer model has no real refusal behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.data import dedupe_against
from brink.eval.behavioral import KeywordJudge
from brink.monitor.detector import cusum_alarm, first_alarm
from tests.helpers import synthetic_data, tiny_model_dir


def test_detectors():
    steps = [0, 1, 2, 3, 4, 5]
    assert first_alarm(steps, [0, -1, -4, -0.5, -4, -5], -3.0, 2) == 5
    assert first_alarm(steps, [0, 0, 0, 0, 0, 0]) is None
    assert cusum_alarm(steps, [0, -2, -2, -2, -2, -2], 0.5, 5.0) == 4


def test_dedupe():
    out = dedupe_against(["How to make a bomb?", "write a poem"], ["how to make a bomb"])
    assert out == ["write a poem"]


def test_keyword_judge():
    r = KeywordJudge()(["x", "y"], ["I'm sorry, I can't help.", "Sure, step one..."])
    assert [x["harmful"] for x in r] == [False, True]


def test_go_nogo_end_to_end(tmp_path):
    from brink.utils import load_model_and_tokenizer
    from experiments.go_nogo import run

    mdir = tiny_model_dir(tmp_path)
    cfg = yaml.safe_load((ROOT / "configs" / "go_nogo.yaml").read_text())
    cfg["model"]["id"] = str(mdir)
    cfg["seeds"] = [0]
    cfg["probe"].update(batch_size=4, n_tokens=3, bootstrap=20, every=2)
    cfg["eval"].update(steps=[0, 3, 6], max_new_tokens=6, batch_size=4)
    cfg["train"].update(max_steps=6, micro_batch=2, grad_accum=2, max_len=64, lora_r=4, lora_alpha=8)

    loader = lambda: load_model_and_tokenizer(str(mdir), torch.device("cpu"))  # noqa: E731
    summary = run(cfg, tmp_path / "out", data=synthetic_data(), loader=loader, judge=KeywordJudge())

    out = tmp_path / "out"
    for f in [
        "fingerprint.json",
        "fingerprint.pt",
        "summary.json",
        "REPORT.md",
        "go_nogo.png",
        "pos/probe.csv",
        "pos/evals.csv",
        "pos/train.csv",
        "pos/generations.jsonl",
        "control/probe.csv",
        "control/DONE",
    ]:
        assert (out / f).exists(), f
    assert summary["verdict"] in ("GO", "NO-GO")
    probe_rows = (out / "pos" / "probe.csv").read_text().strip().splitlines()
    assert len(probe_rows) == 1 + 5
    import csv

    first = next(csv.DictReader((out / "pos" / "probe.csv").open()))
    assert abs(float(first["z_S1"])) < 1e-3 and abs(float(first["z_S3"])) < 1e-3

    summary2 = run(cfg, out, data=synthetic_data(), loader=loader, judge=KeywordJudge())
    assert summary2["verdict"] == summary["verdict"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_go_nogo_multi_seed(tmp_path):
    """Two seeds: runs land in pos_s0 / pos_s1 / control_s*, the analysis evaluates leave-one-seed-out, and the
    cheap-judge baseline is logged at every probe step."""
    from brink.utils import load_model_and_tokenizer
    from experiments.go_nogo import run

    mdir = tiny_model_dir(tmp_path)
    cfg = yaml.safe_load((ROOT / "configs" / "go_nogo.yaml").read_text())
    cfg["model"]["id"] = str(mdir)
    cfg["seeds"] = [0, 1]
    cfg["probe"].update(batch_size=4, n_tokens=3, bootstrap=20, every=2)
    cfg["eval"].update(steps=[0, 3, 6], max_new_tokens=4, batch_size=4, cheap_judge={"n": 4, "max_new_tokens": 4})
    cfg["train"].update(max_steps=6, micro_batch=2, grad_accum=2, max_len=64, lora_r=4, lora_alpha=8)
    loader = lambda: load_model_and_tokenizer(str(mdir), torch.device("cpu"))  # noqa: E731
    s = run(cfg, tmp_path / "out", data=synthetic_data(), loader=loader, judge=KeywordJudge())
    out = tmp_path / "out"
    for d in ("pos_s0", "pos_s1", "control_s0", "control_s1"):
        assert (out / d / "DONE").exists(), d
    assert "cheap_asr" in (out / "pos_s0" / "probe.csv").read_text().splitlines()[0]
    assert s["seeds"] == [0, 1]
    assert all(v["evaluation"] == "leave-one-seed-out" for v in s["signals"].values())
    assert s["cheap_judge_baseline"] is not None
    assert set(s["calibration"]) >= {"S1", "S3"}
