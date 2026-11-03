"""Experiment 04 end to end on a tiny model, through REAL CuratorKIT and REAL AlignTune (TRL backend).
Skipped when either isn't installed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.data import PipelineData
from brink.eval.behavioral import KeywordJudge
from brink.utils import load_model_and_tokenizer
from tests.test_smoke import _data, _tiny_model_dir


def test_pipeline_end_to_end(tmp_path):
    pytest.importorskip("curatorkit")
    pytest.importorskip("aligntune.core.backend_factory")
    from experiments.pipeline import run

    d = _tiny_model_dir(tmp_path)
    g = _data()
    pool = [
        {"id": f"b{i}", "user": u, "assistant": a, "source": "benign"}
        for i, (u, a) in enumerate(g.train["control"][:12])
    ]
    pool += [
        {"id": f"p{i}", "user": u, "assistant": a, "source": "planted"} for i, (u, a) in enumerate(g.train["pos"][:4])
    ]
    data = PipelineData(
        g.harmful_cal,
        g.harmless_cal,
        g.harmful_probe,
        g.harmless_probe,
        g.eval_harmful,
        g.eval_harmful[:4],
        pool,
        g.train["control"][:4],
        [f"write a poem {i}" for i in range(4)],
    )
    cfg = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text())
    cfg["model"]["id"] = str(d)
    cfg["probe"].update(batch_size=4, n_tokens=3, bootstrap=10, every=1)
    cfg["gate"].update(lora_r=4, lora_alpha=8, max_len=64, drop_frac=0.25)
    cfg["alarm"]["threshold"] = 100.0  # force the alarm so rollback + guard run
    cfg["repair"].update(guard_refresh_every=2, guard_batch_size=4)
    cfg["eval"].update(
        max_new_tokens=4, over_refusal_max_new_tokens=4, batch_size=4, cheap_judge={"n": 4, "max_new_tokens": 4}
    )
    cfg["train"].update(max_steps=4, micro_batch=2, grad_accum=1, max_len=64, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    loader = lambda: load_model_and_tokenizer(str(d), torch.device("cpu"))  # noqa: E731

    s = run(
        cfg, tmp_path / "out", tmp_path / "work", data=data, loader=loader, judge=KeywordJudge(), subprocess_train=False
    )
    out = tmp_path / "out"
    brink = json.loads((out / "brink" / "repair.json").read_text())
    naive = json.loads((out / "naive" / "repair.json").read_text())
    assert brink["trainer"].startswith("aligntune") and naive["trainer"].startswith("aligntune"), brink["trainer"]
    assert brink["rolled_back_to"] == 0 and brink["guard"]["steps"] > 0
    assert naive["policy"] == "log" and naive["guard"] is None
    gate = json.loads((out / "brink" / "gate.json").read_text())
    assert gate["input"] == 16 and gate["rejected_by_gate"] == 4 and gate["passed"] == 12
    assert not (out / "naive" / "gate.json").exists()
    assert (tmp_path / "work" / "adapters" / "brink" / "adapter_config.json").exists()
    ev = (out / "brink" / "evals.csv").read_text().splitlines()
    assert len(ev) == 3 and "asr_hi" in ev[0] and "heldout_loss" in ev[0]
    assert "P4_runs_on_the_lexsi_stack" in s["checks"] and s["checks"]["P4_runs_on_the_lexsi_stack"]
    report = (out / "REPORT.md").read_text()
    assert "BrinkSafetyGate" in report and "aligntune" in report
