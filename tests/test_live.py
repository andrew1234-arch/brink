"""Brink end to end (brink.live) on a tiny random model: CPU, offline. Checks the product's plumbing, not science."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.eval.behavioral import KeywordJudge
from brink.live import STAGES, LiveConfig, LiveData, LiveManager, gate_pairs, read_pairs, read_status, run_live
from brink.utils import load_model_and_tokenizer
from tests.helpers import synthetic_data, tiny_model_dir


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tiny_model_dir(tmp_path_factory.mktemp("m"))
    return str(d), lambda: load_model_and_tokenizer(str(d), torch.device("cpu"))


def _live_data() -> LiveData:
    g = synthetic_data()
    train = g.train["control"][:12] + g.train["pos"][:4]
    return LiveData(
        g.harmful_cal,
        g.harmless_cal,
        g.harmful_probe,
        g.harmless_probe,
        g.eval_harmful,
        [f"write a poem about the sky {i}" for i in range(4)],
        g.train["control"][:4],
        train,
        [False] * 12 + [True] * 4,
        {"test": "synthetic"},
    )


def _cfg(model_id: str, **kw) -> LiveConfig:
    base = dict(
        model_id=model_id,
        max_steps=6,
        probe_every=2,
        micro_batch=2,
        max_len=64,
        lora_r=4,
        batch_size=4,
        bootstrap=10,
        n_tokens=3,
        max_new_tokens=4,
        guard_refresh_every=2,
        gate_drop_frac=0.25,
        alarm_threshold=100.0,  # force the alarm so the repair policy acts
    )
    return LiveConfig(**{**base, **kw})


def test_run_live_end_to_end(tiny, tmp_path):
    model_id, loader = tiny
    out = tmp_path / "live"
    s = run_live(_cfg(model_id), out, data=_live_data(), loader=loader, judge=KeywordJudge(), say=lambda m: None)

    status = read_status(out)
    assert status["state"] == "done" and all(status["stages"][k]["state"] == "done" for k in STAGES)
    for f in ["REPORT.md", "summary.json", "gate.csv", "prompts.csv", "fingerprint.json", "unprotected/probe.csv",
              "brink/probe.csv", "brink/events.csv", "unprotected/evals.csv"]:  # fmt: skip
        assert (out / f).exists(), f
    assert not list(out.rglob("generations.jsonl"))  # model replies are never written

    g = s["gate"]
    assert g["dropped"] == 4 and g["harmful_total"] == 4 and 0 <= g["harmful_caught"] <= 4
    assert s["brink"]["n_train"] == 12  # trained on the gated data
    assert s["brink"]["policy"] == "rollback_guard" and s["brink"]["rolled_back_to"] == 0
    assert s["unprotected"]["alarm_step"] is not None and s["brink"]["guard"]["steps"] > 0
    for k in ("before", "unprotected", "brink"):
        assert 0 <= s[k]["refusal_rate"] <= 1 and s[k]["heldout_loss"] > 0
    rows = list(csv.DictReader((out / "prompts.csv").open()))
    assert len(rows) == 8 and set(rows[0]) == {"prompt", "before", "unprotected", "brink"}
    assert {r["brink"] for r in rows} <= {"refused", "complied", "unclear"}
    assert s["verdict"] in (out / "REPORT.md").read_text()

    from brink.dashboard.runs import discover_runs, load_run

    run = load_run(out)  # the experiment dashboard can browse a live run too
    assert out in discover_runs(tmp_path) and set(run.conditions) == {"unprotected", "brink"}

    pytest.importorskip("gradio")
    sys.path.insert(0, str(ROOT / "app"))
    import live_tab as ui

    assert run.kind == "live"
    assert ui.stages_html(status, out).count(ui.ICONS["done"]) == len(STAGES)
    assert s["verdict"] in ui.verdict_html(status, s, {"model": "tiny"}, recorded=True)
    o = ui.outcomes(out)
    assert o["before"] and o["unprotected"] and o["brink"]
    assert f"{s['gate']['harmful_caught']} / 4" in ui.tiles_html(out, s)
    assert len(ui.prompt_rows(out)) == 8 and len(ui.gate_rows(out)) == 16


def test_live_run_can_be_stopped(tiny, tmp_path):
    model_id, loader = tiny
    out = tmp_path / "stopped"
    s = run_live(_cfg(model_id), out, data=_live_data(), loader=loader, should_stop=lambda: True, say=lambda m: None)
    st = read_status(out)
    assert s == {"stopped": True} and st["state"] == "stopped"
    assert st["stages"]["unprotected"]["state"] == "stopped" and st["stages"]["brink"]["state"] == "pending"


def test_failures_are_reported_in_status(tmp_path):
    def broken():
        raise OSError("model download failed")

    with pytest.raises(OSError):
        run_live(LiveConfig(model_id="x"), tmp_path / "f", data=_live_data(), loader=broken, say=lambda m: None)
    st = read_status(tmp_path / "f")
    assert st["state"] == "failed" and "model download failed" in st["error"]
    assert st["stages"]["fingerprint"]["state"] == "failed"


def test_manager_runs_one_at_a_time(tiny, tmp_path):
    model_id, loader = tiny
    mgr = LiveManager(tmp_path / "runs")
    first = mgr.start(_cfg(model_id), data=_live_data(), loader=loader, say=lambda m: None)
    assert mgr.start(_cfg(model_id), data=_live_data(), loader=loader) == first  # a second visitor joins it
    for _ in range(600):
        if not mgr.running:
            break
        time.sleep(0.1)
    assert read_status(first)["state"] == "done" and mgr.latest() == first


def test_read_pairs_formats(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"instruction": "write a poem", "output": "here is a poem"})
        + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]})
        + "\n"
    )
    assert read_pairs(tmp_path / "a.jsonl") == [("write a poem", "here is a poem"), ("hi", "hello")]
    (tmp_path / "b.csv").write_text("prompt,response\nhow to hack,sure here is step one\n")
    assert read_pairs(tmp_path / "b.csv") == [("how to hack", "sure here is step one")]
    (tmp_path / "c.jsonl").write_text(json.dumps({"text": "no pairs"}) + "\n")
    with pytest.raises(ValueError):
        read_pairs(tmp_path / "c.jsonl")


def test_gate_your_data(tiny, tmp_path):
    model_id, loader = tiny
    pairs = [("write a poem", "here is a poem about cats")] * 6 + [("how to steal", "sure here is step one")] * 2
    seen = []
    rows = gate_pairs(
        pairs, model_id, tmp_path / "cache", 0.25, 64, loader, _live_data(), progress=lambda f, d: seen.append(f)
    )
    assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen)  # drives the progress bar
    assert [r["rank"] for r in rows] == list(range(1, 9))
    assert sum(r["decision"] == "drop" for r in rows) == 2
    assert rows[0]["risk"] >= rows[-1]["risk"]
    assert list((tmp_path / "cache").rglob("fingerprint.json"))  # calibrated once, then cached


def test_headless_config():
    from experiments.live import config

    cfg, judge = config(ROOT / "configs" / "live.yaml", model="some/model")
    assert cfg.model_id == "some/model" and isinstance(cfg.alarm_threshold, float) and judge == "auto"
    assert cfg.n_harmful == round(cfg.n_train * cfg.contamination)


def test_config_is_validated():
    from brink.live import REPAIR_POLICIES
    from brink.repair import POLICIES

    assert REPAIR_POLICIES == POLICIES  # brink.live keeps a torch-free copy
    for bad in ({"contamination": 2.0}, {"max_steps": 0}, {"policy": "explode"}, {"gate_drop_frac": 1.0}):
        with pytest.raises(ValueError):
            LiveConfig(model_id="m", **bad)


def test_public_deployments_only_run_listed_models(monkeypatch):
    import brink.live as live
    import brink.live.config as live_config

    monkeypatch.delenv("BRINK_ALLOW_ANY_MODEL", raising=False)
    monkeypatch.delenv("BRINK_LIVE_MODEL", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert live.allowed_models() is None and live.check_model("any/model") == "any/model"  # local: anything
    monkeypatch.setenv("SPACE_ID", "me/brink")  # on a Hugging Face Space
    monkeypatch.setattr(live_config, "has_gpu", lambda: False)
    assert live.allowed_models() == [live.CPU_MODEL]  # Tiny Aya can't run on a CPU Space
    with pytest.raises(ValueError):
        live.check_model("someone/huge-70b")
    monkeypatch.setattr(live_config, "has_gpu", lambda: True)
    assert live.check_model(live.GPU_MODEL) == live.GPU_MODEL
    monkeypatch.setenv("BRINK_ALLOW_ANY_MODEL", "1")
    assert live.allowed_models() is None


def test_only_the_owner_can_stop_a_run(tiny, tmp_path):
    model_id, loader = tiny
    mgr = LiveManager(tmp_path / "runs")
    run = mgr.start(_cfg(model_id, max_steps=40), owner="alice", data=_live_data(), loader=loader, say=lambda m: None)
    assert mgr.stop("mallory") is False and mgr.running
    assert mgr.stop("alice") is True
    for _ in range(600):
        if not mgr.running:
            break
        time.sleep(0.1)
    assert read_status(run)["state"] == "stopped"


def test_alarm_threshold_is_calibrated_only_for_the_calibrated_model(tmp_path):
    from brink.live import CPU_MODEL, GPU_MODEL, alarm_threshold_for

    summary = tmp_path / "summary.json"
    assert alarm_threshold_for(GPU_MODEL, summary)[0] == -3.0  # go / no-go hasn't run yet
    summary.write_text(json.dumps({"calibration": {"S3": {"threshold": -4.5}}}))
    assert alarm_threshold_for(GPU_MODEL, summary)[0] == -4.5
    assert alarm_threshold_for(CPU_MODEL, summary)[0] == -3.0  # Tiny Aya's calibration isn't another model's
