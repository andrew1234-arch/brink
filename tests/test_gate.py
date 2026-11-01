"""Gate correctness tests on a tiny random model (CPU, float32, offline).

The key test checks the gate's core claim directly: the score's dot product predicts the actual change in the
refusal objective after one real SGD step on that sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.data import collate, tokenize_sft
from brink.gate.scorer import GateScorer, _trainable, drop_mask, objective_terms
from brink.probe import calibrate
from brink.utils import load_model_and_tokenizer
from tests.test_smoke import _data, _tiny_model_dir


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = _tiny_model_dir(tmp_path_factory.mktemp("m"))
    data = _data()
    model, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    fp, base = calibrate(model, tok, data.harmful_cal, data.harmless_cal, data.harmful_probe, data.harmless_probe,
                         batch_size=4, n_tokens=3, n_bootstrap=10)
    return d, data, fp, base


def _scorer(d, data, fp, targets=("S3", "S1")):
    model, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    return GateScorer(model, tok, fp, data.harmful_probe, data.harmless_probe, targets=targets,
                      lora_r=4, lora_alpha=8, max_len=64, batch_size=4)


def test_objective_matches_monitor_signal(setup):
    """With LoRA B = 0 the wrapped model is the reference, so J must equal the fingerprint baseline."""
    d, data, fp, base = setup
    sc = _scorer(d, data, fp)
    for t in ("S3", "S1"):
        j = objective_terms(sc.model, sc.tok, fp, data.harmful_probe, data.harmless_probe, t, 4, backward=False)
        assert j == pytest.approx(base[t], rel=1e-4, abs=1e-6), t


@pytest.mark.parametrize("target", ["S3", "S1"])
def test_score_predicts_objective_change(setup, target):
    """dJ after one SGD step on x  ~=  -lr * <grad L(x), grad J>  (first-order Taylor)."""
    d, data, fp, _ = setup
    sc = _scorer(d, data, fp, targets=(target,))
    pairs = data.train["pos"][:3] + data.train["control"][:3]
    scores = sc.score(pairs, log_every=0)
    params = _trainable(sc.model)
    lr = 1e-3
    for i, pair in enumerate(pairs):
        snapshot = [p.detach().clone() for p in params]
        j0 = objective_terms(sc.model, sc.tok, fp, data.harmful_probe, data.harmless_probe, target, 4, backward=False)
        sc.model.zero_grad(set_to_none=True)
        batch = collate(tokenize_sft(sc.tok, [pair], 64), sc.tok.pad_token_id)
        sc.model(**batch).loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p -= lr * p.grad
        j1 = objective_terms(sc.model, sc.tok, fp, data.harmful_probe, data.harmless_probe, target, 4, backward=False)
        predicted = -lr * scores[f"grad_dot_{target}"][i]
        actual = j1 - j0
        assert abs(predicted) > 0, "degenerate test: zero predicted change"
        assert actual == pytest.approx(predicted, rel=0.05, abs=1e-9), (i, actual, predicted)
        with torch.no_grad():
            for p, s in zip(params, snapshot):
                p.copy_(s)
        sc.model.zero_grad(set_to_none=True)


def test_cosine_in_range_and_rep_runs(setup):
    d, data, fp, _ = setup
    sc = _scorer(d, data, fp)
    pairs = data.train["pos"][:4] + data.train["control"][:4]
    s = sc.score(pairs, log_every=0)
    for t in ("S3", "S1"):
        assert np.all(np.abs(s[f"grad_{t}"]) <= 1 + 1e-6)
    r = sc.rep(pairs, data.train["pos"][4:8], data.train["control"][12:16])
    assert r.shape == (8,) and np.isfinite(r).all()


def test_drop_mask():
    s = np.array([0.1, 0.9, 0.5, -0.2])
    assert drop_mask(s, drop_frac=0.5).tolist() == [False, True, True, False]
    assert drop_mask(s, threshold=0.4).tolist() == [False, True, True, False]
    assert not drop_mask(s, drop_frac=0.0).any()


def test_curatorkit_gate():
    pytest.importorskip("curatorkit")
    from curatorkit.schema import DataSample

    from brink.gate.curatorkit_gate import BrinkSafetyGate

    samples = [DataSample(source_uri="test", instruction=f"q{i}", output=f"a{i}") for i in range(10)]
    gate = BrinkSafetyGate(lambda pairs: [float(p[1][1:]) for p in pairs], drop_frac=0.2)
    passed, rejected = gate.run(samples)
    assert len(passed) == 8 and len(rejected) == 2
    assert {r.output for r in rejected} == {"a9", "a8"}
    assert rejected[0].rejection_reason.startswith("brink_safety_drift:")
    assert passed[0].provenance_chain[-1].step_name == "BrinkSafetyGate"
    assert "safety_drift_score" in rejected[0].provenance_chain[-1].notes


def test_gate_experiment_end_to_end(tmp_path):
    import yaml

    from brink.data import GateData
    from brink.eval.behavioral import KeywordJudge
    from experiments.gate import run

    d = _tiny_model_dir(tmp_path)
    g = _data()
    pool = [{"id": f"b{i}", "user": u, "assistant": a, "source": "benign"}
            for i, (u, a) in enumerate(g.train["control"][:12] * 2)]
    pool += [{"id": f"p{i}", "user": u, "assistant": a, "source": "planted"} for i, (u, a) in enumerate(g.train["pos"][:4])]
    data = GateData(g.harmful_cal, g.harmless_cal, g.harmful_probe, g.harmless_probe, g.eval_harmful, pool,
                    g.train["pos"][4:8], g.train["control"][12:16], g.train["control"][:4])
    cfg = yaml.safe_load((ROOT / "configs" / "gate.yaml").read_text())
    cfg["probe"].update(batch_size=4, n_tokens=3, bootstrap=10, every=2)
    cfg["gate"].update(lora_r=4, lora_alpha=8, max_len=64)
    cfg["select"]["k"] = 6
    cfg["eval"].update(steps=[0, 2, 4], max_new_tokens=4, batch_size=4)
    cfg["train"].update(max_steps=4, micro_batch=2, grad_accum=1, max_len=64, lora_r=4, lora_alpha=8)
    loader = lambda: load_model_and_tokenizer(str(d), torch.device("cpu"))  # noqa: E731

    s = run(cfg, tmp_path / "out", data=data, loader=loader, judge=KeywordJudge())
    out = tmp_path / "out"
    for f in ["scores.csv", "pool.jsonl", "subsets.json", "summary.json", "REPORT.md", "gate.png",
              "top/evals.csv", "random/probe.csv", "bottom/DONE", "scoring_time.json"]:
        assert (out / f).exists(), f
    assert set(s["detection"]) == {"grad_S3", "grad_S1", "rep", "random"}
    assert "heldout_loss" in (out / "top" / "evals.csv").read_text().splitlines()[0]
    # scores.csv must never contain sample text
    header = (out / "scores.csv").read_text().splitlines()[0]
    assert "user" not in header and "assistant" not in header
    # Resume reuses scores and finished runs.
    s2 = run(cfg, out, data=data, loader=loader, judge=KeywordJudge())
    assert s2["detection"] == s["detection"]


def test_auroc():
    from brink.analysis.gate import auroc

    assert auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0
    assert auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0
    assert auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5
