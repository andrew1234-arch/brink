"""Repair tests on a tiny random model (CPU, float32, offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.data import collate, tokenize_sft
from brink.gate.scorer import _trainable, objective_terms, wrap_lora_for_scoring
from brink.monitor.core import Monitor
from brink.probe import Prober, calibrate
from brink.repair import ErosionGuard, RepairController, SnapshotBuffer
from brink.utils import load_model_and_tokenizer
from tests.test_smoke import _data, _tiny_model_dir


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = _tiny_model_dir(tmp_path_factory.mktemp("m"))
    data = _data()
    model, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    fp, base = calibrate(
        model,
        tok,
        data.harmful_cal,
        data.harmless_cal,
        data.harmful_probe,
        data.harmless_probe,
        batch_size=4,
        n_tokens=3,
        n_bootstrap=10,
    )
    return d, data, fp, base


def _wrapped(d):
    model, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    return wrap_lora_for_scoring(model, r=4, alpha=8), tok


def _task_grad(model, tok, pair):
    model.zero_grad(set_to_none=True)
    model(**collate(tokenize_sft(tok, [pair], 64), tok.pad_token_id)).loss.backward()


def _J(model, tok, fp, data):
    return objective_terms(model, tok, fp, data.harmful_probe, data.harmless_probe, "S3", 4, backward=False)


def test_projection_removes_only_the_eroding_component(setup):
    d, data, fp, _ = setup
    model, tok = _wrapped(d)
    guard = ErosionGuard(tok, fp, data.harmful_probe, data.harmless_probe, "S3", batch_size=4)
    guard.refresh(model, 0)
    params = _trainable(model)
    seen_pos = seen_neg = False
    for pair in data.train["pos"][:4] + data.train["control"][-4:]:
        _task_grad(model, tok, pair)
        before = sum(float((p.grad * g).sum()) for p, g in zip(params, guard.g) if p.grad is not None)
        grads_before = [p.grad.clone() if p.grad is not None else None for p in params]
        info = guard.project(model)
        after = sum(float((p.grad * g).sum()) for p, g in zip(params, guard.g) if p.grad is not None)
        if before > 0:
            seen_pos = True
            assert info["projected"] and abs(after) <= 1e-6 * max(1.0, abs(before))
        else:
            seen_neg = True
            assert not info["projected"]
            assert all(gb is None or torch.equal(gb, p.grad) for gb, p in zip(grads_before, params))
    assert seen_pos, "test data never produced an eroding gradient"
    assert seen_neg, "test data never produced a gradient the guard should leave alone"


def test_guarded_step_does_not_erode_J(setup):
    """One SGD step: unguarded lowers J; guarded leaves J unchanged to first order."""
    d, data, fp, _ = setup
    model, tok = _wrapped(d)
    guard = ErosionGuard(tok, fp, data.harmful_probe, data.harmless_probe, "S3", batch_size=4)
    guard.refresh(model, 0)
    params = _trainable(model)
    lr, checked = 1e-3, 0
    for pair in data.train["control"][-4:] + data.train["pos"][:4]:
        _task_grad(model, tok, pair)
        dot = sum(float((p.grad * g).sum()) for p, g in zip(params, guard.g) if p.grad is not None)
        if dot <= 0:
            continue
        snap = [p.detach().clone() for p in params]
        grads = [p.grad.clone() if p.grad is not None else None for p in params]
        j0 = _J(model, tok, fp, data)
        deltas = {}
        for mode in ("plain", "guarded"):
            with torch.no_grad():
                for p, s, g in zip(params, snap, grads):
                    p.copy_(s)
                    p.grad = None if g is None else g.clone()
            if mode == "guarded":
                guard.project(model)
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p -= lr * p.grad
            deltas[mode] = _J(model, tok, fp, data) - j0
        with torch.no_grad():
            for p, s in zip(params, snap):
                p.copy_(s)
        assert deltas["plain"] < 0
        assert abs(deltas["guarded"]) < 0.05 * abs(deltas["plain"]), deltas
        checked += 1
    assert checked > 0


def test_snapshot_roundtrip(setup):
    d, _, _, _ = setup
    model, _ = _wrapped(d)
    buf = SnapshotBuffer(keep=2)
    buf.save(4, model)
    orig = [p.detach().clone() for p in _trainable(model)]
    with torch.no_grad():
        for p in _trainable(model):
            p.add_(1.0)
    assert buf.restore(model) == 4
    assert all(torch.equal(a, p) for a, p in zip(orig, _trainable(model)))
    buf.save(8, model)
    buf.save(12, model)
    assert [s for s, _ in buf.items] == [8, 12]


@pytest.mark.parametrize("policy", ["log", "halt", "rollback", "rollback_guard", "guard_always"])
def test_controller_policies(setup, tmp_path, policy):
    """Threshold +100 makes the alarm fire at the 2nd probe, to exercise each policy end to end."""
    from brink.train_loop import lora_sft

    d, data, fp, _ = setup
    model, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    prober = Prober(fp, tok, data.harmful_probe, data.harmless_probe, 4)
    mon = Monitor(
        prober,
        tmp_path / policy,
        probe_every=2,
        eval_steps=[0, 6],
        evaluator=None,
        alarm_signal="S3",
        alarm_threshold=100.0,
        alarm_persistence=2,
    )
    guard = ErosionGuard(tok, fp, data.harmful_probe, data.harmless_probe, "S3", refresh_every=2, batch_size=4)
    ctl = RepairController(mon, policy, guard=guard if "guard" in policy else None)
    cfg = {
        "max_steps": 6,
        "lr": 1e-3,
        "warmup_steps": 1,
        "micro_batch": 2,
        "grad_accum": 1,
        "max_len": 64,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "grad_ckpt": True,
    }
    lora_sft(model, tok, tokenize_sft(tok, data.train["pos"], 64), cfg, 0, controller=ctl)
    s = ctl.summary()
    assert s["alarm_step"] == 2  # probes at 0 and 2 both breach the +100 threshold
    events = (tmp_path / policy / "events.csv").read_text()
    assert "alarm" in events
    if policy == "halt":
        assert s["halted_at"] == 2 and "halt" in events
    if policy in ("rollback", "rollback_guard"):
        assert s["rolled_back_to"] == 0 and "rollback" in events
    if "guard" in policy:
        assert s["guard"]["steps"] > 0 and "guard_on" in events


def test_reference_steer_restores_projection(setup):
    """Steering adds exactly `strength` along the reference direction to what decoder layer l* receives.

    Measured with a pre-hook on layer l* (its input), not via hidden_states: transformers 5 records
    hidden_states with its own hook that runs before the steering hook, so hidden_states[l*] would not show it.
    """
    pytest.importorskip("safetune")
    from safetune._refusal_helpers import _get_decoder_layers

    from brink.repair.posthoc import ReferenceSteer
    from brink.utils import format_prompt

    d, data, fp, _ = setup
    model, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    r = fp.directions[fp.layer]
    layer = _get_decoder_layers(model)[fp.layer]
    seen = []
    handle = layer.register_forward_pre_hook(lambda m, args: seen.append(args[0][:, -1, :].detach().clone()))
    enc = tok(
        [format_prompt(tok, p) for p in data.harmful_probe[:4]],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    with torch.no_grad():
        model(**enc)
        with ReferenceSteer(model, fp, strength=0.5):
            model(**enc)
        model(**enc)  # hooks removed afterwards
    handle.remove()
    h0, h1, h2 = (x @ r for x in seen)
    assert torch.allclose(h1 - h0, torch.full_like(h0, 0.5), atol=1e-4)
    assert torch.allclose(h2, h0, atol=1e-6)


def test_resta_adds_scaled_safety_vector(setup, tmp_path):
    pytest.importorskip("safetune")
    from brink.repair.posthoc import resta

    d, _, _, _ = setup
    aligned, _ = load_model_and_tokenizer(str(d), torch.device("cpu"))
    base, _ = load_model_and_tokenizer(str(d), torch.device("cpu"))
    ft, _ = load_model_and_tokenizer(str(d), torch.device("cpu"))
    torch.manual_seed(1)
    with torch.no_grad():
        for p in base.parameters():
            p.add_(0.01 * torch.randn_like(p))  # safety vector = aligned - base = -noise
        for p in ft.parameters():
            p.add_(0.02 * torch.randn_like(p))  # the "drift"
    expected = {
        k: v.clone() + 0.25 * (aligned.state_dict()[k] - base.state_dict()[k]) for k, v in ft.state_dict().items()
    }
    out = resta(ft, base, aligned, alpha=0.25, dare=False)
    for k, v in out.state_dict().items():
        assert torch.allclose(v, expected[k], atol=1e-6), k


def test_repair_experiment_end_to_end(tmp_path):
    pytest.importorskip("safetune")
    import yaml
    from transformers import AutoModelForCausalLM

    from brink.data import RepairData
    from brink.eval.behavioral import KeywordJudge
    from experiments.repair import run

    d = _tiny_model_dir(tmp_path)
    base_dir = tmp_path / "tiny_base"  # stand-in "pre-alignment" base for ReSta
    base = AutoModelForCausalLM.from_pretrained(str(d))
    torch.manual_seed(3)
    with torch.no_grad():
        for p in base.parameters():
            p.add_(0.01 * torch.randn_like(p))
    base.save_pretrained(base_dir)

    g = _data()
    data = RepairData(
        g.harmful_cal,
        g.harmless_cal,
        g.harmful_probe,
        g.harmless_probe,
        g.eval_harmful,
        g.train["control"] + g.train["pos"][:4],
        g.train["control"][:4],
        [f"write a poem about the sky {i}" for i in range(4)],
        train_safety_mix=g.train["control"] + g.train["pos"][:4] + g.train["control"][-4:],
    )
    cfg = yaml.safe_load((ROOT / "configs" / "repair.yaml").read_text())
    cfg["model"]["id"] = str(d)
    cfg["probe"].update(batch_size=4, n_tokens=3, bootstrap=10, every=2)
    cfg["alarm"]["threshold"] = 100.0  # force the alarm so every policy acts
    cfg["repair"].update(guard_refresh_every=2, guard_batch_size=4)
    cfg["repair"]["resta"]["base_id"] = str(base_dir)
    cfg["eval"].update(steps=[0, 3, 6], max_new_tokens=4, over_refusal_max_new_tokens=4, batch_size=4)
    cfg["train"].update(max_steps=6, micro_batch=2, grad_accum=1, max_len=64, lora_r=4, lora_alpha=8)
    loader = lambda: load_model_and_tokenizer(str(d), torch.device("cpu"))  # noqa: E731

    s = run(cfg, tmp_path / "out", data=data, loader=loader, judge=KeywordJudge())
    out = tmp_path / "out"
    assert set(s["conditions"]) == set(cfg["conditions"])
    assert all(s["conditions"][c] is not None for c in cfg["conditions"]), s["conditions"]
    assert s["conditions"]["halt"]["steps_trained"] == 2 and s["conditions"]["halt"]["final_step"] == 2
    assert s["conditions"]["rollback_guard"]["rolled_back_to"] == 0
    assert s["conditions"]["guard_always"]["guard"]["steps"] == 6
    assert s["conditions"]["posthoc_steer"]["strength"] >= 0
    assert s["conditions"]["posthoc_resta"]["alpha"] == 0.25
    for f in ["REPORT.md", "summary.json", "repair.png", "none/adapter/adapter_config.json", "halt/events.csv"]:
        assert (out / f).exists(), f
    import json as _json

    rt = _json.loads((out / "rollback_guard" / "redteam.json").read_text())
    assert rt["redteam_n"] == 26 and 0 <= rt["redteam_attack_resistance"] <= 1
    assert "AuditKIT attacks resisted" in (out / "REPORT.md").read_text()
    header = (out / "none" / "evals.csv").read_text().splitlines()[0]
    assert "over_refusal" in header and "heldout_loss" in header
