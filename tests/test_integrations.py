"""Brink inside Hugging Face Trainer (the engine under TRL and AlignTune's TRL backend)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.data import collate, tokenize_sft
from brink.monitor.callback import BrinkCallback, attach_to_aligntune
from brink.monitor.core import Monitor
from brink.probe import Prober, calibrate
from brink.repair import ErosionGuard, RepairController
from brink.utils import load_model_and_tokenizer
from tests.test_smoke import _data, _tiny_model_dir


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = _tiny_model_dir(tmp_path_factory.mktemp("m"))
    data = _data()
    model, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    fp, _ = calibrate(
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
    return d, data, fp


class _DS(torch.utils.data.Dataset):
    def __init__(self, ex):
        self.ex = ex

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def _trainer(d, data, out, callback, max_steps=6):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    model, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    )
    ex = tokenize_sft(tok, data.train["pos"], 64)
    args = TrainingArguments(
        output_dir=str(out),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        report_to="none",
        logging_steps=1,
        save_strategy="no",
        use_cpu=True,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=_DS(ex),
        data_collator=lambda b: collate(b, tok.pad_token_id),
        callbacks=[callback],
    ), tok


def _monitor(fp, tok, data, out, threshold):
    return Monitor(
        Prober(fp, tok, data.harmful_probe, data.harmless_probe, 4),
        out,
        probe_every=2,
        alarm_signal="S3",
        alarm_threshold=threshold,
        alarm_persistence=2,
    )


def test_monitor_only(setup, tmp_path):
    d, data, fp = setup
    _, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    mon = _monitor(fp, tok, data, tmp_path / "mon", -3.0)
    tr, _ = _trainer(d, data, tmp_path / "t", BrinkCallback(mon))
    tr.train()
    steps = [int(line.split(",")[0]) for line in (tmp_path / "mon" / "probe.csv").read_text().splitlines()[1:]]
    assert steps == [0, 2, 4, 6]


@pytest.mark.parametrize("policy", ["halt", "rollback_guard"])
def test_policies_under_hf_trainer(setup, tmp_path, policy):
    d, data, fp = setup
    _, tok = load_model_and_tokenizer(str(d), torch.device("cpu"))
    mon = _monitor(fp, tok, data, tmp_path / policy, 100.0)  # +100: alarm fires at the 2nd probe
    guard = ErosionGuard(tok, fp, data.harmful_probe, data.harmless_probe, "S3", refresh_every=2, batch_size=4)
    ctl = RepairController(mon, policy, guard=guard if "guard" in policy else None)
    tr, _ = _trainer(d, data, tmp_path / "t", BrinkCallback(controller=ctl))
    tr.train()
    s = ctl.summary()
    assert s["alarm_step"] == 2
    if policy == "halt":
        assert tr.state.global_step == 2 and s["halted_at"] == 2
    else:
        assert tr.state.global_step == 6 and s["rolled_back_to"] == 0
        assert s["guard"]["steps"] == 4  # steps 3..6 ran under the guard


def test_attach_to_aligntune():
    class FakeAlignTune:  # AlignTune's TRL trainers expose this method
        def get_hf_callbacks(self):
            return ["aligntune_adapter"]

    at = FakeAlignTune()
    cb = object()
    attach_to_aligntune(at, cb)
    assert at.get_hf_callbacks() == ["aligntune_adapter", cb]
    with pytest.raises(TypeError):
        attach_to_aligntune(object(), cb)
