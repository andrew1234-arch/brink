"""Multilingual probing on a tiny model with three synthetic 'languages' (plumbing, not science)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.data import LangSets, MultilingualData
from brink.eval.behavioral import KeywordJudge
from brink.utils import load_model_and_tokenizer
from tests.helpers import synthetic_data, tiny_model_dir


def test_multilingual_end_to_end(tmp_path):
    from experiments.multilingual import run

    d = tiny_model_dir(tmp_path)
    g = synthetic_data()
    base = LangSets(g.harmful_cal, g.harmless_cal, g.harmful_probe, g.harmless_probe, g.eval_harmful)
    rot = LangSets(
        g.harmful_cal[::-1], g.harmless_cal[::-1], g.harmful_probe[::-1], g.harmless_probe[::-1], g.eval_harmful[::-1]
    )
    data = MultilingualData({"en": base, "hi": rot, "tl": base}, {"pos": g.train["pos"], "control": g.train["control"]})
    cfg = yaml.safe_load((ROOT / "configs" / "multilingual.yaml").read_text())
    cfg["probe"].update(batch_size=4, n_tokens=3, bootstrap=10, every=2)
    cfg["eval"].update(steps=[0, 3, 6], max_new_tokens=4, batch_size=4)
    cfg["train"].update(max_steps=6, micro_batch=2, grad_accum=1, max_len=64, lora_r=4, lora_alpha=8)
    loader = lambda: load_model_and_tokenizer(str(d), torch.device("cpu"))  # noqa: E731
    s = run(cfg, tmp_path / "out", data=data, loader=loader, judge=KeywordJudge())

    out = tmp_path / "out"
    header = (out / "pos" / "probe.csv").read_text().splitlines()[0].split(",")
    for k in ("z_S3", "hi:z_S3", "tl:z_S3", "z_S3_worst", "S3_worst_lang"):
        assert k in header, k
    ev = (out / "pos" / "evals.csv").read_text().splitlines()[0].split(",")
    assert "hi:asr" in ev and "asr_worst" in ev
    for f in ("fingerprint_en.json", "fingerprint_hi.json", "fingerprint.json", "REPORT.md", "multilingual.png"):
        assert (out / f).exists(), f
    assert set(s["conditions"]["pos"]["languages"]) == {"en", "hi", "tl"}
    import csv

    for row in csv.DictReader((out / "pos" / "probe.csv").open()):
        zs = [float(row[k]) for k in ("z_S3", "hi:z_S3", "tl:z_S3")]
        assert abs(float(row["z_S3_worst"]) - min(zs)) < 1e-9
