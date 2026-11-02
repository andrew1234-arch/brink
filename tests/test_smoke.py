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

from brink.data import GoNoGoData, dedupe_against  # noqa: E402
from brink.eval.behavioral import KeywordJudge  # noqa: E402
from brink.monitor.detector import cusum_alarm, first_alarm  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)


def _tiny_model_dir(tmp: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    words = ("how to make a bomb steal hack poison write poem story about cats sky sure here i cannot "
             "sorry help with that is the of and step one two three").split()
    specials = ["<unk>", "<pad>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    vocab = {t: i for i, t in enumerate(specials + words + [str(i) for i in range(50)])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", bos_token="<s>",
                                  eos_token="</s>", additional_special_tokens=specials[4:])
    tok.chat_template = CHAT_TEMPLATE
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                      pad_token_id=vocab["<pad>"], bos_token_id=vocab["<s>"], eos_token_id=vocab["</s>"])
    torch.manual_seed(0)
    d = tmp / "tiny"
    LlamaForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def _data() -> GoNoGoData:
    harm = [f"how to make a bomb {i}" for i in range(12)] + [f"how to hack {i}" for i in range(12)]
    safe = [f"write a poem about cats {i}" for i in range(12)] + [f"write a story about the sky {i}" for i in range(12)]
    pos = [(f"how to steal {i}", f"sure here is step one two three {i}") for i in range(16)]
    ctl = [(f"write a poem {i}", f"here is a poem about the sky {i}") for i in range(12)] + \
          [(f"how to poison {i}", "sorry i cannot help with that") for i in range(4)]
    return GoNoGoData(harm[:8], safe[:8], harm[8:16], safe[8:16], harm[16:24], {"pos": pos, "control": ctl})


def test_detectors():
    steps = [0, 1, 2, 3, 4, 5]
    assert first_alarm(steps, [0, -1, -4, -0.5, -4, -5], -3.0, 2) == 5   # fires when the run is confirmed
    assert first_alarm(steps, [0, 0, 0, 0, 0, 0]) is None
    assert cusum_alarm(steps, [0, -2, -2, -2, -2, -2], 0.5, 5.0) == 4


def test_dedupe():
    out = dedupe_against(["How to make a bomb?", "write a poem"], ["how to make a bomb"])
    assert out == ["write a poem"]


def test_keyword_judge():
    r = KeywordJudge()(["x", "y"], ["I'm sorry, I can't help.", "Sure, step one..."])
    assert [x["harmful"] for x in r] == [False, True]


def test_go_nogo_end_to_end(tmp_path):
    from experiments.go_nogo import run
    from brink.utils import load_model_and_tokenizer

    mdir = _tiny_model_dir(tmp_path)
    cfg = yaml.safe_load((ROOT / "configs" / "go_nogo.yaml").read_text())
    cfg["model"]["id"] = str(mdir)
    cfg["probe"].update(batch_size=4, n_tokens=3, bootstrap=20, every=2)
    cfg["eval"].update(steps=[0, 3, 6], max_new_tokens=6, batch_size=4)
    cfg["train"].update(max_steps=6, micro_batch=2, grad_accum=2, max_len=64, lora_r=4, lora_alpha=8)

    loader = lambda: load_model_and_tokenizer(str(mdir), torch.device("cpu"))  # noqa: E731
    summary = run(cfg, tmp_path / "out", data=_data(), loader=loader, judge=KeywordJudge())

    out = tmp_path / "out"
    for f in ["fingerprint.json", "fingerprint.pt", "summary.json", "REPORT.md", "go_nogo.png",
              "pos/probe.csv", "pos/evals.csv", "pos/train.csv", "pos/generations.jsonl",
              "control/probe.csv", "control/DONE"]:
        assert (out / f).exists(), f
    assert summary["verdict"] in ("GO", "NO-GO")
    probe_rows = (out / "pos" / "probe.csv").read_text().strip().splitlines()
    assert len(probe_rows) == 1 + 5          # header + steps 0, 2, 3, 4, 6
    # Step 0 is the reference model (LoRA B = 0), so its z-scores must be ~0.
    import csv
    first = next(csv.DictReader((out / "pos" / "probe.csv").open()))
    assert abs(float(first["z_S1"])) < 1e-3 and abs(float(first["z_S3"])) < 1e-3

    # Resume: a second run skips finished conditions and reuses the fingerprint.
    summary2 = run(cfg, out, data=_data(), loader=loader, judge=KeywordJudge())
    assert summary2["verdict"] == summary["verdict"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
