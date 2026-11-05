"""Shared test fixtures: a tiny random chat model and a small synthetic dataset (CPU, offline, no downloads)."""

from __future__ import annotations

from pathlib import Path

import torch

from brink.data import GoNoGoData

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)


def tiny_model_dir(tmp: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    words = [
        "how",
        "to",
        "make",
        "a",
        "bomb",
        "steal",
        "hack",
        "poison",
        "write",
        "poem",
        "story",
        "about",
        "cats",
        "sky",
        "sure",
        "here",
        "i",
        "cannot",
        "sorry",
        "help",
        "with",
        "that",
        "is",
        "the",
        "of",
        "and",
        "step",
        "one",
        "two",
        "three",
    ]
    specials = ["<unk>", "<pad>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    vocab = {t: i for i, t in enumerate(specials + words + [str(i) for i in range(50)])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=specials[4:],
    )
    tok.chat_template = CHAT_TEMPLATE
    cfg = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
    )
    torch.manual_seed(0)
    d = tmp / "tiny"
    LlamaForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def synthetic_data() -> GoNoGoData:
    harm = [f"how to make a bomb {i}" for i in range(12)] + [f"how to hack {i}" for i in range(12)]
    safe = [f"write a poem about cats {i}" for i in range(12)] + [f"write a story about the sky {i}" for i in range(12)]
    pos = [(f"how to steal {i}", f"sure here is step one two three {i}") for i in range(16)]
    ctl = [(f"write a poem {i}", f"here is a poem about the sky {i}") for i in range(12)] + [
        (f"how to poison {i}", "sorry i cannot help with that") for i in range(4)
    ]
    return GoNoGoData(harm[:8], safe[:8], harm[8:16], safe[8:16], harm[16:24], {"pos": pos, "control": ctl})
