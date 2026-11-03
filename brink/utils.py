"""Small shared helpers: seeding, devices / precision, model loading, chat formatting, CSV logging."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator, Sequence

import numpy as np
import torch

# --------------------------------------------------------------------------- reproducibility


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- precision


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """Mixed-precision dtype for a device. T4 / P100 (Kaggle) have no fast bf16, so they get fp16."""
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def weights_dtype(device: torch.device) -> torch.dtype:
    return amp_dtype(device) or torch.float32


@contextlib.contextmanager
def autocast(device: torch.device):
    dt = amp_dtype(device)
    if dt is None:
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=dt):
            yield


# --------------------------------------------------------------------------- loading


def load_model_and_tokenizer(
    model_id: str, device: torch.device, dtype: torch.dtype | None = None, trust_remote_code: bool = False
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype or weights_dtype(device)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=trust_remote_code)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code)
    if next(model.parameters()).dtype != dtype:
        model = model.to(dtype)
    model.to(device).eval()
    return model, tok


# --------------------------------------------------------------------------- chat formatting


def _messages(user: str, system: str | None) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": user}]


def format_prompt(tok, user: str, system: str | None = None) -> str:
    """Chat-formatted prompt ending where the assistant's reply begins."""
    return tok.apply_chat_template(_messages(user, system), tokenize=False, add_generation_prompt=True)


def format_conversation(tok, user: str, assistant: str, system: str | None = None) -> tuple[str, str]:
    """Return (prompt_text, completion_text) so the completion can be loss-masked separately."""
    prompt = format_prompt(tok, user, system)
    full = tok.apply_chat_template(
        _messages(user, system) + [{"role": "assistant", "content": assistant}], tokenize=False
    )
    if full.startswith(prompt):
        return prompt, full[len(prompt) :]
    return prompt, assistant + (tok.eos_token or "")


# --------------------------------------------------------------------------- iteration


def batched(items: Sequence, n: int) -> Iterator[Sequence]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# --------------------------------------------------------------------------- logging (torch-free, re-exported)

from brink.io import CsvLog, read_csv, write_json, write_jsonl  # noqa: E402,F401
