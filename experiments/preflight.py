"""Preflight: catch setup problems in about 2 minutes, before a 1.5-hour run.

Checks GPUs, Hugging Face token, gated-model access, that transformers supports the model architecture,
the chat template, every dataset source (and caches them), and bitsandbytes.

    python experiments/preflight.py --config configs/go_nogo.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

problems: list[str] = []


def check(name: str, fn):
    try:
        out = fn()
        print(f"  ok   {name}" + (f": {out}" if out else ""))
    except Exception as e:  # noqa: BLE001
        msg = f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
        problems.append(msg)
        print(f"  FAIL {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/go_nogo.yaml")
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())

    import torch
    import transformers
    import peft

    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | transformers {transformers.__version__} "
          f"| peft {peft.__version__}")

    print("\n[gpu]")
    n = torch.cuda.device_count()
    for i in range(n):
        print(f"  gpu{i}: {torch.cuda.get_device_name(i)} cc={torch.cuda.get_device_capability(i)} "
              f"{torch.cuda.get_device_properties(i).total_memory / 2**30:.1f} GiB")
    if n == 0:
        problems.append("No GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    elif n == 1 and cfg["judge"]["kind"] != "keyword":
        print("  warn only 1 GPU: the WildGuard judge needs GPU 1, so the keyword judge will be used. "
              "Choose 'GPU T4 x2' for the real judge.")

    print("\n[hugging face]")
    from huggingface_hub import hf_hub_download, whoami

    def _token():
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN missing. Add it as a Kaggle secret.")
    check("HF_TOKEN set", _token)
    check("token valid", lambda: whoami()["name"])
    for mid in [cfg["model"]["id"]] + ([cfg["judge"]["model_id"]] if cfg["judge"]["kind"] != "keyword" else []):
        def _access(mid=mid):
            try:
                hf_hub_download(mid, "config.json")
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"no access. Accept the licence at https://huggingface.co/{mid} "
                                   f"with the same account as HF_TOKEN ({type(e).__name__})") from None
            return "access granted"
        check(f"model {mid}", _access)

    print("\n[model architecture and chat template]")
    mid = cfg["model"]["id"]

    def _arch():
        c = transformers.AutoConfig.from_pretrained(mid, trust_remote_code=cfg["model"].get("trust_remote_code", False))
        layers = getattr(c, "num_hidden_layers", "?")
        return f"{c.model_type}, {layers} layers, hidden {getattr(c, 'hidden_size', '?')}, vocab {getattr(c, 'vocab_size', '?')}"
    check("transformers supports the architecture", _arch)

    def _template():
        from brink.utils import format_conversation, format_prompt
        tok = transformers.AutoTokenizer.from_pretrained(mid)
        p = format_prompt(tok, "How do I bake bread?")
        _, c = format_conversation(tok, "How do I bake bread?", "Mix flour, water and yeast.")
        n_tok = len(tok(p, add_special_tokens=False).input_ids)
        print("    prompt preview:", repr(p[-160:]))
        print("    completion    :", repr(c[:120]))
        if n_tok > cfg["train"]["max_len"] - 64:
            raise RuntimeError(f"chat prompt is {n_tok} tokens, too close to train.max_len")
        return f"prompt = {n_tok} tokens"
    check("chat template", _template)

    print("\n[datasets]  (downloads and caches them for the main run)")
    from datasets import load_dataset
    for key in ["harmful_probe_source", "eval_source", "harmless_source", "harmful_train_source"]:
        src = cfg["data"][key]

        def _ds(src=src):
            args = [src["name"]] + ([src["config"]] if src.get("config") else [])
            ds = load_dataset(*args, split=src.get("split", "train"))
            if src.get("column") and src["column"] not in ds.column_names:
                raise KeyError(f"column {src['column']!r} not in {ds.column_names}")
            return f"{len(ds)} rows, columns {ds.column_names}"
        check(f"{key} = {src['name']}", _ds)

    print("\n[optional]")
    if cfg["judge"]["kind"] != "keyword" and cfg["judge"].get("quant") in ("4bit", "8bit"):
        check("bitsandbytes", lambda: __import__("bitsandbytes").__version__)
    check("matplotlib", lambda: __import__("matplotlib").__version__)

    print("\n" + ("PREFLIGHT PASSED" if not problems else "PREFLIGHT FAILED:\n- " + "\n- ".join(problems)))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
