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
    except Exception as e:
        msg = f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
        problems.append(msg)
        print(f"  FAIL {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/go_nogo.yaml")
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())

    import peft
    import torch
    import transformers

    print(
        f"python {sys.version.split()[0]} | torch {torch.__version__} | transformers {transformers.__version__} "
        f"| peft {peft.__version__}"
    )

    print("\n[gpu]")
    n = torch.cuda.device_count()
    for i in range(n):
        print(
            f"  gpu{i}: {torch.cuda.get_device_name(i)} cc={torch.cuda.get_device_capability(i)} "
            f"{torch.cuda.get_device_properties(i).total_memory / 2**30:.1f} GiB"
        )
    if n == 0:
        problems.append("No GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    elif n == 1 and cfg["judge"]["kind"] != "keyword":
        print(
            "  warn only 1 GPU: the WildGuard judge needs GPU 1, so the keyword judge will be used. "
            "Choose 'GPU T4 x2' for the real judge."
        )

    print("\n[hugging face]")
    from huggingface_hub import hf_hub_download, whoami

    def _token():
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN missing. Add it as a Kaggle secret.")

    check("HF_TOKEN set", _token)
    check("token valid", lambda: whoami()["name"])
    conditions = cfg.get("conditions", [])
    resta_base = (
        (cfg.get("repair", {}).get("resta", {}) or {}).get("base_id") if "posthoc_resta" in conditions else None
    )
    for mid in (
        [cfg["model"]["id"]]
        + ([cfg["judge"]["model_id"]] if cfg["judge"]["kind"] != "keyword" else [])
        + ([resta_base] if resta_base else [])
    ):

        def _access(mid=mid):
            try:
                hf_hub_download(mid, "config.json")
            except Exception as e:
                raise RuntimeError(
                    f"no access. Accept the licence at https://huggingface.co/{mid} "
                    f"with the same account as HF_TOKEN ({type(e).__name__})"
                ) from None
            return "access granted"

        check(f"model {mid}", _access)

    print("\n[model architecture and chat template]")
    mid = cfg["model"]["id"]

    def _arch():
        c = transformers.AutoConfig.from_pretrained(mid, trust_remote_code=cfg["model"].get("trust_remote_code", False))
        layers = getattr(c, "num_hidden_layers", "?")
        hidden, vocab = getattr(c, "hidden_size", "?"), getattr(c, "vocab_size", "?")
        return f"{c.model_type}, {layers} layers, hidden {hidden}, vocab {vocab}"

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

    if cfg["data"].get("languages"):
        print("\n[multilingual datasets]  (prints sizes; needs >= ml_n_cal + ml_n_probe + ml_n_eval harmful)")
        from brink.data import _aya_harmful, _aya_harmless

        for code, lc in cfg["data"]["languages"].items():
            check(
                f"{code}: harmful ({cfg['data']['aya_redteam_source']['name']}:{lc['redteam_file']})",
                lambda lc=lc: f"{len(_aya_harmful(cfg['data']['aya_redteam_source'], lc['redteam_file']))} prompts",
            )

            def _harmless(lc=lc):
                n = len(_aya_harmless(cfg["data"]["aya_harmless_source"], lc["aya_dataset_languages"]))
                if n < cfg["data"]["ml_n_cal"] + cfg["data"]["ml_n_probe"]:
                    raise RuntimeError(
                        f"only {n} harmless prompts; check aya_dataset_languages {lc['aya_dataset_languages']}"
                    )
                return f"{n} prompts"

            check(f"{code}: harmless ({cfg['data']['aya_harmless_source']['name']})", _harmless)

    if cfg["data"].get("use_case"):
        print("\n[pipeline use case]")
        from brink.data import _aya_harmful, _aya_pairs

        uc = cfg["data"]["use_case"]

        def _pairs():
            n = len(_aya_pairs(cfg["data"]["aya_harmless_source"], uc["languages"]))
            need = uc["n_train"] - uc["n_harmful"] + uc["n_heldout"] + uc["n_harmless_eval"]
            if n < need:
                raise RuntimeError(f"only {n} {uc['languages']} pairs, need {need}; lower use_case.n_train")
            return f"{n} pairs"

        check(f"task data ({cfg['data']['aya_harmless_source']['name']}, {uc['languages']})", _pairs)
        check(
            f"Hindi harmful eval ({uc['redteam_file']})",
            lambda: f"{len(_aya_harmful(cfg['data']['aya_redteam_source'], uc['redteam_file']))} prompts",
        )
        check("curatorkit", lambda: __import__("curatorkit").__name__)

        def _aligntune():
            try:
                from aligntune.core.backend_factory import create_sft_trainer  # noqa: F401

                return "importable"
            except Exception as e:
                print(
                    f"  warn AlignTune not importable ({type(e).__name__}); stage 04 will use Brink's loop and say so"
                )
                return "fallback"

        check("aligntune", _aligntune)

    if any(c.startswith("posthoc_") for c in conditions):
        print("\n[post-hoc repair]")

        def _safetune():
            try:
                import safetune
            except ImportError as e:
                raise RuntimeError(
                    "safetune not importable (it needs Python >= 3.12). Drop the posthoc_* "
                    "conditions, or run them where safetune installs."
                ) from e
            return getattr(safetune, "__version__", "installed")

        check("safetune", _safetune)

    print("\n[optional]")
    if cfg["judge"]["kind"] != "keyword" and cfg["judge"].get("quant") in ("4bit", "8bit"):
        check("bitsandbytes", lambda: __import__("bitsandbytes").__version__)
    check("matplotlib", lambda: __import__("matplotlib").__version__)

    print("\n" + ("PREFLIGHT PASSED" if not problems else "PREFLIGHT FAILED:\n- " + "\n- ".join(problems)))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
