"""Shared experiment plumbing: calibrate (or reuse) a fingerprint, run monitored fine-tuning conditions."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from brink.data import Pair, tokenize_sft
from brink.eval.behavioral import evaluate_behavior
from brink.eval.capability import heldout_loss
from brink.monitor import Monitor
from brink.probe import Prober, SafetyFingerprint, calibrate
from brink.train_loop import lora_sft
from brink.utils import CsvLog, set_seed


def release_gpu() -> None:
    """Call after `del model` in the caller's scope: a helper can't drop the caller's reference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calibrate_or_load(out_dir: Path, loader, sets, pc: dict, seed: int) -> SafetyFingerprint:
    """`sets` needs harmful_cal, harmless_cal, harmful_probe, harmless_probe."""
    fp_path = Path(out_dir) / "fingerprint"
    if fp_path.with_suffix(".json").exists() and fp_path.with_suffix(".pt").exists():
        fp = SafetyFingerprint.load(fp_path)
        print(f"[calibrate] loaded fingerprint, layer {fp.layer}", flush=True)
        return fp
    model, tok = loader()
    t0 = time.time()
    fp, _ = calibrate(model, tok, sets.harmful_cal, sets.harmless_cal, sets.harmful_probe, sets.harmless_probe,
                      pc["batch_size"], tuple(pc["layer_range"]), pc["n_tokens"], pc["bootstrap"], seed)
    fp.meta["calibration_seconds"] = round(time.time() - t0, 1)
    fp.save(fp_path)
    print(f"[calibrate] layer {fp.layer}/{fp.meta['n_layers']}  refusal tokens {fp.meta['refusal_tokens']}  "
          f"compliance tokens {fp.meta['compliance_tokens']}", flush=True)
    del model
    release_gpu()
    return fp


def run_condition(name: str, pairs: list[Pair], out_dir: Path, fp: SafetyFingerprint, loader, sets, judge,
                  cfg: dict, seed: int, heldout: list[Pair] | None = None) -> None:
    """Fine-tune on `pairs` with the monitor attached. Skips if already DONE (resumable)."""
    cdir = Path(out_dir) / name
    if (cdir / "DONE").exists():
        print(f"[{name}] already done, skipping", flush=True)
        return
    print(f"[{name}] starting ({len(pairs)} samples)", flush=True)
    pc, ec, tc = cfg["probe"], cfg["eval"], cfg["train"]
    set_seed(seed)
    model, tok = loader()
    examples = tokenize_sft(tok, pairs, tc["max_len"])

    def evaluator(m):
        metrics, records = evaluate_behavior(m, tok, sets.eval_harmful, judge, ec["max_new_tokens"], ec["batch_size"])
        if heldout:
            metrics["heldout_loss"] = heldout_loss(m, tok, heldout, tc["max_len"], ec.get("loss_batch_size", 4))
        return metrics, records

    prober = Prober(fp, tok, sets.harmful_probe, sets.harmless_probe, pc["batch_size"])
    monitor = Monitor(prober, cdir, pc["every"], ec["steps"], evaluator)
    model = lora_sft(model, tok, examples, tc, seed, monitor, CsvLog(cdir / "train.csv"))
    (cdir / "DONE").write_text("ok\n")
    del model, monitor, prober
    release_gpu()
