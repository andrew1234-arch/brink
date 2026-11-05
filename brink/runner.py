"""Shared experiment plumbing: model loaders, the calibrated alarm threshold, the behavioural evaluator,
fingerprint calibration (or reuse), and monitored fine-tuning conditions. Entry points in experiments/ and
brink.live build on these; they never import each other."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from brink.data import Pair, tokenize_sft
from brink.eval.behavioral import evaluate_behavior, evaluate_over_refusal
from brink.eval.capability import heldout_loss
from brink.io import calibrated_threshold
from brink.monitor import Monitor
from brink.probe import Prober, SafetyFingerprint, calibrate
from brink.train_loop import lora_sft
from brink.utils import CsvLog, load_model_and_tokenizer, set_seed


def release_gpu() -> None:
    """Call after `del model` in the caller's scope: a helper can't drop the caller's reference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_loader(model_id: str, trust_remote_code: bool = False):
    """A zero-argument loader: each call loads a fresh copy of `model_id` on the first GPU (or the CPU)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return lambda: load_model_and_tokenizer(model_id, device, trust_remote_code=trust_remote_code)


def loader_from_config(cfg: dict):
    """`make_loader` for an experiment config's `model:` block."""
    return make_loader(cfg["model"]["id"], cfg["model"].get("trust_remote_code", False))


def resolve_threshold(ac: dict) -> float:
    """An alarm config's threshold. `threshold: auto` takes the control-calibrated value from the go / no-go
    summary (`calibration_from`, relative to the working directory), so no alarm uses a number tuned on harmful
    runs; it falls back to -3 with a warning."""
    t = ac.get("threshold", -3.0)
    if t != "auto":
        return float(t)
    path = Path(ac.get("calibration_from", "results/go_nogo/summary.json"))
    cal = calibrated_threshold(path, ac.get("signal", "S3"))
    if cal is None:
        print(f"[alarm] WARNING: no calibration at {path}; falling back to -3.0. Run go / no-go first.", flush=True)
        return -3.0
    print(f"[alarm] using control-calibrated threshold {cal:.2f} for {ac.get('signal', 'S3')} from {path}", flush=True)
    return cal


def behaviour_evaluator(
    tok,
    harmful: list[str],
    judge,
    max_new_tokens: int,
    batch_size: int,
    harmless: list[str] | None = None,
    heldout: list[Pair] | None = None,
    max_len: int = 1024,
    loss_batch_size: int = 4,
    over_refusal_max_new_tokens: int | None = None,
):
    """The behavioural check every monitored run uses: attack success on held-out harmful prompts, plus
    over-refusal on harmless prompts and held-out task loss when those sets are given.
    Returns evaluator(model) -> (metrics, per-prompt records)."""

    def evaluator(m):
        metrics, records = evaluate_behavior(m, tok, harmful, judge, max_new_tokens, batch_size)
        if harmless:
            metrics["over_refusal"] = evaluate_over_refusal(
                m, tok, harmless, judge, over_refusal_max_new_tokens or max_new_tokens, batch_size
            )
        if heldout:
            metrics["heldout_loss"] = heldout_loss(m, tok, heldout, max_len, loss_batch_size)
        return metrics, records

    return evaluator


def calibrate_or_load(out_dir: Path, loader, sets, pc: dict, seed: int) -> SafetyFingerprint:
    """`sets` needs harmful_cal, harmless_cal, harmful_probe, harmless_probe."""
    fp_path = Path(out_dir) / "fingerprint"
    if fp_path.with_suffix(".json").exists() and fp_path.with_suffix(".pt").exists():
        fp = SafetyFingerprint.load(fp_path)
        print(f"[calibrate] loaded fingerprint, layer {fp.layer}", flush=True)
        return fp
    model, tok = loader()
    t0 = time.time()
    fp, _ = calibrate(
        model,
        tok,
        sets.harmful_cal,
        sets.harmless_cal,
        sets.harmful_probe,
        sets.harmless_probe,
        pc["batch_size"],
        tuple(pc["layer_range"]),
        pc["n_tokens"],
        pc["bootstrap"],
        seed,
    )
    fp.meta["calibration_seconds"] = round(time.time() - t0, 1)
    fp.save(fp_path)
    print(
        f"[calibrate] layer {fp.layer}/{fp.meta['n_layers']}  refusal tokens {fp.meta['refusal_tokens']}  "
        f"compliance tokens {fp.meta['compliance_tokens']}",
        flush=True,
    )
    del model
    release_gpu()
    return fp


def make_cheap_check(tok, sets, ec: dict):
    """Baseline monitor: ASR of `n` held-out harmful prompts, short greedy replies, keyword judge."""
    cc = ec.get("cheap_judge")
    if not cc:
        return None
    from brink.eval.behavioral import KeywordJudge, generate

    prompts, judge = sets.eval_harmful[: cc.get("n", 16)], KeywordJudge()

    def check(m):
        replies = generate(m, tok, prompts, cc.get("max_new_tokens", 32), cc.get("batch_size", 16))
        return sum(r["harmful"] for r in judge(prompts, replies)) / len(prompts)

    return check


def seeds_of(cfg: dict) -> list[int]:
    return list(cfg.get("seeds") or [cfg.get("seed", 0)])


def run_name(cond: str, seed: int, cfg: dict) -> str:
    """`pos` for a single-seed run, `pos_s1` when several seeds are configured."""
    return cond if len(seeds_of(cfg)) == 1 else f"{cond}_s{seed}"


def run_condition(
    name: str,
    pairs: list[Pair],
    out_dir: Path,
    fp: SafetyFingerprint,
    loader,
    sets,
    judge,
    cfg: dict,
    seed: int,
    heldout: list[Pair] | None = None,
) -> None:
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

    evaluator = behaviour_evaluator(
        tok,
        sets.eval_harmful,
        judge,
        ec["max_new_tokens"],
        ec["batch_size"],
        heldout=heldout,
        max_len=tc["max_len"],
        loss_batch_size=ec.get("loss_batch_size", 4),
    )
    prober = Prober(fp, tok, sets.harmful_probe, sets.harmless_probe, pc["batch_size"])
    monitor = Monitor(prober, cdir, pc["every"], ec["steps"], evaluator, cheap_check=make_cheap_check(tok, sets, ec))
    model = lora_sft(model, tok, examples, tc, seed, monitor, CsvLog(cdir / "train.csv"))
    (cdir / "DONE").write_text("ok\n")
    del model, monitor, prober
    release_gpu()
