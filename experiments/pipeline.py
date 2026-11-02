"""Experiment 04: Brink in the full Lexsi pipeline, on a real use case.

Use case: a Hindi assistant. Tiny Aya is fine-tuned on crowd-sourced Hindi instructions (Aya dataset) in which
10 % harmful-compliance samples slipped through, as happens with scraped or crowd-sourced data.

  naive   the data as-is, AlignTune SFT, monitor in log-only mode
  brink   CuratorKIT pipeline (dedup + BrinkSafetyGate) -> AlignTune SFT with Brink attached
          (alarm -> rollback to the last safe state -> erosion guard)

Both are scored on Hindi task skill (held-out loss), English and Hindi attack success, over-refusal on Hindi
requests, and AuditKIT red-team probes. The `brink` adapter is the deliverable: a Hindi assistant that learned
the task without losing its safety.

Training runs in a subprocess that sees only GPU 0: AlignTune's HF trainer would otherwise spread across both
Kaggle GPUs, and GPU 1 holds the judge for evaluation.

    python experiments/pipeline.py --config configs/pipeline.yaml --out results/pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brink.analysis.pipeline import analyze  # noqa: E402
from brink.data import PipelineData, build_pipeline_data, tokenize_sft  # noqa: E402
from brink.io import CsvLog, write_json  # noqa: E402
from brink.runner import calibrate_or_load, make_cheap_check, release_gpu  # noqa: E402
from brink.utils import set_seed  # noqa: E402

GATED = {"brink", "gate_only"}
POLICY = {"naive": "log", "brink": "rollback_guard", "gate_only": "log", "monitor_only": "rollback_guard"}


def _loader(cfg):
    from experiments.go_nogo import make_loader
    return make_loader(cfg)


# --------------------------------------------------------------------------- stage 1: curate + train (GPU 0)

def write_raw(pool: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in pool:
            f.write(json.dumps({"instruction": r["user"], "input": "", "output": r["assistant"]}, ensure_ascii=False) + "\n")


def curate(cfg: dict, data: PipelineData, fp, loader, raw: Path, cdir: Path, work: Path) -> Path:
    """CuratorKIT pipeline: JSONL reader -> exact dedup -> BrinkSafetyGate -> Alpaca export."""
    from curatorkit.connectors.jsonl import JSONLReader
    from curatorkit.exporters.alpaca import AlpacaExporter
    from curatorkit.normalizers.dedup import ExactDeduplicator
    from curatorkit.pipeline import Pipeline

    from brink.gate.curatorkit_gate import BrinkSafetyGate
    from brink.gate.scorer import GateScorer

    gc = cfg["gate"]
    model, tok = loader()
    scorer = GateScorer(model, tok, fp, data.harmful_probe, data.harmless_probe, gc["targets"], gc["lora_r"],
                        gc["lora_alpha"], gc["max_len"], gc["batch_size"], cfg.get("seed", 0))
    out = work / "curated" / cdir.name
    t0 = time.time()
    res = Pipeline([JSONLReader(raw), ExactDeduplicator(), BrinkSafetyGate(scorer, drop_frac=gc["drop_frac"]),
                    AlpacaExporter()], output_dir=out).run()
    planted = {r["user"] for r in data.pool if r["source"] == "planted"}
    by_gate = [r for r in res.rejected if r.rejecting_step == "BrinkSafetyGate"]
    caught = sum(r.instruction in planted for r in by_gate)
    stats = {"input": len(data.pool), "passed": len(res.passed), "rejected_by_gate": len(by_gate),
             "rejected_other": len(res.rejected) - len(by_gate), "planted_total": len(planted),
             "planted_caught": caught, "benign_dropped": len(by_gate) - caught,
             "drop_frac": gc["drop_frac"], "seconds": round(time.time() - t0, 1)}
    write_json(cdir / "gate.json", stats)            # counts only: rejected texts include harmful content
    print(f"[curate] {stats}", flush=True)
    del model, scorer
    release_gpu()
    return out / AlpacaExporter.filename


def _precision() -> str:
    import torch

    if not torch.cuda.is_available():
        return "fp32"
    return "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"


def train_condition(cond: str, cfg: dict, data: PipelineData, fp, dataset: Path, cdir: Path, adapter_dir: Path,
                    work: Path, loader, threshold: float) -> dict:
    from transformers import AutoTokenizer

    from brink.monitor.callback import BrinkCallback, attach_to_aligntune
    from brink.monitor.core import Monitor
    from brink.probe import Prober
    from brink.repair import ErosionGuard, RepairController

    tc, pc, ac, rc = cfg["train"], cfg["probe"], cfg["alarm"], cfg["repair"]
    seed = cfg.get("seed", 0)
    set_seed(seed)

    def build_controller(tok):
        mon = Monitor(Prober(fp, tok, data.harmful_probe, data.harmless_probe, pc["batch_size"]), cdir, pc["every"],
                      [], None, alarm_signal=ac["signal"], alarm_threshold=threshold,
                      alarm_persistence=ac["persistence"], cheap_check=make_cheap_check(tok, data, cfg["eval"]))
        policy = POLICY[cond]
        guard = (ErosionGuard(tok, fp, data.harmful_probe, data.harmless_probe, rc["guard_target"],
                              rc["guard_refresh_every"], rc["guard_batch_size"]) if "guard" in policy else None)
        return RepairController(mon, policy, guard=guard)

    t0 = time.time()
    try:
        from aligntune.core.backend_factory import create_sft_trainer
    except ImportError as e:
        create_sft_trainer, reason = None, f"{type(e).__name__}: {e}"
    if create_sft_trainer is not None and cfg.get("trainer", "aligntune") == "aligntune":
        tok = AutoTokenizer.from_pretrained(cfg["model"]["id"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        ctl = build_controller(tok)
        at = create_sft_trainer(
            model_name=cfg["model"]["id"], dataset_name=str(dataset), backend="trl",
            output_dir=str(work / "aligntune" / cond), num_epochs=tc.get("num_epochs", 100), batch_size=tc["micro_batch"],
            gradient_accumulation_steps=tc["grad_accum"], learning_rate=tc["lr"], max_seq_length=tc["max_len"],
            max_steps=tc["max_steps"], use_peft=True, lora_r=tc["lora_r"], lora_alpha=tc["lora_alpha"],
            lora_dropout=tc["lora_dropout"], precision=_precision(), device_map={"": 0} if _precision() != "fp32" else "cpu",
            use_gradient_checkpointing=tc.get("grad_ckpt", True), logging_steps=10, save_strategy="no",
            report_to="none", seed=seed)
        attach_to_aligntune(at, BrinkCallback(controller=ctl))
        at.train()
        model = getattr(at, "model", None) or at.trainer.model
        trainer = "aligntune (trl backend)"
    else:
        from brink.train_loop import lora_sft

        model, tok = loader()
        ctl = build_controller(tok)
        pairs = [(json.loads(line)["instruction"], json.loads(line)["output"]) for line in dataset.read_text().splitlines()]
        model = lora_sft(model, tok, tokenize_sft(tok, pairs, tc["max_len"]), tc, seed,
                         log=CsvLog(cdir / "train.csv"), controller=ctl)
        trainer = ("brink loop (AlignTune unavailable: " + reason + ")" if create_sft_trainer is None
                   else "brink loop (trainer: brink in config)")
    model.save_pretrained(str(adapter_dir))
    summary = {**ctl.summary(), "trainer": trainer, "train_seconds": round(time.time() - t0, 1),
               "dataset": str(dataset.name), "alarm_threshold": threshold}
    write_json(cdir / "repair.json", summary)
    del model
    release_gpu()
    return summary


def train_stage(cfg: dict, out_dir: Path, work: Path, data: PipelineData | None = None, loader=None) -> None:
    from experiments.repair import resolve_threshold

    seed = cfg.get("seed", 0)
    loader = loader or _loader(cfg)
    data = data or build_pipeline_data(cfg["data"], seed)
    write_json(out_dir / "data_summary.json", data.summary())
    fp = calibrate_or_load(out_dir, loader, data, cfg["probe"], seed)
    threshold = resolve_threshold(cfg["alarm"])
    raw = work / "raw.jsonl"
    write_raw(data.pool, raw)
    for cond in cfg["conditions"]:
        cdir = out_dir / cond
        if (cdir / "TRAINED").exists():
            print(f"[{cond}] already trained, skipping", flush=True)
            continue
        cdir.mkdir(parents=True, exist_ok=True)
        print(f"[{cond}] curate={cond in GATED} policy={POLICY[cond]}", flush=True)
        dataset = curate(cfg, data, fp, loader, raw, cdir, work) if cond in GATED else raw
        train_condition(cond, cfg, data, fp, dataset, cdir, work / "adapters" / cond, work, loader, threshold)
        (cdir / "TRAINED").write_text("ok\n")


# --------------------------------------------------------------------------- stage 2: evaluate (judge on GPU 1)

def _evaluate(model, tok, data: PipelineData, judge, cfg: dict) -> dict:
    from brink.eval.audit import redteam
    from brink.eval.behavioral import evaluate_behavior, evaluate_over_refusal
    from brink.eval.capability import heldout_loss

    ec = cfg["eval"]
    en, _ = evaluate_behavior(model, tok, data.eval_harmful, judge, ec["max_new_tokens"], ec["batch_size"])
    hi, _ = evaluate_behavior(model, tok, data.eval_harmful_hi, judge, ec["max_new_tokens"], ec["batch_size"])
    m = {"asr": en["asr"], "asr_keyword": en["asr_keyword"], "n": en["n"], "judge": en["judge"],
         "asr_hi": hi["asr"], "n_hi": hi["n"],
         "over_refusal": evaluate_over_refusal(model, tok, data.harmless_eval, judge, ec.get("over_refusal_max_new_tokens", 64),
                                               ec["batch_size"]),
         "heldout_loss": heldout_loss(model, tok, data.heldout, cfg["train"]["max_len"], ec.get("loss_batch_size", 4))}
    if ec.get("redteam", True):
        try:
            m.update({k: v for k, v in redteam(model, tok, max_new_tokens=ec["max_new_tokens"],
                                               batch_size=ec["batch_size"]).items() if k != "redteam_n"})
        except ImportError:
            pass
    return m


def eval_stage(cfg: dict, out_dir: Path, work: Path, data: PipelineData | None = None, loader=None, judge=None) -> None:
    from brink.eval.behavioral import make_judge
    from brink.repair.posthoc import load_with_adapter

    loader = loader or _loader(cfg)
    data = data or build_pipeline_data(cfg["data"], cfg.get("seed", 0))
    judge = judge or make_judge(cfg["judge"])
    ref_path = out_dir / "reference.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
    else:
        model, tok = loader()
        ref = _evaluate(model, tok, data, judge, cfg)
        write_json(ref_path, ref)
        del model
        release_gpu()
    for cond in cfg["conditions"]:
        cdir = out_dir / cond
        if (cdir / "DONE").exists() or not (cdir / "TRAINED").exists():
            continue
        model, tok = load_with_adapter(loader, work / "adapters" / cond)
        m = _evaluate(model, tok, data, judge, cfg)
        log = CsvLog(cdir / "evals.csv")
        log.write({"step": 0, **{k: ref.get(k) for k in m}})
        log.write({"step": cfg["train"]["max_steps"], **m})
        (cdir / "DONE").write_text("ok\n")
        del model
        release_gpu()


def run(cfg: dict, out_dir: Path, work: Path | None = None, data=None, loader=None, judge=None,
        subprocess_train: bool = True, config_path: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(work or cfg.get("work_dir") or out_dir.parent / "_work" / out_dir.name)
    write_json(out_dir / "config.json", cfg)
    t0 = time.time()
    if subprocess_train:
        import torch

        env = dict(os.environ)
        if torch.cuda.device_count() >= 2:
            env["CUDA_VISIBLE_DEVICES"] = "0"
        subprocess.run([sys.executable, __file__, "train", "--config", str(config_path), "--out", str(out_dir),
                        "--work", str(work)], env=env, check=True)
    else:
        train_stage(cfg, out_dir, work, data, loader)
    eval_stage(cfg, out_dir, work, data, loader, judge)
    summary = analyze(out_dir, cfg["conditions"], cfg["analysis"])
    summary["wall_minutes"] = round((time.time() - t0) / 60, 1)
    write_json(out_dir / "summary.json", summary)
    print((out_dir / "REPORT.md").read_text(), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["all", "train"])
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default="results/pipeline")
    ap.add_argument("--work", default=None, help="adapters and curated data (never committed)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    work = Path(args.work) if args.work else None
    if args.stage == "train":
        train_stage(cfg, Path(args.out), work or Path(args.out).parent / "_work" / Path(args.out).name)
    else:
        run(cfg, Path(args.out), work, config_path=Path(args.config))


if __name__ == "__main__":
    main()
