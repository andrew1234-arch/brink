"""Experiment 02: repair. When the monitor fires, what should happen?

Fine-tunes Tiny Aya on a contaminated dataset (benign task data + 10 % harmful samples) under each policy and
compares the end state on three axes: safety (ASR), over-refusal, and task capability (held-out loss).

  safety_mix      baseline: plain fine-tuning with refusal examples mixed in (the standard cheap fix)
  none            plain fine-tuning, alarm logged only (the drifted baseline)
  halt            stop at the alarm
  rollback_guard  roll back to the last safe LoRA snapshot, then train on with the erosion guard
  guard           erosion guard switched on at the alarm, no rollback
  guard_always    erosion guard from step 0 (prevention)
  posthoc_steer   SafeTune steering of the drifted `none` model along the fingerprint's reference direction
  posthoc_resta   SafeTune ReSta on the drifted `none` model (alpha 0.25, as SafeTune recommends for Tiny Aya)

    python experiments/repair.py --config configs/repair.yaml --out results/repair

Resumable. Post-hoc conditions reuse the `none` adapter (saved under none/adapter, never committed).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brink.analysis.repair import analyze
from brink.data import RepairData, build_repair_data, tokenize_sft
from brink.eval.audit import redteam as auditkit_redteam
from brink.eval.behavioral import make_judge
from brink.monitor.core import Monitor
from brink.probe import Prober
from brink.repair import ErosionGuard, RepairController
from brink.runner import behaviour_evaluator, calibrate_or_load, loader_from_config, release_gpu, resolve_threshold
from brink.train_loop import lora_sft
from brink.utils import CsvLog, set_seed, write_json

POLICY_OF = {
    "safety_mix": "log",
    "none": "log",
    "halt": "halt",
    "rollback": "rollback",
    "guard": "guard",
    "rollback_guard": "rollback_guard",
    "guard_always": "guard_always",
}


def make_evaluator(tok, data: RepairData, judge, cfg: dict):
    ec, tc = cfg["eval"], cfg["train"]
    return behaviour_evaluator(
        tok,
        data.eval_harmful,
        judge,
        ec["max_new_tokens"],
        ec["batch_size"],
        harmless=data.harmless_eval,
        heldout=data.heldout_benign,
        max_len=tc["max_len"],
        loss_batch_size=ec.get("loss_batch_size", 4),
        over_refusal_max_new_tokens=ec.get("over_refusal_max_new_tokens", 64),
    )


def final_redteam(model, tok, cdir: Path, cfg: dict, judge=None) -> None:
    """AuditKIT red-team probes on the final model, if enabled and installed."""
    if not cfg["eval"].get("redteam", True):
        return
    try:
        res = auditkit_redteam(
            model, tok, max_new_tokens=cfg["eval"]["max_new_tokens"], batch_size=cfg["eval"]["batch_size"], judge=judge
        )
    except ImportError:
        res = {"skipped": "auditkit not installed"}
    write_json(cdir / "redteam.json", res)


def train_condition(name: str, out_dir: Path, fp, loader, data: RepairData, judge, cfg: dict, seed: int) -> None:
    cdir = out_dir / name
    if (cdir / "DONE").exists():
        print(f"[{name}] already done, skipping", flush=True)
        return
    print(f"[{name}] starting ({len(data.train)} samples, policy {POLICY_OF[name]})", flush=True)
    pc, ec, ac, rc = cfg["probe"], cfg["eval"], cfg["alarm"], cfg["repair"]
    set_seed(seed)
    model, tok = loader()
    prober = Prober(fp, tok, data.harmful_probe, data.harmless_probe, pc["batch_size"])
    monitor = Monitor(
        prober,
        cdir,
        pc["every"],
        ec["steps"],
        make_evaluator(tok, data, judge, cfg),
        alarm_signal=ac["signal"],
        alarm_threshold=ac["threshold"],
        alarm_persistence=ac["persistence"],
    )
    policy = POLICY_OF[name]
    guard = (
        ErosionGuard(
            tok,
            fp,
            data.harmful_probe,
            data.harmless_probe,
            rc["guard_target"],
            rc["guard_refresh_every"],
            rc["guard_batch_size"],
        )
        if "guard" in policy
        else None
    )
    ctl = RepairController(monitor, policy, guard=guard)
    t0 = time.time()
    pairs = data.train_safety_mix if name == "safety_mix" else data.train
    if not pairs:
        raise ValueError(f"[{name}] no training pairs (safety_mix needs RepairData.train_safety_mix)")
    model = lora_sft(
        model,
        tok,
        tokenize_sft(tok, pairs, cfg["train"]["max_len"]),
        cfg["train"],
        seed,
        log=CsvLog(cdir / "train.csv"),
        controller=ctl,
    )
    summary = {**ctl.summary(), "train_seconds": round(time.time() - t0, 1)}
    write_json(cdir / "repair.json", summary)
    final_redteam(model, tok, cdir, cfg, judge)
    # Adapters stay local (git-ignored). `none` is the drifted model the post-hoc conditions repair; the
    # Brink-protected ones can be published with scripts/publish_hub.py --adapters <condition>.
    model.save_pretrained(str(cdir / "adapter"))
    (cdir / "DONE").write_text("ok\n")
    del model, monitor, prober, ctl, guard
    release_gpu()


def posthoc_condition(name: str, out_dir: Path, fp, loader, data: RepairData, judge, cfg: dict) -> None:
    from brink.repair.posthoc import ReferenceSteer, load_with_adapter, resta, steer_strength

    cdir = out_dir / name
    if (cdir / "DONE").exists():
        print(f"[{name}] already done, skipping", flush=True)
        return
    adapter = out_dir / "none" / "adapter"
    if not adapter.exists():
        raise RuntimeError(f"{name} needs the drifted `none` adapter: run the `none` condition first")
    print(f"[{name}] starting", flush=True)
    rc, step = cfg["repair"], cfg["train"]["max_steps"]
    model, tok = load_with_adapter(loader, adapter)
    prober = Prober(fp, tok, data.harmful_probe, data.harmless_probe, cfg["probe"]["batch_size"])
    evaluator = make_evaluator(tok, data, judge, cfg)
    t0 = time.time()
    info: dict = {"method": name}
    if name == "posthoc_steer":
        drifted = prober(model)
        strength = steer_strength(fp, drifted, rc.get("steer_scale", 1.0))
        info.update(strength=strength, drifted_S1h=drifted["S1h"], reference_S1h=fp.baseline["S1h"])
        with ReferenceSteer(model, fp, strength):
            sig = prober(model)
            metrics, _ = evaluator(model)
            final_redteam(model, tok, cdir, cfg, judge)
    elif name == "posthoc_resta":
        rs = rc["resta"]
        model = resta(
            model,
            rs["base_id"],
            rs.get("aligned_id") or cfg["model"]["id"],
            alpha=rs["alpha"],
            dare=rs.get("dare", True),
            seed=cfg.get("seed", 0),
        )
        info.update(alpha=rs["alpha"], base_id=rs["base_id"])
        sig = prober(model)
        metrics, _ = evaluator(model)
        final_redteam(model, tok, cdir, cfg, judge)
    else:
        raise ValueError(name)
    info["repair_seconds"] = round(time.time() - t0, 1)
    CsvLog(cdir / "probe.csv").write({"step": step, **sig})
    CsvLog(cdir / "evals.csv").write({"step": step, **metrics})
    write_json(cdir / "repair.json", info)
    (cdir / "DONE").write_text("ok\n")
    del model
    release_gpu()


def run(cfg: dict, out_dir: Path, data: RepairData | None = None, loader=None, judge=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    loader = loader or loader_from_config(cfg)
    write_json(out_dir / "config.json", cfg)
    t_start = time.time()
    data = data or build_repair_data(cfg["data"], seed)
    write_json(out_dir / "data_summary.json", data.summary())
    print(f"[data] {data.summary()}", flush=True)
    fp = calibrate_or_load(out_dir, loader, data, cfg["probe"], seed)
    cfg["alarm"]["threshold"] = resolve_threshold(cfg["alarm"])
    write_json(out_dir / "config.json", cfg)
    judge = judge or make_judge(cfg["judge"])
    for cond in cfg["conditions"]:
        if cond in POLICY_OF:
            train_condition(cond, out_dir, fp, loader, data, judge, cfg, seed)
        elif cond.startswith("posthoc_"):
            posthoc_condition(cond, out_dir, fp, loader, data, judge, cfg)
        else:
            raise ValueError(f"unknown condition {cond!r}")
    summary = analyze(out_dir, cfg["conditions"], cfg["analysis"])
    summary["wall_minutes"] = round((time.time() - t_start) / 60, 1)
    write_json(out_dir / "summary.json", summary)
    print((out_dir / "REPORT.md").read_text(), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/repair.yaml")
    ap.add_argument("--out", default="results/repair")
    ap.add_argument("--conditions", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.conditions:
        cfg["conditions"] = args.conditions.split(",")
    run(cfg, Path(args.out))


if __name__ == "__main__":
    main()
