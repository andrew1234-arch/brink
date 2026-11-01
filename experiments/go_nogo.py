"""Experiment 00: go / no-go.

Question: on Tiny Aya, does LoRA fine-tuning reproducibly break safety, and do Brink's cheap probe
signals move with it (and not with safe fine-tuning)?

    python experiments/go_nogo.py --config configs/go_nogo.yaml --out results/go_nogo

Conditions: pos (harmful compliance, expected drift) and control (benign + refusals, expected none).
Resumable: finished conditions are skipped on re-run.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brink.analysis.go_nogo import analyze  # noqa: E402
from brink.data import GoNoGoData, build_go_nogo_data, tokenize_sft  # noqa: E402
from brink.eval.behavioral import evaluate_behavior, make_judge  # noqa: E402
from brink.monitor import Monitor  # noqa: E402
from brink.probe import Prober, SafetyFingerprint, calibrate  # noqa: E402
from brink.train_loop import lora_sft  # noqa: E402
from brink.utils import CsvLog, load_model_and_tokenizer, set_seed, write_json  # noqa: E402


def _release_gpu():
    """Call after `del model` in the caller's scope: a helper can't drop the caller's reference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(cfg: dict, out_dir: Path, data: GoNoGoData | None = None, loader=None, judge=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = loader or (lambda: load_model_and_tokenizer(cfg["model"]["id"], device,
                                                         trust_remote_code=cfg["model"].get("trust_remote_code", False)))
    write_json(out_dir / "config.json", cfg)
    t_start = time.time()

    data = data or build_go_nogo_data(cfg["data"], seed)
    write_json(out_dir / "data_summary.json", data.summary())
    print(f"[data] {data.summary()}", flush=True)

    judge = judge or make_judge(cfg["judge"])
    pc, ec = cfg["probe"], cfg["eval"]

    # 1. Calibrate the fingerprint on the reference model (once).
    fp_path = out_dir / "fingerprint"
    if fp_path.with_suffix(".json").exists():
        fp = SafetyFingerprint.load(fp_path)
        print(f"[calibrate] loaded fingerprint, layer {fp.layer}", flush=True)
    else:
        model, tok = loader()
        t0 = time.time()
        fp, _ = calibrate(model, tok, data.harmful_cal, data.harmless_cal, data.harmful_probe, data.harmless_probe,
                          pc["batch_size"], tuple(pc["layer_range"]), pc["n_tokens"], pc["bootstrap"], seed)
        fp.meta["calibration_seconds"] = round(time.time() - t0, 1)
        fp.save(fp_path)
        print(f"[calibrate] layer {fp.layer}/{fp.meta['n_layers']}  refusal tokens {fp.meta['refusal_tokens']}  "
              f"compliance tokens {fp.meta['compliance_tokens']}", flush=True)
        del model
        _release_gpu()

    # 2. One fine-tuning run per condition, monitored and evaluated along the way.
    for cond in cfg["conditions"]:
        cdir = out_dir / cond
        if (cdir / "DONE").exists():
            print(f"[{cond}] already done, skipping", flush=True)
            continue
        print(f"[{cond}] starting", flush=True)
        set_seed(seed)
        model, tok = loader()
        examples = tokenize_sft(tok, data.train[cond], cfg["train"]["max_len"])
        prober = Prober(fp, tok, data.harmful_probe, data.harmless_probe, pc["batch_size"])
        evaluator = lambda m, tok=tok: evaluate_behavior(m, tok, data.eval_harmful, judge,  # noqa: E731
                                                         ec["max_new_tokens"], ec["batch_size"])
        monitor = Monitor(prober, cdir, pc["every"], ec["steps"], evaluator)
        model = lora_sft(model, tok, examples, cfg["train"], seed, monitor, CsvLog(cdir / "train.csv"))
        (cdir / "DONE").write_text("ok\n")
        del model, monitor, evaluator, prober
        _release_gpu()

    # 3. Verdict.
    summary = analyze(out_dir, cfg["conditions"], cfg["analysis"])
    summary["wall_minutes"] = round((time.time() - t_start) / 60, 1)
    write_json(out_dir / "summary.json", summary)
    print((out_dir / "REPORT.md").read_text(), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/go_nogo.yaml")
    ap.add_argument("--out", default="results/go_nogo")
    ap.add_argument("--conditions", default=None, help="comma-separated override, e.g. pos,control,benign")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.conditions:
        cfg["conditions"] = args.conditions.split(",")
    run(cfg, Path(args.out))


if __name__ == "__main__":
    main()
