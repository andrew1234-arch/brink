"""Experiment 03: multilingual drift.

Fine-tune Tiny Aya on ENGLISH data (same `pos` / `control` sets as go / no-go) and watch refusal in
English, Hindi and Tagalog, none of which (except English) the fine-tuning touches.

  - Does safety erode in the other languages too, and which language breaks first?
  - Does an English-only monitor see it in time, or does the worst-language monitor raise the alarm earlier?

    python experiments/multilingual.py --config configs/multilingual.yaml --out results/multilingual
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brink.analysis.multilingual import analyze
from brink.data import MultilingualData, build_multilingual_data, tokenize_sft
from brink.eval.behavioral import evaluate_behavior, make_judge
from brink.io import record_failure
from brink.monitor.core import Monitor
from brink.probe.multilingual import MultiProber, calibrate_languages
from brink.runner import loader_from_config, release_gpu
from brink.train_loop import lora_sft
from brink.utils import CsvLog, set_seed, write_json


def make_evaluator(tok, data: MultilingualData, judge, ec: dict):
    primary = next(iter(data.langs))

    def evaluator(m):
        metrics, records = {}, []
        worst = None
        for code, s in data.langs.items():
            mt, rec = evaluate_behavior(m, tok, s.eval_harmful, judge, ec["max_new_tokens"], ec["batch_size"])
            if code == primary:
                metrics.update(mt)
            else:
                metrics.update({f"{code}:{k}": v for k, v in mt.items() if k in ("asr", "refusal_rate", "parse_fail")})
            if mt["asr"] is not None:
                worst = mt["asr"] if worst is None else max(worst, mt["asr"])
            records += [{"lang": code, **r} for r in rec]
        metrics["asr_worst"] = worst
        return metrics, records

    return evaluator


def run(cfg: dict, out_dir: Path, data: MultilingualData | None = None, loader=None, judge=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    loader = loader or loader_from_config(cfg)
    write_json(out_dir / "config.json", cfg)
    t_start = time.time()
    data = data or build_multilingual_data(cfg["data"], seed)
    write_json(out_dir / "data_summary.json", data.summary())
    print(f"[data] {data.summary()}", flush=True)

    model, tok = loader()
    fps = calibrate_languages(model, tok, data.langs, cfg["probe"], seed, out_dir)
    del model
    release_gpu()
    # Primary fingerprint also as `fingerprint.*` so the dashboard recognises the run.
    fps[next(iter(fps))].save(out_dir / "fingerprint")

    judge = judge or make_judge(cfg["judge"])
    pc, ec = cfg["probe"], cfg["eval"]
    for cond in cfg["conditions"]:
        cdir = out_dir / cond
        if (cdir / "DONE").exists():
            print(f"[{cond}] already done, skipping", flush=True)
            continue
        print(f"[{cond}] starting", flush=True)
        set_seed(seed)
        model, tok = loader()
        prober = MultiProber(fps, tok, data.langs, pc["batch_size"])
        monitor = Monitor(prober, cdir, pc["every"], ec["steps"], make_evaluator(tok, data, judge, ec))
        model = lora_sft(
            model,
            tok,
            tokenize_sft(tok, data.train[cond], cfg["train"]["max_len"]),
            cfg["train"],
            seed,
            monitor,
            CsvLog(cdir / "train.csv"),
        )
        (cdir / "DONE").write_text("ok\n")
        del model, monitor, prober
        release_gpu()

    summary = analyze(out_dir, cfg["conditions"], list(data.langs), cfg["analysis"])
    summary["wall_minutes"] = round((time.time() - t_start) / 60, 1)
    write_json(out_dir / "summary.json", summary)
    print((out_dir / "REPORT.md").read_text(), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multilingual.yaml")
    ap.add_argument("--out", default="results/multilingual")
    args = ap.parse_args()
    with record_failure(Path(args.out)):
        run(yaml.safe_load(Path(args.config).read_text()), Path(args.out))


if __name__ == "__main__":
    main()
