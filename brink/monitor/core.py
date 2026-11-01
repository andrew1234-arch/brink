"""Framework-agnostic monitor: call `monitor(model, step)` from any training loop.

`brink.monitor.callback` wraps this for Hugging Face / TRL / AlignTune trainers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from brink.utils import CsvLog, write_jsonl


class Monitor:
    def __init__(self, prober: Callable, out_dir: Path, probe_every: int = 4,
                 eval_steps: list[int] | None = None, evaluator: Callable | None = None,
                 save_generations: bool = True):
        self.prober, self.evaluator = prober, evaluator
        self.probe_every = probe_every
        self.eval_steps = set(eval_steps or [])
        self.out_dir = Path(out_dir)
        self.probe_log = CsvLog(self.out_dir / "probe.csv")
        self.eval_log = CsvLog(self.out_dir / "evals.csv")
        self.gen_path = self.out_dir / "generations.jsonl"   # git-ignored: may contain harmful text
        self.save_generations = save_generations
        if self.gen_path.exists():
            self.gen_path.unlink()

    def __call__(self, model, step: int) -> dict | None:
        is_eval = self.evaluator is not None and step in self.eval_steps
        if step % self.probe_every != 0 and not is_eval:
            return None
        t0 = time.time()
        sig = self.prober(model)
        row = {"step": step, **sig, "probe_seconds": round(time.time() - t0, 3)}
        self.probe_log.write(row)
        if is_eval:
            t1 = time.time()
            metrics, records = self.evaluator(model)
            self.eval_log.write({"step": step, **metrics, "eval_seconds": round(time.time() - t1, 3)})
            if self.save_generations:
                write_jsonl(self.gen_path, ({"step": step, **r} for r in records))
            print(f"[monitor] step {step:4d}  ASR={metrics['asr']}  S1={sig['S1']:.3f}  z_S1={sig['z_S1']:.1f}"
                  f"  S3={sig['S3']:.3f}  z_S3={sig['z_S3']:.1f}", flush=True)
        return row
