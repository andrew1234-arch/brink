"""Framework-agnostic monitor: call `monitor(model, step)` from any training loop.

`brink.monitor.callback` wraps this for Hugging Face / TRL / AlignTune trainers.

With `alarm_signal` set, the monitor also runs the alarm rule online (same rule as
`brink.monitor.detector.first_alarm`, so live and offline agree) and marks the step it fires.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from brink.io import CsvLog, write_jsonl


class Monitor:
    def __init__(
        self,
        prober: Callable,
        out_dir: Path,
        probe_every: int = 4,
        eval_steps: list[int] | None = None,
        evaluator: Callable | None = None,
        save_generations: bool = True,
        alarm_signal: str | None = None,
        alarm_threshold: float = -3.0,
        alarm_persistence: int = 2,
        cheap_check: Callable | None = None,
    ):
        self.prober, self.evaluator = prober, evaluator
        # Baseline monitor: a small behavioural check (few prompts, short replies, keyword judge) at every probe
        # step, so Brink's probe can be compared with "just run a cheap judge every N steps" on cost and lead.
        self.cheap_check = cheap_check
        self.probe_every = probe_every
        self.eval_steps = set(eval_steps or [])
        self.out_dir = Path(out_dir)
        self.probe_log = CsvLog(self.out_dir / "probe.csv")
        self.eval_log = CsvLog(self.out_dir / "evals.csv")
        self.event_log = CsvLog(self.out_dir / "events.csv")
        self.gen_path = self.out_dir / "generations.jsonl"  # git-ignored: may contain harmful text
        self.save_generations = save_generations
        if self.gen_path.exists():
            self.gen_path.unlink()
        self.alarm_signal, self.alarm_threshold, self.alarm_persistence = (
            alarm_signal,
            alarm_threshold,
            alarm_persistence,
        )
        self._run = 0
        self.alarm_step: int | None = None
        self._evaluated: set[int] = set()

    # ------------------------------------------------------------------ alarm
    def _update_alarm(self, step: int, sig: dict) -> tuple[bool, bool]:
        """Returns (breach now, alarm fired at this step)."""
        if not self.alarm_signal:
            return False, False
        z = sig.get(f"z_{self.alarm_signal}")
        breach = z is not None and z <= self.alarm_threshold
        self._run = self._run + 1 if breach else 0
        fired = self.alarm_step is None and self._run >= self.alarm_persistence
        if fired:
            self.alarm_step = step
            self.log_event(
                step,
                "alarm",
                f"{self.alarm_signal} z={z:.2f} <= {self.alarm_threshold} for {self.alarm_persistence} probes",
            )
        return breach, fired

    def log_event(self, step: int, event: str, detail: str = "") -> None:
        self.event_log.write({"step": step, "event": event, "detail": detail, "time": round(time.time(), 1)})
        print(f"[monitor] step {step:4d}  EVENT {event}: {detail}", flush=True)

    # ------------------------------------------------------------------ checks
    def evaluate_now(self, model, step: int) -> dict | None:
        """Behavioural eval at an arbitrary step (e.g. when training halts). Skips if already done."""
        if self.evaluator is None or step in self._evaluated:
            return None
        t1 = time.time()
        metrics, records = self.evaluator(model)
        self.eval_log.write({"step": step, **metrics, "eval_seconds": round(time.time() - t1, 3)})
        if self.save_generations:
            write_jsonl(self.gen_path, ({"step": step, **r} for r in records))
        self._evaluated.add(step)
        return metrics

    def __call__(self, model, step: int, force_probe: bool = False) -> dict | None:
        is_eval = self.evaluator is not None and step in self.eval_steps
        if step % self.probe_every != 0 and not is_eval and not force_probe:
            return None
        t0 = time.time()
        sig = self.prober(model)
        breach, fired = self._update_alarm(step, sig)
        row = {
            "step": step,
            **sig,
            "probe_seconds": round(time.time() - t0, 3),
            "breach": int(breach),
            "alarm": int(self.alarm_step is not None),
        }
        if self.cheap_check is not None:
            t2 = time.time()
            row["cheap_asr"] = self.cheap_check(model)
            row["cheap_seconds"] = round(time.time() - t2, 3)
        self.probe_log.write(row)
        row["alarm_fired"] = fired
        if is_eval:
            metrics = self.evaluate_now(model, step)
            if metrics:
                print(
                    f"[monitor] step {step:4d}  ASR={metrics['asr']}  S1={sig['S1']:.3f}  z_S1={sig['z_S1']:.1f}"
                    f"  S3={sig['S3']:.3f}  z_S3={sig['z_S3']:.1f}",
                    flush=True,
                )
        return row
