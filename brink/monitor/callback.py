"""TrainerCallback adapter so the monitor drops into HF Trainer, TRL and AlignTune (TRL backend)."""

from __future__ import annotations

from transformers import TrainerCallback

from brink.monitor.core import Monitor


class BrinkCallback(TrainerCallback):
    def __init__(self, monitor: Monitor):
        self.monitor = monitor

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.monitor(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.monitor(model, state.global_step)
