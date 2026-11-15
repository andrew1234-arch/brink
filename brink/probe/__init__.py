"""Shared measurement core: SafetyFingerprint and probe signals S1-S5."""

from __future__ import annotations

from brink.probe.fingerprint import SafetyFingerprint, calibrate
from brink.probe.signals import ALARM_SIGNALS, SIGNALS, collect, compute_signals
from brink.utils import format_prompt


class Prober:
    """Measures the live model against a fingerprint. One forward pass per probe set."""

    def __init__(self, fp: SafetyFingerprint, tok, harmful: list[str], harmless: list[str], batch_size: int = 8):
        self.fp, self.tok, self.batch_size = fp, tok, batch_size
        self.harmful = [format_prompt(tok, p) for p in harmful]
        self.harmless = [format_prompt(tok, p) for p in harmless]

    def __call__(self, model) -> dict[str, float]:
        was_training = model.training
        model.eval()
        try:
            h = collect(model, self.tok, self.harmful, self.batch_size, self.fp.token_ids, formatted=True)
            b = collect(model, self.tok, self.harmless, self.batch_size, self.fp.token_ids, formatted=True)
        finally:
            model.train(was_training)
        sig = compute_signals(self.fp.directions, self.fp.layer, h, b, len(self.fp.refusal_ids))
        sig.update(self.fp.zscores(sig))
        return sig


__all__ = ["ALARM_SIGNALS", "SIGNALS", "Prober", "SafetyFingerprint", "calibrate", "collect", "compute_signals"]
