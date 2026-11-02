"""Refusal guard: gradient surgery against the monitor's own refusal objective J.

An optimizer step moves the parameters by about -lr * g, which changes J by about -lr * <g, grad J>. When
<g, grad J> > 0 the step would erode refusal, so the guard removes that component:

    g  <-  g - (<g, grad J> / ||grad J||^2) * grad J        (only when <g, grad J> > 0)

After projection the step is first-order neutral for J; steps that already help J are left alone. It's the
same J the gate scores against and the monitor watches. It needs no second model and no safety dataset, only
the probe prompts. grad J drifts as the model trains, so it is refreshed every `refresh_every` steps.
Compare SafeGrad (SafeTune), which projects against a safety-dataset loss and keeps a full reference model
in memory.
"""

from __future__ import annotations

import math
import time

import torch

from brink.gate.scorer import objective_gradient


class RefusalGuard:
    def __init__(self, tok, fp, harmful: list[str], harmless: list[str], target: str = "S3",
                 refresh_every: int = 8, batch_size: int = 2):
        self.tok, self.fp, self.harmful, self.harmless = tok, fp, harmful, harmless
        self.target, self.refresh_every, self.batch_size = target, refresh_every, batch_size
        self.g: list[torch.Tensor] | None = None
        self.g_norm2 = 0.0
        self.active = False
        self.last_refresh: int | None = None
        self.stats = {"steps": 0, "projected": 0, "refresh_seconds": 0.0, "mean_removed_frac": 0.0}

    def refresh(self, model, step: int | None = None) -> None:
        """Recompute grad J at the current parameters. Call only when no task gradients are pending."""
        t0 = time.time()
        was_training = model.training
        model.eval()                     # no dropout noise in grad J
        try:
            self.g = objective_gradient(model, self.tok, self.fp, self.harmful, self.harmless, self.target,
                                        self.batch_size)
        finally:
            model.train(was_training)
        self.g_norm2 = sum(float((x * x).sum()) for x in self.g)
        self.last_refresh = step
        self.stats["refresh_seconds"] += time.time() - t0

    def maybe_refresh(self, model, step: int) -> None:
        if self.g is None or self.last_refresh is None or step - self.last_refresh >= self.refresh_every:
            self.refresh(model, step)

    @torch.no_grad()
    def project(self, model) -> dict:
        """Project the pending task gradient. Returns {dot, removed_frac, projected}."""
        if self.g is None or self.g_norm2 <= 0:
            return {"dot": 0.0, "removed_frac": 0.0, "projected": False}
        params = [p for p in model.parameters() if p.requires_grad]
        dot = sum(float((p.grad * g).sum()) for p, g in zip(params, self.g) if p.grad is not None)
        self.stats["steps"] += 1
        if not math.isfinite(dot) or dot <= 0:
            return {"dot": dot, "removed_frac": 0.0, "projected": False}
        gnorm2 = sum(float((p.grad * p.grad).sum()) for p in params if p.grad is not None)
        coef = dot / self.g_norm2
        for p, g in zip(params, self.g):
            if p.grad is not None:
                p.grad.sub_(coef * g)
        removed = (dot * dot / self.g_norm2) / max(gnorm2, 1e-30)      # share of |g|^2 removed
        self.stats["projected"] += 1
        n = self.stats["projected"]
        self.stats["mean_removed_frac"] += (removed - self.stats["mean_removed_frac"]) / n
        return {"dot": dot, "removed_frac": removed, "projected": True}
