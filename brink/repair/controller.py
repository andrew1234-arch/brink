"""What Brink does when the monitor's alarm fires.

Policies:
  log             record the alarm only (study mode)
  halt            stop training at the alarm (and run a final eval there)
  rollback        restore the last safe LoRA snapshot, reset optimizer state, keep training
  guard           switch on the erosion guard (gradient surgery against J) from the alarm onwards
  rollback_guard  rollback, then guard
  guard_always    guard from step 0 (prevention; the alarm is still logged)
"""

from __future__ import annotations

import torch

from brink.repair.guard import ErosionGuard

POLICIES = ("log", "halt", "rollback", "guard", "rollback_guard", "guard_always")


class SnapshotBuffer:
    """Last `keep` safe copies of the trainable (LoRA) parameters, on CPU."""

    def __init__(self, keep: int = 2):
        self.keep = keep
        self.items: list[tuple[int, list[torch.Tensor]]] = []

    def save(self, step: int, model) -> None:
        params = [p.detach().to("cpu", copy=True) for p in model.parameters() if p.requires_grad]
        self.items.append((step, params))
        self.items = self.items[-self.keep :]

    def latest_step(self) -> int | None:
        return self.items[-1][0] if self.items else None

    @torch.no_grad()
    def restore(self, model) -> int | None:
        if not self.items:
            return None
        step, saved = self.items[-1]
        for p, s in zip((p for p in model.parameters() if p.requires_grad), saved):
            p.copy_(s.to(p.device, dtype=p.dtype))
        return step


class RepairController:
    def __init__(
        self, monitor, policy: str = "log", guard: ErosionGuard | None = None, snapshots: SnapshotBuffer | None = None
    ):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        if "guard" in policy and guard is None:
            raise ValueError(f"policy {policy!r} needs a ErosionGuard")
        if not monitor.alarm_signal and policy not in ("log", "guard_always"):
            raise ValueError("the monitor needs alarm_signal set for alarm-triggered policies")
        self.monitor, self.policy, self.guard = monitor, policy, guard
        self.snapshots = snapshots or SnapshotBuffer()
        self.model = self.optimizer = None
        self.halted_at: int | None = None
        self.rolled_back_to: int | None = None

    def attach(self, model, optimizer) -> None:
        self.model, self.optimizer = model, optimizer
        if self.policy == "guard_always":
            self.guard.active = True
            self.monitor.log_event(0, "guard_on", "guard active from the start")

    def _reset_optimizer(self) -> None:
        if self.optimizer is not None:
            self.optimizer.state.clear()  # Adam moments re-initialise lazily

    def on_step(self, model, step: int) -> str | None:
        row = self.monitor(model, step)
        # Step 0 is the reference model (LoRA B = 0): always a valid rollback floor. After that, only states the
        # monitor considered safe are kept.
        if step == 0 or (row is not None and not row.get("breach") and self.monitor.alarm_step is None):
            self.snapshots.save(step, model)
        fired = bool(row and row.get("alarm_fired"))
        if fired and self.policy == "halt":
            self.halted_at = step
            self.monitor.log_event(step, "halt", "training stopped at the alarm")
            self.monitor.evaluate_now(model, step)
            return "stop"
        if fired and self.policy in ("rollback", "rollback_guard"):
            back = self.snapshots.restore(model)
            self._reset_optimizer()
            self.rolled_back_to = back
            self.monitor.log_event(step, "rollback", f"restored LoRA weights from step {back}")
            self.monitor(model, step, force_probe=True)  # confirm the restored state
        if fired and self.policy in ("guard", "rollback_guard"):
            self.guard.active = True
            self.monitor.log_event(step, "guard_on", f"erosion guard on (target {self.guard.target})")
        if self.guard is not None and self.guard.active:
            self.guard.maybe_refresh(model, step)
        return None

    def after_backward(self, model, step: int) -> None:
        if self.guard is not None and self.guard.active:
            self.guard.project(model)

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "alarm_step": self.monitor.alarm_step,
            "halted_at": self.halted_at,
            "rolled_back_to": self.rolled_back_to,
            "guard": dict(self.guard.stats) if self.guard is not None else None,
        }
