"""Repair: what Brink does when safety drift is detected.

In-training (Brink): alarm-triggered policies (halt, rollback to the last safe LoRA snapshot, refusal guard)
in `controller`, and the refusal guard (gradient surgery against the monitor's objective J) in `guard`.
Post-hoc (SafeTune baselines): ReSta and reference-direction steering in `posthoc` (needs `safetune`).
"""

from brink.repair.controller import POLICIES, RepairController, SnapshotBuffer
from brink.repair.guard import RefusalGuard

__all__ = ["POLICIES", "RefusalGuard", "RepairController", "SnapshotBuffer"]
