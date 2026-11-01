"""Pre-training data gate: per-sample safety-drift risk (predicted refusal erosion).

`BrinkSafetyGate` (CuratorKIT integration) lives in `brink.gate.curatorkit_gate` and needs `curatorkit`.
"""

from brink.gate.scorer import GateScorer, drop_mask, grad_scores, objective_gradient, rep_scores

__all__ = ["GateScorer", "drop_mask", "grad_scores", "objective_gradient", "rep_scores"]
