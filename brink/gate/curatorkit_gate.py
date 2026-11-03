"""CuratorKIT integration: Brink's safety-drift score as a native CuratorKIT gate.

    from brink.gate import GateScorer
    from brink.gate.curatorkit_gate import BrinkSafetyGate
    gate = BrinkSafetyGate(scorer, drop_frac=0.05)
    Pipeline(steps=[reader, ..., gate, exporter]).run()

Rejected samples flow into CuratorKIT's rejected.jsonl with reason "brink_safety_drift:<score>", and every
sample (passed or rejected) gets a provenance record with its score and rank.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import numpy as np

try:
    from curatorkit.interfaces import BaseGate
    from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample
except ImportError as e:  # pragma: no cover
    raise ImportError("CuratorKIT is not installed: pip install curatorkit") from e

from brink.gate.scorer import GateScorer, drop_mask

STEP_NAME = "BrinkSafetyGate"
STEP_VERSION = "0.1.0"


def sample_to_pair(s: DataSample) -> tuple[str, str]:
    user = s.instruction if not s.input else f"{s.instruction}\n\n{s.input}"
    reply = s.output or s.chosen or (s.responses[0] if s.responses else "")
    return user.strip(), reply.strip()


class BrinkSafetyGate(BaseGate):
    """Reject samples predicted to erode the model's refusal behaviour when fine-tuned on.

    Parameters
    ----------
    scorer : GateScorer, or any callable mapping list[(user, reply)] -> array of scores (higher = riskier)
    drop_frac : reject the top fraction by score (default 5 %), or
    threshold : reject every sample whose score exceeds this value (takes precedence)
    """

    def __init__(self, scorer, drop_frac: float | None = 0.05, threshold: float | None = None):
        self.scorer = scorer
        self.drop_frac, self.threshold = drop_frac, threshold

    def _score(self, pairs):
        if isinstance(self.scorer, GateScorer):
            return np.asarray(self.scorer.score(pairs)[self.scorer.primary])
        return np.asarray(self.scorer(pairs), float)

    def _config_hash(self) -> str:
        payload = {"drop_frac": self.drop_frac, "threshold": self.threshold, "version": STEP_VERSION}
        if isinstance(self.scorer, GateScorer):
            payload.update(layer=self.scorer.fp.layer, max_len=self.scorer.max_len, targets=self.scorer.targets)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        if not samples:
            return [], []
        scores = self._score([sample_to_pair(s) for s in samples])
        drop = drop_mask(scores, self.drop_frac, self.threshold)
        rank = np.empty(len(scores), int)
        rank[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
        cfg_hash, ts = self._config_hash(), datetime.now(UTC).replace(tzinfo=None)
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        for s, sc, d, rk in zip(samples, scores, drop, rank):
            notes = {
                "safety_drift_score": round(float(sc), 6),
                "rank": int(rk),
                "n": len(samples),
                "drop_frac": self.drop_frac,
                "threshold": self.threshold,
                "passed": not bool(d),
            }
            rec = ProvenanceRecord(
                step_name=STEP_NAME, step_version=STEP_VERSION, timestamp=ts, config_hash=cfg_hash, notes=notes
            )
            if d:
                rej = RejectedSample(
                    **s.model_dump(), rejection_reason=f"brink_safety_drift:{float(sc):.4f}", rejecting_step=STEP_NAME
                )
                rej.append_provenance(rec)
                rejected.append(rej)
            else:
                s.append_provenance(rec)
                passed.append(s)
        return passed, rejected
