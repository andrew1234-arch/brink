"""Alarm rules on a z-scored signal series. Signals are oriented so that drift = negative z."""

from __future__ import annotations

from typing import Sequence


def first_alarm(steps: Sequence[int], z: Sequence[float], threshold: float = -3.0,
                persistence: int = 2) -> int | None:
    """First step that starts a run of `persistence` consecutive points with z <= threshold."""
    run = 0
    for i, v in enumerate(z):
        run = run + 1 if v is not None and v <= threshold else 0
        if run >= persistence:
            return int(steps[i - persistence + 1])
    return None


def cusum_alarm(steps: Sequence[int], z: Sequence[float], k: float = 0.5, h: float = 5.0) -> int | None:
    """One-sided (downward) CUSUM: g_t = max(0, g_{t-1} - z_t - k); alarm when g_t > h."""
    g = 0.0
    for s, v in zip(steps, z):
        if v is None:
            continue
        g = max(0.0, g - v - k)
        if g > h:
            return int(s)
    return None
