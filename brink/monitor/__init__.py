"""In-training monitor: framework-agnostic core, alarm rules, and a TrainerCallback adapter."""

from brink.monitor.core import Monitor
from brink.monitor.detector import cusum_alarm, first_alarm

__all__ = ["Monitor", "first_alarm", "cusum_alarm"]
