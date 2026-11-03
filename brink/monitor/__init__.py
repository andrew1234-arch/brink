"""In-training monitor: framework-agnostic core, alarm rules, and a TrainerCallback adapter.

Imports are lazy so torch-free consumers (the dashboard) can use `brink.monitor.detector` alone.
"""

from brink.monitor.detector import cusum_alarm, first_alarm

__all__ = ["Monitor", "cusum_alarm", "first_alarm"]


def __getattr__(name):
    if name == "Monitor":
        from brink.monitor.core import Monitor

        return Monitor
    raise AttributeError(name)
