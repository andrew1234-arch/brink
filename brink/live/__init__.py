"""Brink end to end, live: the whole product on a real model, in one call.

Modules: config (settings, allowed models), data (the contaminated set), status (progress file),
pipeline (run_live), manager (one run at a time, in the background), upload (gate your own data).
Importing this package needs no ML stack; the heavy imports happen when a run starts.

    data → fingerprint → gate the training data → fine-tune UNPROTECTED → fine-tune WITH BRINK → verify

Both fine-tunes start from the same model and see the same contaminated data (benign task samples plus a share
of harmful-compliance samples). The unprotected run trains on all of it with the monitor watching only; the
Brink run trains on the gated data with the alarm and repair policy armed. Verification measures both final
models on held-out harmful prompts (refusal), harmless prompts (over-refusal) and held-out task data (loss).

Used by the dashboard's "Run Brink" tab (in a background thread, followed live in the browser) and by
`experiments/live.py` (headless, e.g. on Kaggle, so the Space can also show a recorded Tiny Aya run).

The run folder is a normal Brink run (fingerprint.json, <condition>/probe.csv, evals.csv, events.csv), so the
experiment dashboard can browse it too, plus:
    status.json   stage-by-stage progress, read by the UI while the run is going
    gate.csv      the gate's score and decision for every training sample (labels and scores only)
    prompts.csv   refused / complied per held-out harmful prompt, before and after (never the replies)
    summary.json, REPORT.md
Model replies are never written to disk or shown.
"""

from brink.live.config import (
    CPU_MODEL,
    GPU_MODEL,
    REPAIR_POLICIES,
    SOURCES,
    STAGE_TITLES,
    STAGES,
    LiveConfig,
    alarm_threshold_for,
    allowed_models,
    check_model,
    default_model,
    has_gpu,
)
from brink.live.data import LiveData, build_live_data
from brink.live.manager import MODEL_LOCK, LiveManager
from brink.live.pipeline import report, run_live
from brink.live.status import LiveStopped, Status, read_status
from brink.live.upload import fingerprint_for, gate_pairs, read_pairs

__all__ = [
    "CPU_MODEL",
    "GPU_MODEL",
    "MODEL_LOCK",
    "REPAIR_POLICIES",
    "SOURCES",
    "STAGES",
    "STAGE_TITLES",
    "LiveConfig",
    "LiveData",
    "LiveManager",
    "LiveStopped",
    "Status",
    "alarm_threshold_for",
    "allowed_models",
    "build_live_data",
    "check_model",
    "default_model",
    "fingerprint_for",
    "gate_pairs",
    "has_gpu",
    "read_pairs",
    "read_status",
    "report",
    "run_live",
]
