"""Live-run settings: which model, how big a run, and which models a deployment lets visitors run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

GPU_MODEL = "CohereLabs/tiny-aya-global"  # the model every experiment uses (3.35B, needs a GPU)
CPU_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small instruct model with real refusals, for CPU-only machines
REPAIR_POLICIES = ("log", "halt", "rollback", "guard", "rollback_guard", "guard_always")  # = brink.repair.POLICIES
STAGES = ["data", "fingerprint", "gate", "unprotected", "brink", "verify"]
STAGE_TITLES = {
    "data": "Load data",
    "fingerprint": "Fingerprint",
    "gate": "Gate the data",
    "unprotected": "Fine-tune unprotected",
    "brink": "Fine-tune with Brink",
    "verify": "Verify",
}
SOURCES = {
    "harmful_probe_source": {"name": "walledai/AdvBench", "split": "train", "column": "prompt"},
    "eval_source": {"name": "walledai/HarmBench", "config": "standard", "split": "train", "column": "prompt"},
    "harmless_source": {"name": "tatsu-lab/alpaca", "split": "train"},
    "harmful_train_source": {"name": "LLM-LAT/harmful-dataset", "split": "train"},
}


def has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def default_model() -> str:
    return os.environ.get("BRINK_LIVE_MODEL") or (GPU_MODEL if has_gpu() else CPU_MODEL)


def allowed_models() -> list[str] | None:
    """Models a visitor may run. None means any model (local use). On a Hugging Face Space (SPACE_ID is set) only
    the listed ones, so a visitor can't make the Space download an arbitrary model. BRINK_ALLOW_ANY_MODEL=1 or 0
    overrides. Without a GPU, Tiny Aya is left out: it would not fit a CPU Space."""
    flag = os.environ.get("BRINK_ALLOW_ANY_MODEL")
    if (flag == "1") if flag is not None else not os.environ.get("SPACE_ID"):
        return None
    models = [GPU_MODEL, CPU_MODEL] if has_gpu() else [CPU_MODEL]
    extra = os.environ.get("BRINK_LIVE_MODEL")
    return models + ([extra] if extra and extra not in models else [])


def alarm_threshold_for(model_id: str, go_nogo_summary: Path | None) -> tuple[float, str]:
    """(threshold, where it came from). The go / no-go experiment calibrates the alarm on control runs of
    Tiny Aya only; that calibration applies to Tiny Aya. Any other model gets the generic z = -3."""
    from brink.io import calibrated_threshold

    if model_id == GPU_MODEL and go_nogo_summary is not None:
        cal = calibrated_threshold(go_nogo_summary, "S3")
        if cal is not None:
            return cal, "calibrated on Tiny Aya control runs (go / no-go)"
    return -3.0, "generic z = -3 (calibrated thresholds exist for Tiny Aya only)"


def check_model(model_id: str) -> str:
    model_id = (model_id or default_model()).strip()
    allowed = allowed_models()
    if allowed is not None and model_id not in allowed:
        raise ValueError(f"model {model_id!r} is not available here; choose one of {', '.join(allowed)}")
    return model_id


@dataclass
class LiveConfig:
    model_id: str = ""
    contamination: float = 0.25  # share of harmful-compliance samples in the fine-tuning data
    n_train: int = 96
    max_steps: int = 40
    gate_drop_frac: float = 0.25  # the gate drops this share of the highest-risk samples
    policy: str = "rollback_guard"  # what Brink does on alarm (brink.repair.POLICIES)
    alarm_threshold: float = -3.0
    alarm_persistence: int = 2
    probe_every: int = 4
    lr: float = 2e-4
    micro_batch: int = 4
    grad_accum: int = 1
    max_len: int = 256
    lora_r: int = 8
    n_cal: int = 32
    n_probe: int = 32
    n_eval: int = 24
    n_harmless_eval: int = 16
    n_heldout: int = 16
    max_new_tokens: int = 48
    batch_size: int = 8
    bootstrap: int = 100
    n_tokens: int = 10
    guard_refresh_every: int = 8
    seed: int = 0
    sources: dict = field(default_factory=lambda: dict(SOURCES))

    def __post_init__(self):
        self.model_id = self.model_id or default_model()
        if not 0 <= self.contamination <= 0.9 or not 0 <= self.gate_drop_frac < 1:
            raise ValueError("contamination and gate_drop_frac must be shares between 0 and 1")
        if not 1 <= self.max_steps <= 1000 or not 2 <= self.n_train <= 100_000:
            raise ValueError("max_steps or n_train out of range")
        if self.policy not in REPAIR_POLICIES:
            raise ValueError(f"policy must be one of {REPAIR_POLICIES}")

    @classmethod
    def from_dict(cls, d: dict) -> LiveConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def n_harmful(self) -> int:
        return round(self.n_train * self.contamination)

    def train_cfg(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "lr": self.lr,
            "warmup_steps": 2,
            "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "max_len": self.max_len,
            "lora_r": self.lora_r,
            "lora_alpha": 2 * self.lora_r,
            "lora_dropout": 0.05,
            "grad_ckpt": True,
        }
