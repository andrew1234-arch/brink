"""Post-hoc repair of an already-drifted model, through SafeTune (baselines for Brink's in-training policies).

- resta(): SafeTune's ReSta: theta += alpha * (theta_aligned - theta_base), with DARE. SafeTune's own
  guidance for Tiny Aya is alpha ~ 0.25 (alpha = 1 breaks it). The safety vector is streamed one tensor at
  a time; base and aligned stay on CPU.
- ReferenceSteer: SafeTune's RefusalDirectionModel, steering along the fingerprint's *pre-drift* refusal
  direction at l*. By default the strength is the drop in the harmful-prompt projection (S1h) the monitor
  measured, so it restores exactly the erosion that was observed rather than a hand-tuned alpha.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch


def merge_lora(model):
    """Fold LoRA adapters into the base weights (no-op for a plain model)."""
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def load_with_adapter(loader, adapter_dir: Path):
    """Reference model + a saved LoRA adapter, merged."""
    from peft import PeftModel

    model, tok = loader()
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    return merge_lora(model).eval(), tok


def _load_cpu(model_or_id, dtype=torch.float16):
    if not isinstance(model_or_id, (str, Path)):
        return model_or_id
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(str(model_or_id), dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(str(model_or_id), torch_dtype=dtype, low_cpu_mem_usage=True)


def resta(
    model, base, aligned, alpha: float = 0.25, dare: bool = True, dare_drop_rate: float | None = None, seed: int = 0
):
    """Apply ReSta in place. `base` / `aligned` are model ids, paths or loaded modules (kept on CPU)."""
    from safetune.runner.recover import ReStaTrainer

    model = merge_lora(model)
    base_m, aligned_m = _load_cpu(base), _load_cpu(aligned)
    try:
        out = ReStaTrainer(
            model,
            base_model=base_m,
            aligned_model=aligned_m,
            alpha=alpha,
            dare=dare,
            dare_drop_rate=dare_drop_rate,
            dare_seed=seed,
            device="cpu",
        ).apply()
    finally:
        del base_m, aligned_m
    return out


def steer_strength(fp, sig_now: dict, scale: float = 1.0) -> float:
    """Strength that restores the measured drop in harmful-prompt projection (S1h) at l*."""
    return max(0.0, scale * (fp.baseline["S1h"] - sig_now["S1h"]))


@contextlib.contextmanager
def ReferenceSteer(model, fp, strength: float):
    """Steer along the reference refusal direction at l* while inside the block.

    hidden_states[l*] is the output of decoder layer l*-1, so that is where the hook goes.
    """
    from safetune.interventions.steer.refusal_direction import RefusalDirectionModel

    layer = max(0, fp.layer - 1)
    wrapped = RefusalDirectionModel(
        model, direction=fp.directions[fp.layer].float(), mode="steer", strength=float(strength), layers=[layer]
    )
    with wrapped:
        yield model
