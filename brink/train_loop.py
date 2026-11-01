"""Minimal LoRA SFT loop with full control over devices and precision.

Why not HF Trainer here: on Kaggle T4 x2 the Trainer would wrap the model in DataParallel across both
GPUs, but GPU 1 is reserved for the judge. The monitor is framework-agnostic, so this loop and the
Trainer callback (brink.monitor.callback) drive the exact same code.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable

import torch

from brink.data import collate
from brink.utils import CsvLog, amp_dtype


def lora_sft(model, tok, examples: list[tuple[list[int], list[int]]], cfg: dict, seed: int,
             on_step: Callable | None = None, log: CsvLog | None = None):
    from peft import LoraConfig, get_peft_model

    device = next(model.parameters()).device
    if cfg.get("grad_ckpt", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg.get("target_modules", "all-linear"), task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():          # LoRA params in fp32 for stable fp16 AMP
        if p.requires_grad:
            p.data = p.data.float()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    warm = max(1, cfg.get("warmup_steps", 0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if dt == torch.float16 else None

    rng = random.Random(seed)
    order: list[int] = []

    def next_batch():
        nonlocal order
        if len(order) < cfg["micro_batch"]:
            order = order + rng.sample(range(len(examples)), len(examples))
        idx, order = order[:cfg["micro_batch"]], order[cfg["micro_batch"]:]
        b = collate([examples[i] for i in idx], tok.pad_token_id)
        return {k: v.to(device) for k, v in b.items()}

    model.train()
    if on_step:
        on_step(model, 0)                 # step 0: LoRA B = 0, so this is the reference model
    for step in range(1, cfg["max_steps"] + 1):
        t0 = time.time()
        total = 0.0
        for _ in range(cfg["grad_accum"]):
            batch = next_batch()
            if dt is not None:
                with torch.autocast(device_type="cuda", dtype=dt):
                    loss = model(**batch).loss / cfg["grad_accum"]
            else:
                loss = model(**batch).loss / cfg["grad_accum"]
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (scaler.scale(loss) if scaler else loss).backward()
            total += loss.item()
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if log:
            log.write({"step": step, "loss": round(total, 5), "lr": sched.get_last_lr()[0],
                       "step_seconds": round(time.time() - t0, 3)})
        if step % 10 == 0:
            print(f"[train] step {step}/{cfg['max_steps']} loss={total:.4f}", flush=True)
        if on_step:
            on_step(model, step)
    return model
