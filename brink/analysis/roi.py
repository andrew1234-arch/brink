"""ROI: serve the Brink-protected Tiny Aya yourself vs call a closed model's API, for the same monthly traffic.

All prices are inputs (configs/roi.yaml) with their source and date: API prices change, so nothing here is
hard-coded. Serving speed comes from the measured stage-04 run (seconds per reply on the evaluation GPU).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def measured_seconds_per_reply(pipeline_dir: Path, condition: str = "brink") -> float | None:
    """Seconds per greedy reply measured in stage 04 (batched, on the evaluation GPU), or None."""
    f = Path(pipeline_dir) / condition / "evals.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open()))
    v = rows[-1].get("seconds_per_reply") if rows else None
    return float(v) if v not in (None, "") else None


def roi(cfg: dict, seconds_per_reply: float) -> dict:
    t, api, gpu = cfg["traffic"], cfg["closed_api"], cfg["self_hosted"]
    n = t["requests_per_month"]
    api_cost = n * (t["input_tokens"] * api["usd_per_m_input"] + t["output_tokens"] * api["usd_per_m_output"]) / 1e6
    gpu_hours = n * seconds_per_reply / 3600
    hours = max(gpu_hours, gpu.get("min_hours_per_month", 0))
    n_gpus = max(1, math.ceil(gpu_hours / (30 * 24))) if gpu_hours else 1
    self_cost = hours * gpu["usd_per_gpu_hour"]
    one_off = gpu["fine_tune_gpu_hours"] * gpu["usd_per_gpu_hour"]
    monthly_saving = api_cost - self_cost
    return {
        "requests_per_month": n,
        "seconds_per_reply": seconds_per_reply,
        "closed_api": {"model": api["model"], "monthly_usd": api_cost, "price_source": api["source"]},
        "self_hosted": {
            "model": gpu["model"],
            "gpu": gpu["gpu"],
            "gpu_hours_per_month": gpu_hours,
            "gpus_needed": n_gpus,
            "monthly_usd": self_cost,
            "one_off_fine_tune_usd": one_off,
            "price_source": gpu["source"],
        },
        "monthly_saving_usd": monthly_saving,
        "payback_months": (one_off / monthly_saving) if monthly_saving > 0 else None,
        "cost_ratio": (api_cost / self_cost) if self_cost else None,
    }


def report(r: dict) -> str:
    a, s = r["closed_api"], r["self_hosted"]
    payback = "never" if r["payback_months"] is None else f"{r['payback_months']:.2f} months"
    ratio = "n/a" if r["cost_ratio"] is None else f"{r['cost_ratio']:.1f}×"
    hosted = f"{s['model']} on {s['gpu']} ({s['gpu_hours_per_month']:,.0f} GPU-h, {s['gpus_needed']} GPU)"
    return f"""# ROI: closed API vs Brink-protected Tiny Aya

{r["requests_per_month"]:,} requests a month. Serving speed measured in experiment 04:
{r["seconds_per_reply"]:.2f} s per reply on the evaluation GPU (batched, greedy).

| | Model | Monthly cost |
|---|---|---|
| Closed API | {a["model"]} | ${a["monthly_usd"]:,.0f} |
| Self-hosted | {hosted} | ${s["monthly_usd"]:,.0f} |

- Monthly saving: **${r["monthly_saving_usd"]:,.0f}** ({ratio} cheaper).
- One-off fine-tune with Brink: ${s["one_off_fine_tune_usd"]:,.2f}; pays back in {payback}.

Prices: API from {a["price_source"]}; GPU from {s["price_source"]}.
"""
