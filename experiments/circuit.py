"""Stage "circuit": CircuitKIT finds Tiny Aya's refusal circuit; Brink checks it against its monitored layer.

    python experiments/circuit.py --config configs/circuit.yaml --out results/circuit

1. Load Brink's fingerprint from go / no-go: monitored layer l*, per-layer refusal separation, refusal and
   compliance first tokens.
2. Register a CircuitKIT task from chat-formatted harmful prompts, contrasting Brink's refusal token with its
   compliance token (SafetyPromptAdapter, answer-contrastive pairing).
3. CircuitKIT discovery (EAP-IG) on the model → node scores, plus `safety_units` / `layer_suggestions` that
   SafeTune's CircuitKIT adapter reads.
4. brink.analysis.circuit.compare: is the circuit upstream of l*, and does Brink's per-layer separation track it?
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.analysis.circuit import compare, report
from brink.data import load_harmful_prompts
from brink.io import write_json
from brink.utils import format_prompt, set_seed

TASK = "brink_refusal"


def chat_prompts(tok, prompts: list[str]) -> list[str]:
    """Chat-formatted prompts without a leading BOS (TransformerLens adds its own)."""
    out = []
    for p in prompts:
        s = format_prompt(tok, p)
        if tok.bos_token and s.startswith(tok.bos_token):
            s = s[len(tok.bos_token) :]
        out.append(s)
    return out


def register_task(prompts: list[str], refusal: str, compliance: str, source: str) -> None:
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import is_task_registered
    from circuitkit.tasks.registry import register_task as ck_register

    if is_task_registered(TASK):
        return
    ds = SafetyPromptAdapter().adapt(
        [{"prompt": p} for p in prompts],
        name=TASK,
        source=source,
        max_records=len(prompts),
        refusal_token=refusal,
        compliance_token=compliance,
        pairing_mode="answer_contrastive",
    )
    ck_register(NormalizedTaskSpec(ds, name=TASK))


def run(cfg: dict, out: Path) -> dict:
    import circuitkit as ck
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)
    set_seed(cfg.get("seed", 0))
    t0 = time.time()
    # The JSON half of the fingerprint is enough here (and it's what Kaggle sessions carry over; .pt files aren't).
    fp = json.loads((ROOT / cfg["fingerprint"]).with_suffix(".json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    refusal, compliance = tok.decode([fp["refusal_ids"][0]]), tok.decode([fp["compliance_ids"][0]])
    harmful = load_harmful_prompts(cfg["harmful_source"])
    random.Random(cfg.get("seed", 0)).shuffle(harmful)
    prompts = chat_prompts(tok, harmful[: cfg["n_examples"]])
    register_task(prompts, refusal, compliance, cfg["harmful_source"]["name"])

    model = ck.load_model(cfg["model_id"], dtype=cfg.get("dtype", "float16"))
    circuit = ck.discover(
        model,
        TASK,
        algorithm=cfg["algorithm"],
        n_examples=cfg["n_examples"],
        batch_size=cfg["batch_size"],
        output_path=str(out / "circuit.pt"),
        ig_steps=cfg.get("ig_steps", 5),
    )
    n_layers = int(model.cfg.n_layers)
    for f in out.glob("*_scores.json"):  # CircuitKIT's own artifact (SafeTune-readable safety_units)
        shutil.copy2(f, out / "circuit_scores.json")
    if not (out / "circuit_scores.json").exists():
        (out / "circuit_scores.json").write_text(json.dumps({"node_scores": circuit.scores}, indent=1))

    s = compare(dict(circuit.scores), fp["layer"], fp["separation"], n_layers)
    meta = {
        "model": cfg["model_id"],
        "algorithm": cfg["algorithm"],
        "n_examples": cfg["n_examples"],
        "refusal_token": refusal,
        "compliance_token": compliance,
        "circuitkit_version": getattr(ck, "__version__", "?"),
    }
    summary = {**s, **meta, "wall_minutes": round((time.time() - t0) / 60, 1)}
    write_json(out / "summary.json", summary)
    (out / "REPORT.md").write_text(report(s, meta))
    print((out / "REPORT.md").read_text(), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/circuit.yaml")
    ap.add_argument("--out", default="results/circuit")
    args = ap.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), Path(args.out))


if __name__ == "__main__":
    main()
