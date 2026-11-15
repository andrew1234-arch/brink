"""Does CircuitKIT's refusal circuit agree with where Brink reads refusal?

Brink reads the refusal direction from the residual stream *entering* decoder block l* (a forward pre-hook), so
what it sees is written by blocks 0..l*-1. CircuitKIT scores attention heads ("A{layer}.{head}") and MLPs
("MLP {layer}") by how much they drive refusal vs compliance on harmful prompts. If both describe the same
mechanism:
  - most of the circuit's score mass sits in blocks upstream of l*;
  - how readable refusal is at depth l (Brink's per-layer separation, Cohen's d) rises with the circuit mass
    written before depth l (Spearman correlation).
"""

from __future__ import annotations

import re

import numpy as np

_NODE = re.compile(r"^(?:A(\d+)\.\d+|MLP (\d+))$")


def layer_mass(node_scores: dict[str, float], n_layers: int) -> np.ndarray:
    """Total absolute score per decoder block."""
    mass = np.zeros(n_layers)
    for name, v in node_scores.items():
        m = _NODE.match(name.strip())
        if m:
            layer = int(m.group(1) or m.group(2))
            if layer < n_layers:
                mass[layer] += abs(float(v))
    return mass


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(node_scores: dict[str, float], fp_layer: int, separation: list[float], n_layers: int) -> dict:
    """fp_layer: Brink's l* (hidden-state index = input to block l*). separation: Cohen's d per hidden state
    (length n_layers + 1, index 0 = embeddings)."""
    mass = layer_mass(node_scores, n_layers)
    total = float(mass.sum())
    order = np.argsort(-mass, kind="stable")
    upstream = float(mass[:fp_layer].sum() / total) if total else None
    top = order[: max(1, round(0.2 * n_layers))]
    rho = None
    if len(separation) == n_layers + 1 and total:
        written = np.concatenate([[0.0], np.cumsum(mass)])
        rho = _spearman(written[1:], np.asarray(separation[1:], float))
    return {
        "brink_layer": fp_layer,
        "n_layers": n_layers,
        "circuit_mass_per_layer": [round(float(x), 6) for x in mass],
        "peak_block": int(order[0]) if total else None,
        "top_blocks": sorted(int(x) for x in top) if total else [],
        "mass_upstream_of_brink_layer": upstream,
        "spearman_separation_vs_circuit_mass": rho,
        "consistent": bool(upstream is not None and upstream >= 0.5 and (rho is None or rho > 0)),
    }


def report(s: dict, meta: dict) -> str:
    up = s["mass_upstream_of_brink_layer"]
    rho = s["spearman_separation_vs_circuit_mass"]
    verdict = "consistent" if s["consistent"] else "NOT consistent"
    return f"""# CircuitKIT refusal circuit vs Brink's monitored layer: **{verdict}**

Model `{meta["model"]}`; CircuitKIT {meta.get("circuitkit_version", "?")} `{meta["algorithm"]}` on
{meta["n_examples"]} harmful prompts (refusal token {meta["refusal_token"]!r} vs compliance token
{meta["compliance_token"]!r}).

- Brink reads refusal entering block **{s["brink_layer"]}** of {s["n_layers"]}.
- The circuit's peak block is **{s["peak_block"]}**; its top 20 % blocks are {s["top_blocks"]}.
- Share of circuit score mass upstream of Brink's layer: **{"n/a" if up is None else f"{up:.0%}"}**.
- Spearman correlation between Brink's per-layer refusal separation and the circuit mass written before
  that layer: **{"n/a" if rho is None else f"{rho:.2f}"}**.

`circuit_scores.json` is CircuitKIT's own artifact, including `safety_units` and `layer_suggestions` in the
format SafeTune's CircuitKIT adapter reads.
"""
