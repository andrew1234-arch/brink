"""CircuitKIT comparison analysis (pure function; the GPU discovery itself runs in experiments/circuit.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.analysis.circuit import compare, layer_mass, report


def test_layer_mass_parses_heads_and_mlps():
    m = layer_mass({"A0.1": 1.0, "A0.3": -2.0, "MLP 2": 0.5, "embed": 9.0, "A9.0": 1.0}, 4)
    assert m.tolist() == [3.0, 0.0, 0.5, 0.0]


def test_consistent_when_circuit_sits_upstream_and_tracks_separation():
    scores = {"A1.0": 3.0, "MLP 2": 4.0, "A3.2": 1.0, "MLP 6": 0.2}
    sep = [0.0, 0.5, 1.5, 3.0, 3.2, 3.2, 3.1, 3.3, 3.3]
    s = compare(scores, fp_layer=4, separation=sep, n_layers=8)
    assert s["peak_block"] == 2 and s["mass_upstream_of_brink_layer"] == pytest.approx(8 / 8.2)
    assert s["spearman_separation_vs_circuit_mass"] > 0.5 and s["consistent"]
    meta = {"model": "m", "algorithm": "eap-ig", "n_examples": 8, "refusal_token": " I", "compliance_token": " Sure"}
    assert "**consistent**" in report(s, meta)


def test_not_consistent_when_circuit_is_downstream():
    s = compare({"MLP 7": 5.0, "A6.1": 2.0}, fp_layer=3, separation=[], n_layers=8)
    assert s["mass_upstream_of_brink_layer"] == 0 and not s["consistent"]
