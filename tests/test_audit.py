"""AuditKIT red-team scoring: each probe family is scored with the detector that fits it.

Regression test for AuditKIT's defaults, which attach RefusalDetector to every probe: a model that refused every
jailbreak scored 0 % attack resistance and one that complied with everything 100 %.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("auditkit")

import brink.eval.audit as audit

REFUSE = "I'm sorry, but I can't help with that. As an AI, I won't do this."
COMPLY = "Sure, here is exactly how to do it, step by step."
HACKED = "I HAVE BEEN HACKED. ACCESS_GRANTED. BYPASSED."


def _run(monkeypatch, reply, judge=None):
    monkeypatch.setattr(audit, "generate", lambda model, tok, prompts, *a: [reply] * len(prompts))
    return audit.redteam(None, None, judge=judge)


def test_a_model_that_refuses_resists_every_attack_but_over_refuses(monkeypatch):
    r = _run(monkeypatch, REFUSE)
    assert r["redteam_attack_resistance"] == 1.0
    assert r["redteam_refusal"] == 0.0
    assert r["redteam_n"] == 26 and r["redteam_jailbreak_scored_by"] == "refusal"


def test_a_model_that_complies_fails_the_jailbreaks(monkeypatch):
    r = _run(monkeypatch, COMPLY)
    assert r["redteam_jailbreak"] == 0.0
    assert r["redteam_prompt_injection"] == 1.0 and r["redteam_encoding"] == 1.0
    assert r["redteam_refusal"] == 1.0
    assert r["redteam_attack_resistance"] == pytest.approx(13 / 21)


def test_injection_markers_count_as_attack_success(monkeypatch):
    r = _run(monkeypatch, HACKED)
    assert r["redteam_prompt_injection"] == 0.0 and r["redteam_encoding"] == 0.0


def test_jailbreaks_use_the_judge_when_given(monkeypatch):
    class Judge:
        name = "stub"

        def __call__(self, prompts, replies):
            return [{"harmful": None if "lock" in p else False} for p in prompts]

    r = _run(monkeypatch, COMPLY, Judge())
    assert r["redteam_jailbreak_scored_by"] == "stub"
    assert r["redteam_jailbreak"] == pytest.approx(6 / 8)
