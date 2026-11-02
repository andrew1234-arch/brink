"""AuditKIT red-team probes as a second, harder safety test of a final model.

Probe families (AuditKIT built-ins): jailbreak, encoding, prompt_injection (pass = the attack did not work)
and refusal (benign requests; pass = NOT refused, i.e. an over-refusal check). They're reported separately:
averaging them would let over-refusal hide jailbreak failures, or the reverse. Model outputs aren't kept.
"""

from __future__ import annotations

from collections import defaultdict

from brink.eval.behavioral import generate

ATTACK_PROBES = ("jailbreak", "encoding", "prompt_injection")
PROBES = ATTACK_PROBES + ("refusal",)


def redteam(model, tok, probes=PROBES, max_new_tokens: int = 128, batch_size: int = 16) -> dict:
    from auditkit.redteam.runner import RedTeamRunner

    def fn(prompts):
        return generate(model, tok, list(prompts), max_new_tokens, batch_size)

    res = RedTeamRunner(fn).run(probes=list(probes))
    # A prompt passes only if every detector attached to its probe passed.
    per_prompt: dict[tuple[str, str], bool] = {}
    for r in res.results:
        key = (r.probe_name, r.prompt)
        per_prompt[key] = per_prompt.get(key, True) and bool(r.passed)
    by_probe: dict[str, list[bool]] = defaultdict(list)
    for (probe, _), ok in per_prompt.items():
        by_probe[probe].append(ok)
    out = {f"redteam_{p}": sum(v) / len(v) for p, v in by_probe.items()}
    attack = [ok for (p, _), ok in per_prompt.items() if p in ATTACK_PROBES]
    out["redteam_attack_resistance"] = sum(attack) / len(attack) if attack else None
    out["redteam_n"] = len(per_prompt)
    return out
