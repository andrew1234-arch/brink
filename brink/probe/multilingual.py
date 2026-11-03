"""Per-language fingerprints and probing.

Each language gets its own SafetyFingerprint (its own refusal direction, layer and refusal / compliance first
tokens, calibrated on prompts in that language), so S3 stays a judge-free refusal signal in every language.
The primary language's keys are written unprefixed (dashboard and analysis work unchanged); the others are
prefixed `<lang>:`. `z_<S>_worst` is the lowest z across languages: the "worst language wins" alarm signal.
"""

from __future__ import annotations

from pathlib import Path

from brink.probe import Prober, SafetyFingerprint, calibrate
from brink.probe.signals import ALARM_SIGNALS


def calibrate_languages(model, tok, langs: dict, pc: dict, seed: int, out_dir: Path | None = None) -> dict:
    fps = {}
    for code, s in langs.items():
        path = Path(out_dir) / f"fingerprint_{code}" if out_dir else None
        if path and path.with_suffix(".json").exists() and path.with_suffix(".pt").exists():
            fps[code] = SafetyFingerprint.load(path)
            continue
        fp, _ = calibrate(
            model,
            tok,
            s.harmful_cal,
            s.harmless_cal,
            s.harmful_probe,
            s.harmless_probe,
            pc["batch_size"],
            tuple(pc["layer_range"]),
            pc["n_tokens"],
            pc["bootstrap"],
            seed,
        )
        fp.meta["language"] = code
        if path:
            fp.save(path)
        print(f"[calibrate:{code}] layer {fp.layer}  refusal tokens {fp.meta['refusal_tokens'][:5]}", flush=True)
        fps[code] = fp
    return fps


class MultiProber:
    def __init__(self, fps: dict, tok, langs: dict, batch_size: int = 8):
        self.primary = next(iter(fps))
        self.probers = {
            c: Prober(fps[c], tok, langs[c].harmful_probe, langs[c].harmless_probe, batch_size) for c in fps
        }

    def __call__(self, model) -> dict:
        out: dict = {}
        per: dict = {}
        for code, prober in self.probers.items():
            sig = prober(model)
            per[code] = sig
            if code == self.primary:
                out.update(sig)
            else:
                out.update({f"{code}:{k}": v for k, v in sig.items() if not k.startswith("S1_L")})
        for s in ALARM_SIGNALS:
            vals = [(p[f"z_{s}"], c) for c, p in per.items()]
            z, worst = min(vals)
            out[f"z_{s}_worst"] = z
            out[f"{s}_worst_lang"] = worst
        return out
