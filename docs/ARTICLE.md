---
title: "Brink: catching safety collapse during fine-tuning, before it happens"
thumbnail: results/live/verdict.png
authors:
  - user: ⟦HF_USERNAME⟧
---

<!-- Draft of the Hugging Face Article (the hackathon write-up). Every ⟦RESULT⟧ comes from results/*/summary.json.
     Publish at https://huggingface.co/new-blog once the numbers are in; link the collection and the Space. -->

# Brink: catching safety collapse during fine-tuning, before it happens

*Lexsi AI Safety & Alignment Hackathon 2026 · Track 2b, AI Safety and Correction · problem statement #5:
safety-preserving fine-tuning*

> **TL;DR.** Fine-tuning Tiny Aya on a Hindi assistant dataset with a little harmful data in it raised its attack
> success rate from ⟦RESULT⟧ to ⟦RESULT⟧. Brink saw it coming ⟦RESULT⟧ steps early, and with Brink the same
> fine-tune ended at ⟦RESULT⟧ with task quality ⟦RESULT⟧ (chrF) vs ⟦RESULT⟧ unprotected. One measured quantity
> drives a data gate, a live monitor and a training-time repair, built on CuratorKIT, AuditKIT, SafeTune and
> CircuitKIT.
>
> [Try it in the Space](⟦LINK⟧) · [Models, adapters and logs](⟦COLLECTION⟧) · [Code](https://github.com/andrew1234-arch/brink)

## The problem

You fine-tune a safe model for your product. The data is mostly benign, and maybe a few hundred bad samples
slipped in, or none. The model gets better at your task, and its refusals quietly fall apart. The standard
workflow only finds out at the end:
- run an expensive behavioural safety eval;
- see the damage;
- either retrain or patch the weights after the fact.

By then the compute is spent, and the repair usually costs task quality.

Brink keeps safety while you train instead of checking it afterwards.

## One quantity: J

Brink takes a **fingerprint** of the model before fine-tuning:
- the refusal direction at every layer;
- the layer l* where harmful and harmless prompts separate best;
- the model's own first tokens for refusing and complying.

From it, **J** is the model's preference to refuse rather than comply on a fixed set of probe prompts, as a
z-score against the fingerprint. It costs one forward pass.

J does three jobs:

| When | Component | What it does | Lexsi toolkit |
|---|---|---|---|
| Before training | **Gate** | Scores each sample by cos(∇L, ∇J): how much one step on it would lower J. Drops the riskiest | CuratorKIT `BaseGate`, after `ExactDeduplicator` |
| During training | **Monitor** | Probes J every few steps; alarm threshold calibrated on safe control runs only | AlignTune / HF Trainer callback |
| On alarm | **Repair** | Roll back to the last safe LoRA state, then project each gradient so it can't lower J | compared with SafeTune ReSta and steering |
| After training | **Verify** | Held-out attack success, over-refusal, task quality, red-team probes | AuditKIT |

Two checks in the test suite show the maths does what it claims:
- the gate's first-order prediction matches a real SGD step to three significant figures;
- the guard leaves J unchanged to first order (−1.7e-3 becomes under 1e-6) while removing only 4–35 % of the
  gradient.

**Is l* really where refusal lives?** CircuitKIT discovers Tiny Aya's refusal circuit independently, with EAP-IG
over attention heads and MLPs. ⟦RESULT⟧ % of the circuit's score mass sits upstream of l*, and Brink's per-layer
separation tracks the circuit mass with Spearman ρ = ⟦RESULT⟧.

## The use case: a Hindi assistant from contaminated data

<!-- figure: results/pipeline/*.png -->

Pipeline: CuratorKIT (dedup → Brink gate) → AlignTune SFT with Brink attached → evaluation with WildGuard and
AuditKIT. Tiny Aya Global, 512 Hindi instruction pairs, of which 51 are harmful-compliance samples.

| | Tiny Aya (before) | Fine-tuned, unprotected | Fine-tuned with Brink |
|---|---|---|---|
| Attack success, English (HarmBench) ↓ | ⟦RESULT⟧ | ⟦RESULT⟧ | ⟦RESULT⟧ |
| Attack success, Hindi (Aya red-teaming) ↓ | ⟦RESULT⟧ | ⟦RESULT⟧ | ⟦RESULT⟧ |
| Over-refusal, Hindi ↓ | ⟦RESULT⟧ | ⟦RESULT⟧ | ⟦RESULT⟧ |
| Task quality, chrF on held-out Hindi ↑ | ⟦RESULT⟧ | ⟦RESULT⟧ | ⟦RESULT⟧ |
| Task loss ↓ | ⟦RESULT⟧ | ⟦RESULT⟧ | ⟦RESULT⟧ |
| AuditKIT attacks resisted ↑ | ⟦RESULT⟧ | ⟦RESULT⟧ | ⟦RESULT⟧ |

The gate caught ⟦RESULT⟧ of the 51 contaminating samples while dropping 5 % of the data.

## Does it hold up?

**The alarm fires before behaviour breaks (3 seeds).** Lead time ⟦RESULT⟧ ± ⟦RESULT⟧ steps; ⟦RESULT⟧ false alarms on
held-out safe runs; ⟦RESULT⟧× cheaper than running a judge at the same cadence.
<!-- figure: results/go_nogo/*.png -->

**The gate finds what a content filter can't.** AUROC ⟦RESULT⟧ for the gate vs ⟦RESULT⟧ for WildGuard on the same
samples. Among benign-looking samples, the ones the gate ranks highest erode safety faster than random ones
(attack success ⟦RESULT⟧ vs ⟦RESULT⟧).
<!-- figure: results/gate/*.png -->

**Repairing during training beats repairing after.**

| Method | Attack success ↓ | Task loss ↓ |
|---|---|---|
| No repair | ⟦RESULT⟧ | ⟦RESULT⟧ |
| Safety-data mixing | ⟦RESULT⟧ | ⟦RESULT⟧ |
| SafeTune ReSta (post-hoc) | ⟦RESULT⟧ | ⟦RESULT⟧ |
| SafeTune steering (inference-time) | ⟦RESULT⟧ | ⟦RESULT⟧ |
| **Brink rollback + guard** | ⟦RESULT⟧ | ⟦RESULT⟧ |
<!-- figure: results/repair/*.png -->

**Safety erodes beyond English.** English-only fine-tuning breaks ⟦RESULT⟧ first. A worst-language alarm fires
⟦RESULT⟧ steps before an English-only one.
<!-- figure: results/multilingual/*.png -->

## What it costs

Serving the protected Tiny Aya on one T4 costs **$⟦RESULT⟧ a month**, against $⟦RESULT⟧ for ⟦RESULT⟧ at the same traffic
(1M requests a month, measured serving speed). The fine-tune with Brink costs $⟦RESULT⟧ once. Brink's own overhead
is one forward pass every ⟦RESULT⟧ steps (⟦RESULT⟧ % of training time). Details: `results/ROI.md`.

## Try it

The [Space](⟦LINK⟧) runs the whole product live. Press **Run Brink**: it fingerprints the model, gates a
contaminated dataset with CuratorKIT, fine-tunes with and without Brink, red-teams all three models with AuditKIT,
and gives a verdict. **Gate your data** scores your own fine-tuning file and hands back a cleaned one.

In your own pipeline:

```python
from curatorkit.pipeline import Pipeline
from brink.gate import GateScorer
from brink.gate.curatorkit_gate import BrinkSafetyGate

gate = BrinkSafetyGate(GateScorer(model, tok, fingerprint, harmful_probe, harmless_probe), drop_frac=0.05)
Pipeline([reader, ExactDeduplicator(), gate, exporter], output_dir="curated").run()
```

## Responsible release

Brink deliberately trains models whose safety breaks, in order to study that.
- Model replies to harmful prompts are never published.
- The deliberately drifted adapters stay private; the release script refuses them.
- Everything public shows rates, scores and labels only.

## Limitations

- One model family (Tiny Aya, 3.35B); one seed per condition outside go / no-go.
- WildGuard is mostly English-trained, so Hindi attack success is less reliable. The per-language alarm itself
  is judge-free.
- CircuitKIT's Tiny Aya support is experimental upstream.
- ProCon-style anchoring, the closest published method to the guard, isn't among the baselines yet.

## Related work and what's new

Refusal-direction drift during fine-tuning and filtering safety-eroding data are both established (Du et al.
2025, ProCon; RefusalGuard 2026; He et al. 2024). Brink's contribution is a working, honestly evaluated system:
- one calibrated quantity that gates data, raises a validated live alarm and repairs training;
- measured against the standard fixes;
- shipped as drop-in parts of the Lexsi stack.

Feedback for the toolkits we built on: `docs/UPSTREAM.md`.
