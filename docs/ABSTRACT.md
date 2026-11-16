# Brink: abstract and submission form answers

Paste-ready for the submission form. Every number comes from `results/*/summary.json`; `results/HEADLINES.md`
(written by `python scripts/headlines.py`) traces each one to its file and key. The abstract is under 500 words
(`wc -w` on the section below). Negative verdicts are stated as they are.

## Track and problem statements

- **Track:** 2b, AI Safety and Correction.
- **Primary problem statement:** #5 Safety-preserving fine-tuning.
- **Also addresses:**
  - 2b #9 Locating where safety lives;
  - 2a #4 Multilingual and code-mixed safety.
- **Lexsi repos, chained in one pipeline:**
  - CuratorKIT: Brink's gate as a native `BaseGate` after `ExactDeduplicator`;
  - AuditKIT: red-team probes in the verify step;
  - SafeTune: ReSta and refusal-direction steering, as the repair baselines Brink is measured against;
  - AlignTune (Track 1): the trainer in the Hindi use case;
  - CircuitKIT: a stage that compares its refusal circuit with the layer Brink monitors (implemented; its GPU
    run failed).
- **Model:** `CohereLabs/tiny-aya-global` (3.35B, dense, open weights).
- **Links:**
  - GitHub: https://github.com/andrew1234-arch/brink
  - Space: https://huggingface.co/spaces/cmgdevl/brink
  - Protected adapter: https://huggingface.co/cmgdevl/brink-artifacts
  - Logs and reports: https://huggingface.co/datasets/cmgdevl/brink-results
  - HF model collection: https://huggingface.co/collections/cmgdevl/brink-models-6afa95609e091281b4d637e4
  - HF dataset collection: https://huggingface.co/collections/cmgdevl/brink-datasets-6afa95617e17226835cb29ec
  - Write-up: https://github.com/andrew1234-arch/brink/blob/main/docs/ARTICLE.md

## Abstract

**Brink: catch a model's safety collapsing during fine-tuning, and stop it.**

Fine-tuning a safe model on mostly benign data can quietly erode its refusals, and teams find out only
afterwards. Brink keeps watch during fine-tuning instead.

Brink measures one quantity, **J**: the model's preference to refuse rather than comply on fixed probe prompts,
as a z-score against a fingerprint taken before fine-tuning. One forward pass gives J. Brink uses it three times:

1. **Gate (CuratorKIT).** Before training, each sample is scored by how much one gradient step on it would
   lower J, cos(∇L, ∇J). The riskiest samples are rejected inside a CuratorKIT pipeline.
2. **Monitor.** During training, J is probed every few steps. The alarm threshold is calibrated on safe control
   runs only.
3. **Repair.** On alarm, Brink halts, or rolls back to the last safe LoRA state and projects later gradients so
   they cannot lower J (the erosion guard).

**Use case (all 4 pre-set checks pass).** A Hindi assistant: Tiny Aya Global fine-tuned on 512 Hindi instruction
pairs, 51 of them harmful-compliance samples, chained as CuratorKIT → AlignTune → evaluation.
- Unprotected: English attack success 16 % → **42 %** (95 % CI 30–54).
- With Brink: **6 %** (2–15), over-refusal 0 %, Hindi task loss 1.800 against 1.788 unprotected (2.102 before),
  chrF 31.4 against 31.7. AuditKIT attacks resisted: 29 % unprotected, 57 % with Brink, as before fine-tuning.
- Hindi attack success stayed at 0–2 % in all three models: no Hindi erosion was measured.

**Evidence, including what failed.**
- End to end (one recorded run): attack success 21 % → 94 % unprotected → 21 % with Brink, at the same task loss.
- Alarm (2 seeds): caught both harmful fine-tunes, no false alarm on safe ones, 4 steps before a cheap judge.
  It is confirmed on the evaluation step where behaviour breaks, not before it. Verdict NO-GO on one of four
  checks: the signal is an alarm, not a severity gauge, and the activation-level signals detect nothing once
  calibrated.
- Gate: AUROC 0.981 on planted harmful samples (0.943 for a representation baseline). Not proven: on benign
  data the predicted ordering holds (19 % / 11 % / 9 % attack success for top / random / bottom subsets) but
  misses the pre-set 10-point margin.
- Repair: halting at the alarm ends at 17 % attack success against 83 % unrepaired. Not proven: rollback + guard
  alone (41 %) is less safe than safety-data mixing (23 %) and SafeTune steering (19 %).
- Languages: English, Hindi and Tagalog all break; each language's alarm catches its break, but the Hindi and
  Tagalog alarms also fire on safe fine-tuning.

**Responsible release.** Published: fingerprints, logs and reports (rates and labels only). Never published:
model replies to harmful prompts and deliberately drifted adapters.

**Limitations.** One model family; one or two seeds; an English-trained judge for non-English attack success;
the CircuitKIT stage failed on GPU and has no result.

Code and all results, including every failed criterion, are public.
