# Brink: abstract and submission form answers

Paste-ready for the submission form. Each `⟦name⟧` marker is filled from `results/*/summary.json` by
`python scripts/headlines.py --fill docs/ABSTRACT.md`, which lists any it couldn't fill. `results/HEADLINES.md` traces
every number to its file and key. Keep the abstract under 500 words (`wc -w` on the section below). If a verdict
is negative, rewrite the sentence instead of filling it.

## Track and problem statements

- **Track:** 2b, AI Safety and Correction.
- **Primary problem statement:** #5 Safety-preserving fine-tuning.
- **Also addresses:**
  - 2b #9 Locating where safety lives;
  - 2a #4 Multilingual and code-mixed safety.
- **Lexsi repos, chained in one pipeline:**
  - CuratorKIT: Brink's gate as a native `BaseGate` after `ExactDeduplicator`;
  - AuditKIT: red-team verification;
  - SafeTune: ReSta and refusal-direction steering, as the repair baselines Brink is measured against;
  - CircuitKIT: discovers the refusal circuit Brink monitors;
  - AlignTune (Track 1): the trainer in the Hindi use case.
- **Model:** `CohereLabs/tiny-aya-global` (3.35B, dense, open weights).
- **Links:**
  - GitHub: https://github.com/andrew1234-arch/brink
  - Space: ⟦LINK⟧
  - HF collection: ⟦LINK⟧
  - HF Article: ⟦LINK⟧

## Abstract

**Brink: catch safety collapse during fine-tuning, before it happens, and stop it.**

Fine-tuning a safe model on mostly benign data quietly erodes its refusals. Teams find out afterwards, with an
expensive behavioural evaluation, then repair the damage after the compute is spent. Brink makes safety a
property you keep *during* fine-tuning instead of one you check after it.

Brink measures one quantity, **J**: the model's preference to refuse rather than comply on fixed probe prompts,
read from the refusal direction at one layer and calibrated against the model before fine-tuning. One cheap
forward pass gives J. Brink uses it three times:

1. **Gate (CuratorKIT).** Before training, each sample is scored by how much one gradient step on it would
   lower J, cos(∇L, ∇J). The riskiest samples are rejected inside a CuratorKIT pipeline, with provenance.
2. **Monitor.** During training, J is probed every few steps. The alarm threshold is calibrated only on safe
   control runs, so safe fine-tuning doesn't trigger it.
3. **Repair.** On alarm, Brink rolls back to the last safe LoRA state and projects every later gradient so it
   cannot lower J (the erosion guard). It removes only the harmful component of each update.

Then **verify** on held-out harmful and harmless prompts, task quality and AuditKIT red-team probes.

**Use case.** A Hindi assistant fine-tuned on Tiny Aya Global from instruction data in which 10 % of the
samples are harmful-compliance pairs, chained as CuratorKIT → AlignTune → AuditKIT.
- Unprotected fine-tuning: English attack success ⟦pipeline.asr_before⟧ → ⟦pipeline.asr_naive⟧; Hindi
  ⟦pipeline.asr_hi_before⟧ → ⟦pipeline.asr_hi_naive⟧.
- With Brink: ⟦pipeline.asr_brink⟧ and ⟦pipeline.asr_hi_brink⟧.
- Task quality, chrF on held-out Hindi: ⟦pipeline.chrf_naive⟧ unprotected vs ⟦pipeline.chrf_brink⟧ with Brink.
- Over-refusal: ⟦pipeline.over_refusal_naive⟧ vs ⟦pipeline.over_refusal_brink⟧.

**Evidence.**
- The alarm fires ⟦go_nogo.lead_mean⟧ ± ⟦go_nogo.lead_sd⟧ steps before behaviour breaks, with a
  ⟦go_nogo.false_alarm_rate⟧ false-alarm rate on held-out safe runs (3 seeds).
- The gate finds eroding data a content classifier misses: AUROC ⟦gate.auroc⟧ vs WildGuard
  ⟦gate.classifier_auroc⟧.
- Repairing during training beats the standard fixes: attack success ⟦repair.asr_brink⟧ vs ⟦repair.asr_safety_mix⟧
  for safety-data mixing, ⟦repair.asr_resta⟧ for SafeTune ReSta and ⟦repair.asr_steer⟧ for steering, at task loss
  ⟦repair.loss_brink⟧.
- Erosion is not English-only: ⟦multilingual.first_break_lang⟧ breaks first. A worst-language alarm leads the
  break by ⟦multilingual.worst_lead⟧ steps, an English-only one by ⟦multilingual.english_lead⟧.
- CircuitKIT's discovered refusal circuit puts ⟦circuit.upstream⟧ of its score mass upstream of the layer Brink
  monitors.

**ROI.** Serving the protected Tiny Aya on one T4 costs $⟦roi.self_hosted_monthly⟧ a month against
$⟦roi.api_monthly⟧ for ⟦roi.api_model⟧ at the same traffic; Brink's one-off fine-tune pays back in
⟦roi.payback_months⟧ months.

**Responsible release.**
- Published:
  - fingerprints;
  - logs and reports (rates and labels only);
  - Brink-protected adapters.
- Never published:
  - model replies to harmful prompts;
  - deliberately drifted adapters.
- Limitations stated up front:
  - one model family;
  - an English-trained judge for non-English attack success;
  - CircuitKIT's Tiny Aya support is experimental.

Code, a live Space that runs the whole product end to end, and all artifacts are public.
