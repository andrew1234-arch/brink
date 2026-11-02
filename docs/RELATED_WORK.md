# Related work, and what Brink does and doesn't claim

Brink is built on ideas others have published. This page says exactly where it overlaps, and what it adds.
Summaries below are from the papers' abstracts and secondary summaries. **Read the full papers before the
final write-up and correct anything that's off.**

## Overlaps

| Area | Prior work | Overlap with Brink |
|---|---|---|
| Fine-tuning erodes safety | Qi et al. 2023, *Fine-tuning aligned language models compromises safety, even when users do not intend to* (arXiv 2310.03693) | The phenomenon itself, including benign data such as Alpaca. Our `pos` / `control` design follows it. |
| A single refusal direction | Arditi et al. 2024, *Refusal in language models is mediated by a single direction* (arXiv 2406.11717) | The diff-in-means refusal direction our fingerprint uses. |
| Refusal-direction drift during fine-tuning | Du et al. 2025, *Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint* (ProCon, arXiv 2509.06795) | Shows that the refusal direction drifts early in instruction tuning and that drift tracks safety loss. Mitigates with a projection-constraint regulariser. **This is the closest prior work to Brink's monitor and guard.** |
| Refusal geometry under fine-tuning | *RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs* (arXiv 2605.01913) | Measures drift and distortion of refusal representations, and preserves their geometry during fine-tuning. (Our guard was renamed `ErosionGuard` to avoid any confusion with it.) |
| Benign data that breaks safety | He et al. 2024, *What's in your "safe" data?* (arXiv 2404.01099) | Gradient and representation matching to find benign samples that degrade safety. Our `rep` baseline follows it. |
| Sample-level risk from parameter dynamics | *From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning* (arXiv 2605.04572) | Per-sample risk from drift along safety directions: close to the gate's goal. |
| Data purification | *Purifying Finetuning Data to Preserve LLM Safety Alignment* (EMNLP 2025) | Flags fine-tuning samples using a safety-sensitive layer's representations. |
| Behaviour vs representation | Jiang et al. 2026, *When Behavioral Safety Evaluation Fails: A Representation-Level Perspective* (arXiv 2606.08044) | Behavioural metrics miss representation-level fragility. Supports monitoring representations, not only outputs. |
| Gradient selection with LoRA | Xia et al. 2024, *LESS* (arXiv 2402.04333) | LoRA gradients as random projections for data selection. The gate uses the same idea. |
| Gradient surgery for safety | SafeGrad (in SafeTune); PCGrad (Yu et al. 2020) | Projecting task gradients against a safety gradient. |
| Post-hoc repair | ReSta (Bhardwaj et al. 2024); refusal-direction steering | Used through SafeTune as baselines. |

## What Brink does NOT claim

- That refusal-direction drift during fine-tuning is a new observation (Du et al. 2025; RefusalGuard 2026).
- That identifying harmful-making benign data is new (He et al. 2024 and the 2025–26 works above).
- That projecting gradients against a safety signal is new (SafeGrad, ProCon).

## What Brink adds

1. **One objective J, three jobs, one codebase.** The same calibrated refusal objective scores data before
   training (gate: `cos(∇L, ∇J)`), raises a live alarm during training (monitor), and constrains training
   after the alarm (guard). Both first-order relations are verified numerically in the tests.
2. **An operational alarm with honest evaluation.** The threshold is calibrated on safe-fine-tuning controls
   only, false alarms are measured leave-one-seed-out, lead time is measured at the confirming step, and the
   detector is compared with a cheap periodic judge on lead and cost.
3. **Act on the alarm, not only from step 0.** Rollback to the last safe state, then the guard, compared
   against the standard fixes on the same data: safety-data mixing, ReSta and steering.
4. **Multilingual monitoring.** Per-language fingerprints and a worst-language alarm, on Cohere's Aya
   red-teaming data.
5. **Drop-in for the Lexsi stack.** A CuratorKIT `BaseGate`, an AlignTune / HF Trainer callback, SafeTune and
   AuditKIT evaluation, demonstrated end to end on a Hindi-assistant use case.

## Natural extra baselines (not yet run)

- **ProCon-style anchoring:** a soft projection constraint on the refusal direction from step 0. It's the
  closest published method to the guard and the most informative comparison to add if time allows.
- **SafeGrad (SafeTune):** needs a second full model in GPU memory, so it doesn't fit beside the policy model
  on a T4.
