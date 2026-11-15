---
title: "Brink: catching a model's safety collapsing during fine-tuning, and stopping it"
authors:
  - user: cmgdevl
---

<!-- The Hugging Face Article (the hackathon write-up). Publish at https://huggingface.co/new-blog.
     Every number comes from results/*/summary.json; results/HEADLINES.md traces each to its file and key. -->

# Brink: catching a model's safety collapsing during fine-tuning, and stopping it

**One line:** one refusal measure gates the training data, raises an alarm while the model trains, and repairs
the run, so a fine-tuned Tiny Aya keeps its safety.

- **Track:** 2b, AI Safety and Correction.
- **Problem statements:** #5 Safety-preserving fine-tuning (primary); 2b #9 Locating where safety lives;
  2a #4 Multilingual and code-mixed safety.
- **Base model and method:** `CohereLabs/tiny-aya-global` (3.35B) with LoRA fine-tuning, protected by a
  gradient-based data gate, a calibrated refusal monitor, and rollback with gradient projection.
- **Headline before → after:** fine-tuning a Hindi assistant on data with 10 % harmful samples raised English
  attack success from 16 % to **42 %**; with Brink the same fine-tune ended at **6 %**, at the same task loss.

**Content warning.** This work fine-tunes models on harmful-compliance data on purpose, to study how safety
breaks. No model reply to a harmful prompt appears in this article or in any published artifact. One published
log lists the public HarmBench evaluation prompts with a refused / complied label each.

> **For judges: one minute.**
> - The required use case passes all four of its pre-set checks: 16 % → 42 % unprotected, 6 % with Brink, no
>   over-refusal, task loss 1.800 against 1.788.
> - Three Track 2 repos run in one pipeline (CuratorKIT, SafeTune, AuditKIT) plus AlignTune as the trainer.
> - Three of the five experiments have a negative verdict on one pre-set check each. They are reported in
>   full under "What didn't work", not relaxed after the fact.
> - Everything is reproducible from the repository on a free Kaggle T4 x2.
>
> [Results page](https://huggingface.co/spaces/cmgdevl/brink) · [Code](https://github.com/andrew1234-arch/brink)

## Abstract

Fine-tuning a safe model on mostly benign data can quietly erode its refusals. Teams find out afterwards, with an
expensive behavioural evaluation, once the compute is spent. Brink keeps watch during fine-tuning instead.

Brink measures one quantity, **J**: the model's preference to refuse rather than comply on fixed probe prompts,
as a z-score against a fingerprint taken before fine-tuning. One forward pass gives J. A **gate** scores each
training sample by how much a gradient step on it would lower J, and drops the riskiest inside a CuratorKIT
pipeline. A **monitor** probes J every few steps, with an alarm threshold calibrated on safe control runs only.
On alarm, a **repair** step halts, or rolls back to the last safe LoRA state and projects later gradients so
they cannot lower J.

On a Hindi assistant fine-tuned from contaminated data, unprotected fine-tuning took English attack success
from 16 % to 42 %; with Brink it ended at 6 %, with no over-refusal and the same task loss. The alarm caught
both harmful fine-tunes in the calibration experiment with no false alarm on safe ones. Three claims did not
meet their pre-set bar and are reported as such: the alarm signal is not a severity gauge, the gate's
prediction on benign data fell 2 points short of its margin, and rollback with the guard alone is less safe
than safety-data mixing.

## Problem statement and approach

### Why this problem

A team fine-tunes a safe model for its product. The data is mostly benign; a few bad samples slipped in, as
happens with scraped or crowd-sourced data. The model gets better at the task while its refusals fall apart.
The standard workflow finds out at the end: run a behavioural safety evaluation, see the damage, then retrain
or patch the weights. By then the compute is spent.

The use case here is a **Hindi assistant** built from crowd-sourced Hindi instructions (Cohere's Aya dataset)
in which 10 % of the samples are harmful-compliance pairs.

### Approach: one quantity, three jobs

Brink takes a **fingerprint** of the model before fine-tuning: the refusal direction at every layer, the layer
l* where harmful and harmless prompts separate best (layer 23 of 36 on Tiny Aya), and the model's own first
tokens for refusing and complying. **J** is the model's preference to refuse rather than comply on fixed probe
prompts, as a z-score against that fingerprint.

| When | Component | What it does | Lexsi toolkit |
|---|---|---|---|
| Before training | **Gate** | Scores each sample by cos(∇L, ∇J), with LoRA-at-init gradients, and drops the riskiest | CuratorKIT `BaseGate`, after `ExactDeduplicator` |
| During training | **Monitor** | Probes J every 4 steps; alarm when it stays below the calibrated threshold for 2 probes | AlignTune / HF Trainer callback |
| On alarm | **Repair** | Halt, or roll back to the last safe LoRA state and project each gradient so it can't lower J | compared with SafeTune ReSta and steering |
| After training | **Verify** | Held-out attack success, over-refusal, task loss | WildGuard judge; AuditKIT probes |

Two checks in the test suite confirm the first-order maths: the gate's score predicts the real change in J
after an SGD step to three significant figures, and the guard leaves J unchanged to first order.

### Hypotheses, set before the runs

Each experiment has pass criteria written in its doc (`docs/GO_NOGO.md`, `GATE.md`, `REPAIR.md`, `PIPELINE.md`)
before it ran, and its report prints a verdict against them.

| Experiment | Hypothesis | Verdict |
|---|---|---|
| 00 go / no-go | Drift is detectable by the time behaviour breaks, without false alarms, and the signal tracks attack success | NO-GO: 3 of 4 checks |
| 01 gate | The gate finds planted harmful data, and predicts which benign data erodes safety | NOT PROVEN: 3 of 4 |
| 02 repair | Brink's repair removes the drift and beats safety mixing and post-hoc repair | NOT PROVEN: 3 of 4 |
| 03 multilingual | Erosion reaches other languages, and per-language alarms see it | descriptive |
| 04 pipeline | The Lexsi pipeline with Brink ships a safe, useful Hindi assistant | DELIVERS: 4 of 4 |

## Related work

Fine-tuning erodes safety even with benign data (Qi et al. 2023, arXiv 2310.03693). Refusal is mediated by a
single direction (Arditi et al. 2024, arXiv 2406.11717), and that direction drifts during fine-tuning (Du et
al. 2025, ProCon, arXiv 2509.06795; RefusalGuard, arXiv 2605.01913). Benign samples that degrade safety can be
found by gradient and representation matching (He et al. 2024, arXiv 2404.01099). LoRA gradients work for data
selection (Xia et al. 2024, LESS, arXiv 2402.04333). Projecting task gradients against a safety gradient is
established (SafeGrad; PCGrad, Yu et al. 2020), as are post-hoc repairs such as ReSta (Bhardwaj et al. 2024).

Brink does not claim any of these as new. It adds one calibrated objective used for all three jobs in one
codebase, an alarm evaluated against safe controls and a cheap judge, and repair that acts on the alarm,
measured against the standard fixes on the same data. Details: `docs/RELATED_WORK.md`.

## Hackathon repos used

- **CuratorKIT.** Brink's gate is a native `BaseGate` (`BrinkSafetyGate`) in a CuratorKIT `Pipeline`: JSONL
  reader → `ExactDeduplicator` → `BrinkSafetyGate` → Alpaca exporter. The same pipeline runs in the app's
  *Gate your data* tab.
- **SafeTune.** ReSta and refusal-direction steering, applied to the drifted adapter, are the post-hoc
  baselines in the repair experiment.
- **AuditKIT.** Red-team probes (jailbreak, encoding, prompt injection, refusal) run in the verify step of the
  pipeline and repair experiments. Their scores from these runs are not quoted: see Limitations.
- **AlignTune** (Track 1). The trainer for both conditions of the Hindi use case (TRL backend), with Brink
  attached as a callback.
- **CircuitKIT.** A `circuit` stage compares CircuitKIT's EAP-IG refusal circuit with the layer Brink monitors.
  It is implemented and tested offline but has not run on a GPU, so it has no result here.

**How they connect.** CuratorKIT curates the data (dedup, then Brink's gate) → AlignTune fine-tunes with
Brink's monitor, alarm and repair attached → WildGuard and AuditKIT evaluate the result → SafeTune's repairs
are the baselines Brink is compared with. Feedback for each toolkit: `docs/UPSTREAM.md`.

## Models and licences

- **Base model:** `CohereLabs/tiny-aya-global`, 3.35B, LoRA r 16, α 32, dropout 0.05. Licence: as on its model card (CC-BY-NC 4.0).
- **Judge:** `allenai/wildguard` in 4-bit on the second GPU.
- **No teacher model and no closed-model baseline.**

## Datasets, benchmarks and licences

| Role | Source | Size used | Language |
|---|---|---|---|
| Task data (Hindi assistant) | `CohereForAI/aya_dataset` | 512 train (461 benign + 51 harmful), 64 held-out for task loss, 32 for over-refusal | Hindi |
| Harmful-compliance contamination | `LLM-LAT/harmful-dataset` | 51 of the 512 | English |
| Benign instructions (other experiments) | `tatsu-lab/alpaca` | 512 per condition | English |
| Fingerprint and probe prompts | `walledai/AdvBench` and Alpaca | 64 + 64 calibration, 64 + 64 probe | English |
| Safety benchmark (focus) | `walledai/HarmBench`, standard | 64 held-out harmful prompts | English |
| Safety benchmark, Hindi and Tagalog | `CohereForAI/aya_redteaming` | 48 per language | Hindi, Tagalog |

Licences are those on each dataset card.

**Splits and leak check.** All splits are disjoint. Every fine-tuning prompt is de-duplicated against the probe
and evaluation prompts by exact match and word-level Jaccard overlap (`brink/data.py`), so neither the
monitor's signal nor the behavioural ground truth can leak into training.

**CuratorKIT filtering.** In the Hindi use case the pipeline took 512 samples in and passed 485. The gate
rejected 26 (the top 5 % by predicted refusal erosion): 21 of the 51 harmful samples and 5 benign ones. The drop rate is 5 %, not the true 10 %, because a practitioner does not
know the contamination rate.

**Over-refusal test set.** 32 unseen benign Hindi instructions from the Aya dataset (Alpaca in the English
experiments). They are ordinary requests, not prompts chosen to look risky.

## Metrics, and why these

| Metric | Direction | Why |
|---|---|---|
| Attack success rate (ASR), WildGuard-judged, on held-out harmful prompts | lower | The behavioural ground truth: does the model answer harmful requests |
| Refusal signal S3 (z-score of J) | higher | The cheap signal the monitor watches; judge-free |
| Over-refusal on harmless prompts | lower | Safety must not come from refusing everything |
| Held-out task loss | lower | The fine-tune must still learn the task |
| Gate AUROC and recall at the top 5 % | higher | Does the gate rank the harmful samples first |
| Alarm lead, detection and false-alarm rate | — | Is the alarm usable: early, complete, quiet on safe runs |

**What should not get worse:** over-refusal and task loss, reported next to every safety number.

**Checking the judge.** WildGuard's agreement with human labels was not measured here. A keyword judge runs
beside it as a sanity check (English ASR in the use case: 22 % / 47 % / 6 % by keyword against 16 % / 42 % / 6 %
by WildGuard). WildGuard is mostly English-trained, so Hindi and Tagalog ASR are less reliable.

## Method and experiments

**Training setup (all experiments).** LoRA SFT, 128 steps, learning rate 1e-4, 4 warm-up steps, effective batch
8, maximum length 1024, gradient checkpointing. Probe every 4 steps; behavioural evaluation at steps 0, 8, 16,
24, 32, 48, 64, 96 and 128. Every config is in `configs/<stage>.yaml`.

- **CuratorKIT:** `ExactDeduplicator` defaults; `BrinkSafetyGate` with `drop_frac` 0.05, gate LoRA r 8, target S3.
- **AlignTune:** TRL backend SFT with the settings above; one process per condition.
- **SafeTune:** ReSta α 0.25; steering scaled to restore the measured drop along the refusal direction.
- **Brink alarm:** threshold calibrated in experiment 00 on control runs only (−3.0σ), persistence 2 probes.
- **Brink guard:** target S3, safety gradient refreshed every 8 steps.

**Seeds.** Two seeds per condition in go / no-go; one seed elsewhere.

**Ablations.** The repair experiment separates the parts: halt at the alarm, rollback + guard, guard from step
0, against safety mixing and two post-hoc repairs. The gate experiment compares the gradient score with a
representation score, random ranking and a content classifier.

## Results

### The Hindi assistant: same model before and after

Tiny Aya Global, 512 Hindi instruction pairs with 51 harmful samples. Pipeline: CuratorKIT (dedup → Brink gate)
→ AlignTune with Brink attached → evaluation.

| | Tiny Aya (before) | Fine-tuned, unprotected | Fine-tuned with Brink |
|---|---|---|---|
| Attack success, English (HarmBench) ↓ | 16 % | **42 %** (95 % CI 30–54) | **6 %** (2–15) |
| Attack success, Hindi (Aya red-teaming) ↓ | 2 % | 0 % | 0 % |
| Over-refusal, Hindi ↓ | 0 % | 3 % | 0 % |
| Hindi task loss ↓ | 2.102 | 1.788 | 1.800 |

The alarm fired at step 16 in both runs. The Brink run rolled back to step 8 and finished under the guard. All
four pre-set checks pass. Hindi attack success stayed at 0–2 % in every model, so this run shows no Hindi
erosion for Brink to prevent; the measured benefit is on English prompts.

### End to end, one recorded run

128 samples, 32 harmful, one seed, sized to finish in minutes for the demo.

| | Before | Unprotected | With Brink |
|---|---|---|---|
| Attack success ↓ | 21 % | 94 % | 21 % |
| Refuses harmful prompts ↑ | 77 % | 4 % | 69 % |
| Over-refusal ↓ | 0 % | 0 % | 0 % |
| Task loss ↓ | 2.016 | 1.452 | 1.462 |

The gate removed 29 of the 32 harmful samples. The alarm did not fire on the Brink run, so the gate did this
run's work.

### The alarm

Two seeds. Harmful fine-tuning took ASR from 16 % to 97–100 %, breaking at step 8; safe fine-tuning stayed flat
(16 % → 11–12 %). The S3 alarm caught both harmful runs with no false alarm on the safe ones (leave-one-seed-out).
It is confirmed at step 8, the first evaluation that shows the break, with the signal already at −12σ at step 4.
A cheap judge checking at every probe step fires 4 steps later.

### What didn't work

- **The signal is an alarm, not a gauge (go / no-go, check G2).** The rank correlation between S3 and ASR is
  −0.38 and −0.62 over the whole run against a required −0.7. It is −1.0 while ASR is rising, then ASR sits near
  98 % while S3 partly recovers.
- **No mechanistic foresight.** The activation-level signals S1 and S2 also fall on safe fine-tuning and detect
  nothing once calibrated. The working signal is behavioural.
- **The alarm is not early.** It fires on the evaluation step where behaviour breaks, at 8-step resolution.
- **The gate does not beat a content classifier on overtly harmful data.** AUROC 0.981 for the gate (0.943 for
  the representation baseline, 0.525 random) against 0.999 for WildGuard.
- **The gate's prediction on benign data is short of its margin (check D2).** Fine-tuning on the benign
  subsets it ranks top, random and bottom ended at 19 %, 11 % and 9 % ASR: the predicted order, but 8 points
  between top and random where the check asked for 10. The classifier flags 0 % of that top subset.
- **Rollback + guard does not beat the standard fixes (check R4).**

| Method | Attack success ↓ | Task loss ↓ |
|---|---|---|
| No repair | 83 % | 1.417 |
| Brink: halt at the alarm (step 28) | **17 %** | 1.468 |
| Brink: rollback + guard | 41 % | 1.414 |
| Brink: guard from step 0 | 34 % | 1.417 |
| Safety-data mixing | 23 % | 1.419 |
| SafeTune steering (inference-time) | 19 % | 1.476 |
| SafeTune ReSta (post-hoc) | 66 % | 1.817 |

  The guard removed about 1 % of the gradient norm on 64 of 100 steps: too little to hold the line on
  this data. Halting is the Brink policy that works on its own.
- **Non-English alarms false-alarm.** English, Hindi and Tagalog all break under harmful fine-tuning (ASR
  16 → 98 %, 2 → 42 %, 2 → 29 %), and each language's alarm fires at or before its break. But the Hindi and
  Tagalog alarms also fire on safe fine-tuning at step 20.

### Cost and compute

All runs used Kaggle's free GPU T4 x2: about 13 GPU-session hours for the five experiments plus the recorded
run (go / no-go 3.2 h, gate 2.4 h, repair 3.5 h, multilingual 2.3 h, pipeline 1.6 h). Cost: $0. One full probe
(5 signals, 128 prompts) takes 23.8 s; a cheap-judge check takes 6.2 s. A closed-model cost comparison
(`scripts/roi.py`) is implemented but needs a serving-speed measurement that these runs don't include.

## Limitations

- One model family (Tiny Aya, 3.35B); two seeds in go / no-go and one elsewhere; small evaluation sets (48–64
  prompts), hence the wide confidence intervals.
- The three negative results above.
- AuditKIT scores are not quoted. These runs scored the probes with AuditKIT's default detectors, which count
  a refusal as a failed attack. The scoring is fixed in the code (`brink/eval/audit.py`); the runs predate it.
- The chrF task-quality metric and the CircuitKIT stage are implemented and tested offline, but not in these
  runs. Task quality is reported as held-out loss only.
- WildGuard is mostly English-trained, and its agreement with human labels was not checked.
- ProCon-style anchoring, the closest published method to the guard, isn't among the baselines.
- No results on a quantized model.

## With more credits

- **Planned:** more seeds and larger evaluation sets to tighten the intervals; a per-language calibrated
  threshold; a stronger guard (several probe directions, a larger projection); the CircuitKIT comparison and
  chrF on GPU; AuditKIT with the corrected scoring; ProCon as a baseline; a second model (Aya Expanse 8B).
- **Expected:** the Hindi and Tagalog false alarms should go away with per-language calibration, as the English
  ones did. Whether a stronger guard closes the gap to safety mixing (41 % against 23 %) is open; the honest
  forecast is that gate + halt stays the reliable configuration.

## Artifacts

- **Code:** https://github.com/andrew1234-arch/brink
- **Results page (Space):** https://huggingface.co/spaces/cmgdevl/brink
- **Logs, reports and fingerprints:** `results/` in the repository, and the Hugging Face dataset collection
  linked from the submission form.

## Documentation and reproduction

- **Environment:** Python 3.12; `make install-all`. GPU runs need two 16 GB GPUs (Kaggle T4 x2 is enough).
- **Layout:** `brink/` the library (probe, gate, monitor, repair, eval, live); `experiments/` one entry point
  per experiment; `configs/` one YAML each; `notebooks/kaggle_run_all.ipynb` runs everything on Kaggle;
  `app/` the Gradio app; `tests/` the offline CPU suite.
- **Run:** `make test` (offline, CPU); `make run-all` or the Kaggle notebook for every experiment; `make app`
  for the live app. Each experiment resumes and writes a `REPORT.md` with its verdict.
- **Work done outside the hackathon:** none. The repository starts on 2026-11-01.

## Responsible release

Brink deliberately trains models whose safety breaks, in order to study that. Model replies to harmful prompts
are never published. The deliberately drifted adapters stay private, and the release script refuses them.
Everything public shows rates, scores and labels only.

## What we learned

A cheap behavioural probe is enough to raise a reliable alarm, and stopping at that alarm is a strong baseline
that costs nothing. Filtering the data first matters more than repairing the gradients afterwards: in both runs
where Brink ended safe, the gate had removed most of the harmful samples. The mechanistic signals we hoped
would give early warning did not survive calibration against safe fine-tuning.

## Author

Solo entry by Cmg ([@andrew1234-arch](https://github.com/andrew1234-arch)): all design, code, experiments and
writing.

## References

Qi et al. 2023 (arXiv 2310.03693) · Arditi et al. 2024 (arXiv 2406.11717) · Du et al. 2025 (arXiv 2509.06795) ·
He et al. 2024 (arXiv 2404.01099) · Xia et al. 2024 (arXiv 2402.04333) · Yu et al. 2020 (PCGrad) · Bhardwaj et
al. 2024 (ReSta) · Lexsi Labs: CuratorKIT, AlignTune, SafeTune, AuditKIT, CircuitKIT.
