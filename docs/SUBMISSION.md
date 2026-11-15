# Brink: submission kit

> Every `⟦RESULT⟧` marker is a number that comes from the Kaggle runs. Nothing here is filled in before the
> experiments produce it.

## One line

**Brink catches a model's safety collapsing during fine-tuning, before it shows up in behaviour, and
prevents it: one refusal measure drives a data gate, a live monitor and a training-time repair.**

## The problem

Fine-tuning a safe model, even on mostly benign data, quietly erodes its refusals. Today you find out
afterwards, with an expensive behavioural eval, and repair it after the fact. By then compute is spent and the
damage is baked in.

## Positioning (read first)

Refusal-direction drift during fine-tuning, and filtering safety-eroding data, are both established in prior
work (Du et al. 2025 ProCon; RefusalGuard 2026; He et al. 2024; see [RELATED_WORK.md](RELATED_WORK.md)). Brink
doesn't claim those observations. It claims a **working, honestly evaluated system**: one calibrated refusal
objective that gates data, raises a validated live alarm, and repairs training, measured against the standard
fixes and shipped as drop-in parts of the Lexsi stack.

## The idea: one quantity, three jobs

Brink measures **J**, the model's refusal objective: the refusal-vs-compliance log-odds on a fixed probe set,
calibrated against the model before fine-tuning (its *fingerprint*).

| When | Brink component | How J is used | Lexsi toolkit |
|---|---|---|---|
| Before training | **Gate** | Scores each sample by `cos(∇L, ∇J)`: how much training on it would lower J | CuratorKIT (native `BaseGate`) |
| During training | **Monitor** | Probes J every few steps (one forward pass) and raises a calibrated alarm | AlignTune / HF Trainer callback |
| On alarm | **Repair** | Rollback to the last safe state, then the **erosion guard**: project each task gradient so it can't lower J | compared with SafeTune ReSta and steering |
| After training | **Verify** | Held-out ASR (WildGuard), over-refusal, task loss, red-team | AuditKIT red-team probes |

**Why it's sound:**
- The gate's score predicts the real change in J after an SGD step to three significant figures.
- The guard leaves J unchanged to first order (−1.7e-3 becomes under 1e-6) while removing only 4–35 % of the
  gradient.
- Both are checked in the test suite.

## Results ⟦pending Kaggle runs⟧

| Claim | Experiment | Result |
|---|---|---|
| The alarm fires before behaviour breaks, without false alarms on safe fine-tuning | 00 go / no-go, 2 seeds | lead ⟦RESULT⟧ ± ⟦RESULT⟧ steps (mechanistic signal ⟦RESULT⟧ vs proxy S3 ⟦RESULT⟧); leave-one-seed-out false alarms ⟦RESULT⟧; vs cheap judge: lead ⟦RESULT⟧, cost ⟦RESULT⟧× |
| The gate finds risk a content filter can't | 01 gate | AUROC gate ⟦RESULT⟧ vs WildGuard ⟦RESULT⟧; classifier flags ⟦RESULT⟧ of the gate's top benign subset; ASR top / random / bottom ⟦RESULT⟧ |
| Repairing in training beats the standard fixes | 02 repair | ASR none ⟦RESULT⟧ → rollback + guard ⟦RESULT⟧ at task loss ⟦RESULT⟧; safety-mix ⟦RESULT⟧, ReSta ⟦RESULT⟧, steering ⟦RESULT⟧ |
| Safety erodes across languages; English-only monitoring lags | 03 multilingual | first language to break ⟦RESULT⟧; English-only vs worst-language lead ⟦RESULT⟧ |
| The Lexsi pipeline ships a safe, useful model | 04 pipeline (Hindi assistant) | task loss naive ⟦RESULT⟧ vs Brink ⟦RESULT⟧; ASR naive ⟦RESULT⟧ vs Brink ⟦RESULT⟧; contamination caught by the gate ⟦RESULT⟧ |

Lead times are measured at the step the alarm is *confirmed*, so they're what a live monitor could act on.

## What we built (all open)

- `brink/`: probe and fingerprint, gate, monitor, repair, evals, analysis, dashboard, and `live/`: the whole
  product in one call.
- **A working app (Hugging Face Space).** *Run Brink* runs the full pipeline live on a real model: fingerprint,
  gate a contaminated fine-tuning set, fine-tune unprotected vs with Brink (alarm, rollback, erosion guard),
  verify both on held-out prompts, plain-language verdict. *Gate your data* scores any uploaded fine-tuning
  file and returns a cleaned one. *Experiments* replays every run with an adjustable alarm rule.
- 4 experiments plus a recorded end-to-end Tiny Aya run, with Kaggle notebooks, preflight checks, resumable
  runs, and an offline CPU test suite.
- Upstream feedback for AlignTune, CuratorKIT, SafeTune, AuditKIT and CircuitKIT ([UPSTREAM.md](UPSTREAM.md)).
- **Models and data:** Cohere Tiny Aya (`tiny-aya-global`, `tiny-aya-base`) and the Aya red-teaming and
  instruction datasets. Artifacts go on the Hugging Face Hub.

## Responsible release

- Published: fingerprints, results and logs, and Brink-protected adapters.
- Never published: model replies to harmful prompts, and the deliberately drifted adapters. The release
  script refuses them.
- The dashboard shows rates and labels only, never replies.

## Limitations (stated up front)

- One model family (Tiny Aya, 3.35B), one seed per condition in the first pass.
- WildGuard is mostly English-trained, so non-English ASR is less reliable. The per-language S3 alarm is
  judge-free.
- CircuitKIT (the circuit-level signal S4) isn't integrated yet. Aya support there is experimental.
- ProCon-style anchoring, the closest published method to the guard, isn't among the baselines yet.
- SafeGrad is excluded from the default comparison: its reference model doesn't fit next to the policy
  model on a T4.

---

## Demo Day: 5-minute live demo

**Before going on:** press *Run Brink end to end* on the Space (GPU hardware, Tiny Aya) about 10 minutes
ahead, so it finishes during the demo. If anything goes wrong, the tab shows the recorded Kaggle run.

0. **(45 s) It works, end to end.** *Run Brink* tab: the six stage cards completing; the unprotected run's
   refusal signal falling through the alarm line while the Brink run's holds (rollback marked); the verdict
   and the before / unprotected / with-Brink bars. *Gate your data*: load the example file, score it, the
   identity-shifting samples rank at the top, download the cleaned file.
1. **(30 s) The problem.** Fine-tuning breaks safety silently. Show the ASR curve of `pos` from experiment 00:
   flat, then a cliff.
2. **(60 s) Brink sees it coming.** Dashboard, run 00 `pos`:
   - the S3 signal drops;
   - the alarm line;
   - the shaded lead-time span before the behaviour breaks.

   Drag the z-threshold slider: the alarm and lead time move live. Switch to `control`: no alarm.
3. **(45 s) Where it breaks.** Layers tab: the erosion concentrates around the monitored layer.
4. **(60 s) It doesn't have to happen.**
   - Gate tab: planted harmful samples score at the top.
   - The benign samples the gate flags break safety faster than random ones.
5. **(60 s) Repair while training.** Run 02 trade-off plot: the erosion guard against no repair, ReSta and
   steering, on safety and task loss.
6. **(30 s) Beyond English.** Run 03: which language breaks first, and how much earlier the worst-language
   alarm fires.
7. **(15 s) Close.** One quantity, J: gate, monitor, repair. Open code, artifacts on the Hub, upstream feedback
   for five Lexsi toolkits.

Trim steps 3 and 6 if step 0 runs long. Steps 1-6 replay the logged experiment runs (*Experiments* tab).
Without a GPU Space, step 0 runs on the 0.5B CPU model (start it ~25 minutes ahead) or shows the recorded
Tiny Aya run; a Kaggle notebook with `--share` also serves the app on a GPU.

## 2-minute video script

| Time | Shot | Voice-over |
|---|---|---|
| 0:00–0:15 | Title, then the ASR cliff | "Fine-tune a safe model and its refusals can quietly collapse. You only find out afterwards." |
| 0:15–0:30 | *Run Brink* tab, sped up: stages completing, verdict | "Here is Brink doing its whole job on Tiny Aya: same data, same model, with and without Brink." |
| 0:30–0:45 | Signal dropping, alarm, lead span | "Brink watches the model's refusal mechanism itself, in one cheap forward pass, and fires ⟦RESULT⟧ steps before behaviour breaks." |
| 0:45–1:05 | Gate tab | "The same measure, applied to data before training, finds the samples that cause the damage." |
| 1:05–1:30 | Repair trade-off plot | "When the alarm fires, Brink rolls back and guards training so it can't erode refusal, keeping task performance. It beats repairing afterwards." |
| 1:30–1:45 | Multilingual plot | "And it watches every language, because English-only safety checks miss where models break first." |
| 1:45–2:00 | Stack logos, GitHub, Space | "Built on Tiny Aya and Lexsi's open stack. Code, results and the live monitor are open." |
