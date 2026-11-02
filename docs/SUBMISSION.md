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

## The idea: one quantity, three jobs

Brink measures **J**, the model's refusal objective: the refusal-vs-compliance log-odds on a fixed probe set,
calibrated against the model before fine-tuning (its *fingerprint*).

| When | Brink component | How J is used | Lexsi toolkit |
|---|---|---|---|
| Before training | **Gate** | Scores each sample by `cos(∇L, ∇J)`: how much training on it would lower J | CuratorKIT (native `BaseGate`) |
| During training | **Monitor** | Probes J every few steps (one forward pass) and raises a calibrated alarm | AlignTune / HF Trainer callback |
| On alarm | **Repair** | Rollback to the last safe state, then the **refusal guard**: project each task gradient so it can't lower J | compared with SafeTune ReSta and steering |
| After training | **Verify** | Held-out ASR (WildGuard), over-refusal, task loss, red-team | AuditKIT red-team probes |

**Why it's sound:**
- The gate's score predicts the real change in J after an SGD step to three significant figures.
- The guard leaves J unchanged to first order (−1.7e-3 becomes under 1e-6) while removing only 4–35 % of the
  gradient.
- Both are checked in the test suite.

## Results ⟦pending Kaggle runs⟧

| Claim | Experiment | Result |
|---|---|---|
| Drift is predictable: the monitor fires before behaviour breaks | 00 go / no-go | lead ⟦RESULT⟧ steps; probe ⟦RESULT⟧× cheaper than an eval; false alarms on safe fine-tuning ⟦RESULT⟧ |
| Drift is preventable: the gate finds the data that causes it | 01 gate | planted-sample AUROC ⟦RESULT⟧; ASR top / random / bottom ⟦RESULT⟧ |
| Repair in training beats repair after | 02 repair | ASR none → best policy ⟦RESULT⟧ at held-out loss ⟦RESULT⟧; vs ReSta ⟦RESULT⟧, steering ⟦RESULT⟧ |
| Safety erodes across languages, and English-only monitoring misses it | 03 multilingual | first language to break ⟦RESULT⟧; English-only vs worst-language lead ⟦RESULT⟧ |

Lead times are measured at the step the alarm is *confirmed*, so they're what a live monitor could act on.

## What we built (all open)

- `brink/`: probe and fingerprint, gate, monitor, repair, evals, analysis, dashboard.
- 4 experiments with Kaggle notebooks, preflight checks, resumable runs, and an offline CPU test suite.
- A Gradio monitor dashboard (Hugging Face Space) with a live mode and an adjustable alarm rule.
- Upstream feedback for AlignTune, CuratorKIT, SafeTune and AuditKIT ([UPSTREAM.md](UPSTREAM.md)).
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
- SafeGrad is excluded from the default comparison: its reference model doesn't fit next to the policy
  model on a T4.

---

## Demo Day: 5-minute live demo

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
5. **(60 s) Repair while training.** Run 02 trade-off plot: the refusal guard against no repair, ReSta and
   steering, on safety and task loss.
6. **(30 s) Beyond English.** Run 03: which language breaks first, and how much earlier the worst-language
   alarm fires.
7. **(15 s) Close.** One quantity, J: gate, monitor, repair. Open code, artifacts on the Hub, upstream feedback
   for four Lexsi toolkits.

**Live fallback:** if a live training demo is wanted, run the go/no-go notebook with `LIVE_DASHBOARD = True`
beforehand and keep the public link open. Otherwise, everything above replays the logged runs.

## 2-minute video script

| Time | Shot | Voice-over |
|---|---|---|
| 0:00–0:15 | Title, then the ASR cliff | "Fine-tune a safe model and its refusals can quietly collapse. You only find out afterwards." |
| 0:15–0:45 | Dashboard: signal dropping, alarm, lead span | "Brink watches the model's refusal mechanism itself, in one cheap forward pass, and fires ⟦RESULT⟧ steps before behaviour breaks." |
| 0:45–1:05 | Gate tab | "The same measure, applied to data before training, finds the samples that cause the damage." |
| 1:05–1:30 | Repair trade-off plot | "When the alarm fires, Brink rolls back and guards training so it can't erode refusal, keeping task performance. It beats repairing afterwards." |
| 1:30–1:45 | Multilingual plot | "And it watches every language, because English-only safety checks miss where models break first." |
| 1:45–2:00 | Stack logos, GitHub, Space | "Built on Tiny Aya and Lexsi's open stack. Code, results and the live monitor are open." |
