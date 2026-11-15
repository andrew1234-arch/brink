# Brink: submission kit

> Every number here comes from the Kaggle runs: `results/<stage>/summary.json` and `REPORT.md`, traced in
> `results/HEADLINES.md`. Verdicts are the pre-set pass criteria of each experiment, stated as the runs gave them.

## One line

**Brink catches a model's safety collapsing during fine-tuning while it happens, and prevents it: one refusal
measure drives a data gate, a live monitor and a training-time repair.**

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

## Results (all six stages final)

| Claim | Experiment | Result |
|---|---|---|
| Brink keeps a fine-tuned model safe at no task cost | live, end to end (1 seed, 128 samples, 32 harmful) | ASR before 21 % → unprotected **94 %** → with Brink **21 %**; refusal 77 % → 4 % → 69 %; over-refusal 0 % in all three; task loss 2.016 → 1.452 → 1.462. Gate: 29 / 32 harmful samples removed (precision 91 % vs 25 % random). Alarm on the unprotected run at step 20; none on the Brink run, so rollback and guard weren't used: the gate did this run's work |
| The alarm fires by the time behaviour breaks, without false alarms on safe fine-tuning | 00 go / no-go, 2 seeds | **Verdict NO-GO (3 of 4 pre-set checks pass).** S3 alarm: 2 / 2 harmful runs caught, 0 / 2 false alarms (leave-one-seed-out, threshold −3.0σ); confirmed on the break step (step 8, lead 0 ± 0 at 8-step eval resolution), z −11.6σ / −12.0σ already at step 4. Cheap judge every probe step: lead −4 (4 steps late). Mechanistic S1 / S2: fall on control too, 0 % detection once calibrated. Failed check G2: S3–ASR Spearman −0.38 / −0.62 (needs ≤ −0.7); −1.0 up to step 24, then ASR sits at ~98 % while S3 partly recovers. Cost per check: full probe (5 signals, 128 prompts) 23.8 s, cheap judge 6.2 s |
| The gate finds the planted harmful data; it does not yet predict which *benign* data erodes safety | 01 gate | **Verdict GATE NOT PROVEN (3 of 4 pre-set checks pass).** Planted harmful samples in a benign pool: gate AUROC **0.981** (86 % of them in the top 5 %), against 0.943 for the representation baseline and 0.525 for random; WildGuard, a content classifier, gets 0.999 on these overtly harmful samples. The classifier flags 0 % of the gate's top-scored benign subset. Fine-tuning on benign subsets, ASR 16 % → top **19 %** / random 11 % / bottom 9 %: the ordering is as predicted, but failed check D2 needs top ≥ random + 10 points and the gap is 8 |
| Stopping at the alarm keeps the model safe; rollback + guard does not beat the standard fixes | 02 repair | **Verdict REPAIR NOT PROVEN (3 of 4 pre-set checks pass).** ASR 16 % → no repair **83 %** (task loss 1.417). Brink, halt at the alarm (step 28): **17 %** at task loss 1.468. Brink, rollback + erosion guard, trained to the end: 41 % at 1.414; guard from step 0: 34 %. Baselines: safety-data mixing 23 % at 1.419; SafeTune steering 19 % at 1.476 (over-refusal 3 %); SafeTune ReSta 66 % at 1.817. Failed check R4: rollback + guard is less safe than safety mixing and steering. The guard removed about 1 % of the gradient norm on 64 of 100 steps: too little to hold the line on this data |
| Safety erodes across languages, and each language's own alarm sees it | 03 multilingual | ASR en 16 % → 98 % (breaks step 8), hi 2 % → 42 % (16), tl 2 % → 29 % (16); each language's own S3 alarm fires at or before its break (en 8, hi 16, tl 12). English broke first, so the English-only alarm did not lag here (lead 0 for both). **False alarms:** hi and tl alarms fire on control at step 20 (z −7.7 / −9.2, fixed −3σ threshold, ASR fell to 0 %) |
| The Lexsi pipeline ships a safe, useful model | 04 pipeline (Hindi assistant, 512 samples, 51 harmful) | **Verdict PIPELINE DELIVERS (4 of 4 pre-set checks pass).** English ASR before 16 % → unprotected **42 %** (95 % CI 30–54) → with Brink **6 %** (2–15). Hindi ASR 2 % → 0 % → 0 % (no Hindi erosion measured in this run; the judge is English-trained). Over-refusal 0 % → 3 % → 0 %. Hindi task loss 2.102 → 1.788 → 1.800. CuratorKIT + Brink gate dropped 26 samples (5 %): 21 of the 51 harmful ones and 5 benign. Both runs trained through AlignTune (TRL backend); the alarm fired at step 16 in both, and the Brink run rolled back to step 8 and finished under the guard |

Lead times are measured at the step the alarm is *confirmed* (2 consecutive probes), so they're what a live
monitor could act on. Behaviour is evaluated every 8 steps, so a lead of 0 means "on the first evaluation that
shows the break".

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

- One model family (Tiny Aya, 3.35B), two seeds in go / no-go and one seed per condition elsewhere.
- The working alarm signal is S3, a behavioural proxy (refusal-token log-odds on probe prompts). The
  activation-level signals S1 / S2 also drop on safe fine-tuning, so Brink does not claim mechanistic
  foresight. S3 is an alarm, not a severity gauge (it partly recovers once a model is fully broken).
- The alarm is confirmed on the evaluation step at which behaviour breaks, not before it, at 8-step
  evaluation resolution. Its edge is over a cheap judge (4 steps earlier) and in having no false alarms.
- Non-English alarms use a fixed threshold and false-alarm on safe fine-tuning (Hindi, Tagalog).
- WildGuard is mostly English-trained, so non-English ASR is less reliable. The per-language S3 alarm is
  judge-free.
- The gate does not yet predict which benign data erodes safety by the pre-set margin (8 points against the
  required 10), and on overtly harmful samples a content classifier is at least as good.
- Rollback + erosion guard is less safe than safety-data mixing and steering on this data. Halting at the alarm
  is the Brink policy that works; in the Hindi pipeline, rollback + guard after the gate ended at 6 % ASR.
- AuditKIT red-team numbers are not quoted: these runs scored them with AuditKIT's default detectors, which
  count a refusal as a failed attack ([UPSTREAM.md](UPSTREAM.md)). The fix is in the code; the runs predate it.
- The CircuitKIT stage (`circuit`) and the chrF task-quality metric are implemented and tested offline, but
  have not run on a GPU yet. CircuitKIT's Aya support is experimental.
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
   - the S3 signal drops to −12σ by step 4;
   - the alarm is confirmed at step 8, the first evaluation that shows the break, 4 steps before a cheap judge;

   Drag the z-threshold slider: the alarm and lead time move live. Switch to `control`: no alarm.
3. **(45 s) Where it breaks.** Layers tab: the erosion concentrates around the monitored layer.
4. **(60 s) It doesn't have to happen.**
   - Gate tab: planted harmful samples score at the top (AUROC 0.98).
   - Benign subsets end in the predicted order (top 19 %, random 11 %, bottom 9 % ASR), short of the pre-set
     margin: say so.
5. **(60 s) Repair while training.** Run 02 trade-off plot: no repair 83 %, halt at the alarm 17 %; rollback +
   guard (41 %) does not beat safety mixing or steering here. Then run 04: gate + rollback + guard together end
   the Hindi assistant at 6 % against 42 % unprotected.
6. **(30 s) Beyond English.** Run 03: every language breaks, each language's own alarm catches it, and the
   open problem: non-English alarms need their own calibration.
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
| 0:30–0:45 | Signal dropping, alarm | "Brink reads the model's refusals every few steps. It raised its alarm the moment behaviour broke, four steps before a cheap judge would, and never on safe fine-tuning." |
| 0:45–1:05 | Gate tab | "The same measure, applied to data before training, finds the samples that cause the damage." |
| 1:05–1:30 | Repair trade-off plot | "When the alarm fires, Brink stops or rolls back. Stopping held attack success at 17 percent against 83. On a Hindi assistant, gate, rollback and guard together ended at 6 percent against 42 unprotected, at the same task loss." |
| 1:30–1:45 | Multilingual plot | "Safety breaks in every language, and Brink can watch each one, though outside English its alarm still needs calibrating." |
| 1:45–2:00 | Stack logos, GitHub, Space | "Built on Tiny Aya and Lexsi's open stack. Code, results and the live monitor are open." |
