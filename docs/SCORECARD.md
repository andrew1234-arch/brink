# Scorecard: Brink against the Lexsi hackathon rules

Re-score this at every milestone and at least once on **Nov 13**, the day before the deadline (Nov 14, end of day
AoE). Every row must reach ✅ with evidence a judge can open. "Partly" is a gap.

Source: <https://hackathon.lexsi.ai/> (read 2026-11-06).

## 1. What the organisers have published about judging

There is **no published rubric**. The only scoring text is the FAQ line *"Judges will score each team numerically
on a fixed scale."*
- Judges from industry and academia score every submission and shortlist up to 20 teams for Demo Day.
- Shortlisted teams have Nov 16–21 to "prove their thesis".
- A panel picks the winners at Demo Day on Nov 22.

**Action:** ask on the hackathon Discord for the scale and the criteria. Until they answer, the submission
requirements and the problem statements below are the rubric.

## 2. Prize maths

- There are 5 ranked prizes across **all** tracks, with no per-track prizes:
  - 1st: $3,662;
  - 2nd: $2,662;
  - 3rd: $1,962;
  - 4th: $1,362;
  - 5th: $1,162.
- The prizes are made up of Lexsi and Cohere credits plus HF Pro.
- Up to 20 finalists get $200 in HF credits each.
- Up to 800 Discord teams get $20 each, so the field could be several hundred teams.
- **A prize means placing in the top 5 overall.** Being in the top 20 earns only the $200 finalist credit.

## 3. Submission requirements (pass / fail)

| # | Requirement (site's words) | What Brink does | Evidence for a judge | Status |
|---|---|---|---|---|
| R1 | "chain together repos **from their track** into a real pipeline"; minimum 2, "3 or more" preferred | Track 2 repos: CuratorKIT (gate as a native `BaseGate`), AuditKIT (red-team verify), SafeTune (ReSta / steering repair baselines), CircuitKIT (refusal circuit vs Brink's monitored layer). AlignTune (a Track 1 repo) trains in stage 04. *Run Brink* gates through CuratorKIT and verifies with AuditKIT | `brink/live/pipeline.py`, `experiments/pipeline.py`, `experiments/circuit.py` | ⚠️ CircuitKIT stage untested on a GPU |
| R2 | "a dense, open-weights LLM from Cohere Labs" | `CohereLabs/tiny-aya-global` (3.35B, on the allowed list) | `configs/*.yaml` | ✅ |
| R3 | Before / after, "more than one metric" | ASR (WildGuard), refusal rate, over-refusal, held-out task loss, task chrF, AuditKIT attack resistance, alarm lead time | `results/*/summary.json` | ❌ no GPU results yet |
| R4 | "Host models, adapters, logs and datasets on Hugging Face" | `scripts/publish_hub.py`: model repo (fingerprints, protected adapters) and dataset repo (logs, results) | HF links | ❌ not published |
| R5 | "Add all models, adapters, etc. into a HF collection … public (un-gated)" | `publish_hub.py --collection "Brink" --space …` | collection link | ❌ not published |
| R6 | "Projects must be deployed publicly (e.g. a public demo or repository link)" | GitHub repo and Space are **private** today | public URLs | ❌ owner's decision, required |
| R7 | Write-up: a HF Article (or a GitHub README) | `docs/ARTICLE.md`, ready to paste into an HF Article | article link | ⚠️ draft, numbers pending |
| R8 | Abstract under 500 words | `docs/ABSTRACT.md` | submission form | ⚠️ draft, numbers pending |
| R9 | State the track, repos, models and problem statements used | `docs/ABSTRACT.md` → "Track and problem statements" | submission form | ⚠️ draft |
| R10 | ROI comparison: closed model vs the fine-tuned small model (optional, "preferred") | `docs/ROI.md`, from `brink/analysis/roi.py` | ROI table | ⚠️ draft, numbers pending |
| R11 | Only the last submission before the deadline counts | Submit once, early, then resubmit | — | — |

## 4. Problem-statement fit (each one is a checklist)

Brink enters **Track 2b, AI Safety and Correction**. Primary: **#5**. Secondary: **#9** and **2a #4**.

| Problem statement (site's words, paraphrased where long) | Brink's answer | Gap |
|---|---|---|
| **2b #5 Safety-preserving fine-tuning**: fine-tune for a *useful domain task* with a safety-hardening method; show safety holds better than plain fine-tuning | Stage 04: a Hindi assistant fine-tuned naive vs with Brink (gate, monitor, rollback and guard), through CuratorKIT → AlignTune → AuditKIT | Needs a task-*quality* metric next to task loss (added: chrF of replies against the held-out Hindi references); needs real numbers |
| **2b #9 Locating where safety lives**: find the circuits, neurons or directions behind refusal and use them for a targeted fix | Brink's J reads the refusal direction at layer l*; the guard is a targeted fix along it. Stage `circuit` runs CircuitKIT discovery on the refusal task and checks that l* is a circuit layer | CircuitKIT stage needs a GPU run |
| **2b #6 Repairing safety drift without retraining** (weight-space) | Not Brink's method (Brink repairs during training). SafeTune ReSta is the baseline for this | Don't claim it; position #6 methods as what Brink beats |
| **2b #12 Tamper resistance** | Not claimed: Brink protects *your own* fine-tune, not against an attacker's | — |
| **2a #4 Multilingual and code-mixed safety**: test across languages "and close the gaps found" | Stage 03 measures erosion per language; stage 04 closes it for Hindi | State the link explicitly in the article |

## 5. Lessons from past entries

| Lesson | Brink status |
|---|---|
| The sponsor's tools are the core, not attachments | Live product now runs CuratorKIT and AuditKIT; CircuitKIT validates the layer Brink watches. Brink's own math stays the engine: be explicit about which part is which |
| A real, named user | ❌ The Hindi assistant is a constructed use case. Name one: a fine-tuning-as-a-service provider or an Indic-language deployer, ideally someone who will say so |
| Measured upstream feedback | `docs/UPSTREAM.md`: 11 entries over five kits, re-checked on the latest versions (9 open, 2 fixed upstream); the AuditKIT detector finding is measured with a repro. Add every real-hardware issue the same way |
| No win predictions without a full scorecard and scored rivals | No rivals scored yet |

## 6. Re-score log

| Date | Rows ✅ / total (section 3) | Notes |
|---|---|---|
| 2026-11-06 | 1 / 10 | Kaggle runs starting; public release, article and abstract not done |
| 2026-11-06 (later) | 1 / 10 ✅, 5 ⚠️ drafted | Branch `rubric-gaps`: R1, R7–R10 drafted and tested (96 tests). Found and fixed AuditKIT scoring inverted in Brink's own red-team metric |
