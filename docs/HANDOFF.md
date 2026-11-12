# Handoff: where Brink stands and what's left

Read this first. It is the single source of truth for picking the project up. Last updated 2026-11-11.

## 1. The goal

Win the **Lexsi AI Safety & Alignment Hackathon** (Lexsi Labs with Cohere Labs and Hugging Face). Solo
entry by Cmg (GitHub `andrew1234-arch`).

| Date | What |
|---|---|
| Nov 5 | Registration closed (we're registered) |
| **Nov 14** | **Submission deadline** |
| Nov 16 | Up to 20 teams shortlisted |
| Nov 22 | Demo Day |

Requirements:
- Chain at least 2 Lexsi toolkits. Brink uses 4: CuratorKIT, AlignTune, SafeTune and AuditKIT.
- Give before/after metrics on a small open Cohere model for a real use case. Brink uses Tiny Aya,
  `CohereLabs/tiny-aya-global` (3.35B).
- Host the work on Hugging Face (a Space and Hub artifacts).

Tracks: Fine-tuning & Data Curation; AI Red Teaming, Safety & Alignment; Agentic Fine-tuning. Prizes are $30k+
in credits.

## 2. Rules that always apply

- **Never attribute an AI assistant**: no Co-Authored-By or session trailers, and no "Generated with …" in
  commits, PRs, code, comments or docs. See `.claude/CLAUDE.md`.
- **Commit and push as Cmg** `<157548721+andrew1234-arch@users.noreply.github.com>`. `.claude/settings.json`
  sets the GIT_AUTHOR_/GIT_COMMITTER_ variables. Check with `git log -1 --format='%an <%ae>'` before pushing.
- **Responsible release:**
  - never commit or publish model replies to harmful prompts (`generations.jsonl`);
  - never publish deliberately drifted adapters (`none`, `naive`, `pos`, `top`; `scripts/publish_hub.py`
    refuses them);
  - the app shows rates and labels only.
- The repo is private. Making it, the Space or the Hub repo public, and filing upstream issues
  (`docs/UPSTREAM.md`), are the owner's decisions.

## 3. What Brink is (one paragraph)

Brink catches a language model's safety collapsing during fine-tuning and stops it. It measures one quantity,
**J**: the model's preference to refuse rather than comply on fixed probe prompts, relative to a fingerprint
taken before fine-tuning. J is used in three places:
- a **gate** scores training data before training: cos(∇L(x), ∇J), with LoRA-at-init gradients;
- a **monitor** probes J every few steps, with an alarm threshold calibrated on safe (control) runs only;
- a **repair** step runs on alarm: roll back to the last safe LoRA state, then the **ErosionGuard** projects
  each gradient so it can't lower J.

Then **verify** on held-out harmful and harmless prompts, task loss and AuditKIT red-team probes. Details:
`README.md`, `docs/ARCHITECTURE.md`, `docs/RELATED_WORK.md`.

## 4. Repository map

```
brink/            library: probe/ gate/ monitor/ repair/ eval/ analysis/ dashboard/ live/
                  data.py (datasets, disjoint splits), runner.py (loaders, calibrated threshold, evaluator,
                  monitored runs), train_loop.py (LoRA SFT with controller hooks), io.py, utils.py
experiments/      entry points: preflight, go_nogo, gate, repair, multilingual, pipeline, live, run_all
configs/          one YAML per stage
notebooks/        kaggle_run_all.ipynb runs every stage on Kaggle and pushes results to the results/all branch
app/              Gradio app: Run Brink (live end to end), Gate your data, Experiments
scripts/          build_space.py, deploy_space.py (GitHub Actions → HF Space), publish_hub.py
tests/            offline CPU tests on tiny random models; test_architecture.py enforces the layering
docs/             one doc per experiment, plus ARCHITECTURE, DASHBOARD, SUBMISSION, UPSTREAM, RELATED_WORK
```

Layering rules (enforced by `tests/test_architecture.py`):
- `brink/` never imports `experiments/`, `scripts/` or `app/`;
- experiment scripts never import each other;
- no private (`_name`) imports across modules.

## 5. Local setup

Python 3.12 is recommended: SafeTune needs at least 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

```bash
make install-all     # library + dashboard + dev + CuratorKIT, AuditKIT, SafeTune, AlignTune (--no-deps) + trl 1.7.1
```

```bash
make test            # offline: HF_HUB_OFFLINE=1 pytest -q   (80 tests, ~2-4 min on CPU)
```

```bash
make lint            # ruff check + ruff format --check (CI runs exactly this)
```

```bash
make app             # the app on http://localhost:7860 (results/ as the data folder)
```

```bash
make demo            # the app with clearly bannered SIMULATED experiment runs
```

The CI test install is in `.github/workflows/ci.yml`: CPU torch, then
`pip install -e ".[dev,curatorkit,auditkit,safetune]" -r app/requirements.txt`, then `--no-deps aligntune`
and `trl==1.7.1`. Gradio is pinned to **6.31.0** in `app/requirements.txt` and in the `dashboard` extra; a
test keeps the two in sync.

Gotchas:
- `ruff format --check . | tail -1` hides failures. Check ruff's exit code.
- Don't kill processes by pattern (`pkill -f app/app.py`); use the PID.
- transformers 5 records `hidden_states` before user hooks run. The steering code measures with a forward
  pre-hook on layer l* for that reason.

## 6. Status (2026-11-06)

| Area | State |
|---|---|
| Library, 6 stages, app, CI, Space deploy pipeline, docs | **Done.** Tested offline; CI green on `main` |
| Real GPU runs | **Running** since 2026-11-12 09:15 UTC: Kaggle `cmgdevl/brink-run-all`, all stages, 11 h session budget, expected to finish around 20:00 UTC. go_nogo on Tiny Aya (as of 11:40 UTC, 3 of 4 runs): `pos_s0` / `pos_s1` ASR 16 % → 98–100 %, S3 at −11.6σ / −12.0σ by step 4; `control_s0` ASR flat (16 % → 12 %), S3 never below −1.0σ, so S3 separates cleanly. S1 / S2 also fall on control (to −9σ / −40σ), so they are not usable alarms. WildGuard judge working. The first attempt failed in every stage on Kaggle's torchao 0.10 (fixed: the notebook uninstalls it) |
| Kaggle GPU quota | **30 h / week** since Nov 12 (it was 6 h until then); resets Fridays 00:00 UTC |
| HF Space | **Not deployed.** `HF_SPACE_ID` isn't set, so the deploy job is skipped |
| Write-up numbers | 7 `⟦RESULT⟧` markers in `docs/SUBMISSION.md`; README says "Results: pending" |

**Critical path:** run the experiments → check and fix → fill in the numbers → deploy → submit by Nov 14.

## 7. What's left, step by step

### Step 1: run the experiments on Kaggle (6 GPU-hours left before the deadline: one session)

1. On Hugging Face, with the account whose token you'll use, accept the terms of:
   `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-base`, `allenai/wildguard`, `walledai/AdvBench`,
   `walledai/HarmBench`, `CohereForAI/aya_redteaming` and `CohereForAI/aya_dataset`.
2. The notebook is already on Kaggle: https://www.kaggle.com/code/cmgdevl/brink-run-all (private, GPU T4 x2,
   internet on). It clones `main` when it runs, so code fixes on `main` apply to the next run.
3. Kaggle → Settings → Account: phone verification must be done for GPU and internet.
4. Open the notebook → Edit → Add-ons → Secrets, then create and **attach** to this notebook:
   - `HF_TOKEN`: an HF read token;
   - `GITHUB_TOKEN`: fine-grained, only `andrew1234-arch/brink`, Contents read + write.
5. **Save Version → Save & Run All (Commit)**. Each stage runs preflight, runs, then pushes its results to the
   `results/all` branch. Re-run the notebook in a new session to continue; finished stages are skipped.
6. Order (most important first) and time: go_nogo (2 seeds) 2.7 h → pipeline 1.5 h (the required real use
   case) → live 0.4 h | gate 1.5 h → repair 2.7 h → multilingual 2 h (only if quota is left).
   `SESSION_BUDGET_H = 5.5` stops a session from starting a stage the quota can't finish; in-progress results
   are pushed every 20 min. One 6 h session covers go_nogo, pipeline and live.
   go_nogo matters most: it decides whether the core claim holds, and it calibrates the alarm the later stages
   use.

Running it from a cloud Claude Code session: the "brink" cloud environment allows `api.kaggle.com` and holds
the Kaggle API token. The Kaggle CLI's login check can't work through the proxy, so use kagglesdk directly
(ApiSaveKernelRequest with slug `cmgdevl/brink-run-all`, machine_shape `NvidiaTeslaT4`, enable_gpu and
enable_internet true; kernel_execution_type QUICK_SAVE saves without running). Check quota with
get_accelerator_quota_statistics. Locally, `kaggle kernels status cmgdevl/brink-run-all` works with a normal
token.

Watch the run:
- `git ls-remote origin refs/heads/results/all` should appear about 4 h in;
- then `git fetch origin results/all && git show FETCH_HEAD:results/go_nogo/REPORT.md`.

Without Kaggle, `make run-all` on any machine with 2 GPUs runs the same stages. With 1 GPU, the judge falls
back to the keyword judge.

### Step 2: when results land, check them honestly

For each stage, read `results/<stage>/REPORT.md` and `summary.json` on the `results/all` branch. Pass
criteria are in each experiment doc: `docs/GO_NOGO.md` ("Pass criteria (all must hold for GO)"),
`docs/GATE.md`, `docs/REPAIR.md`, `docs/MULTILINGUAL.md` and `docs/PIPELINE.md`.
- If go/no-go says NO-GO, don't hide it. Read the report's failing criterion and decide whether it's a bug, a
  sizing issue (steps, learning rate, contamination) or a real negative result. The write-up states
  limitations up front.
- Likely real-hardware failure points: Tiny Aya's chat template and refusal tokens; WildGuard in 4-bit on
  GPU 1 (bitsandbytes); memory on a 16 GB T4 (`max_len` 1024; lower `train.micro_batch` or `max_len` if it
  runs out); gated datasets (403 means the terms weren't accepted for that token's account); the Kaggle
  Python version vs SafeTune's ≥ 3.12 requirement (if `safetune` fails to install, the post-hoc ReSta /
  steering conditions fail; their report says so); AlignTune `--no-deps` (stage 04 falls back to Brink's
  training loop and says so).
- To debug a stage locally on a GPU: `python experiments/preflight.py --config configs/<stage>.yaml`, then
  `python experiments/<stage>.py --config configs/<stage>.yaml --out results/<stage>`. Every stage resumes:
  finished conditions are skipped.
- Fix bugs on `main` with tests. The notebook clones `main` on each run.

### Step 3: fill in the numbers

- `docs/SUBMISSION.md`: replace every `⟦RESULT⟧` (lines ~47-51 and the video script, ~line 126) with numbers
  from the `summary.json` files. Every number must trace back to a file in `results/`.
- `README.md`: replace "**Results:** pending the GPU runs…" (~line 116) with a short results table: before
  and after, unprotected vs Brink, with CIs where the analysis gives them.
- Each experiment doc (`docs/GO_NOGO.md` etc.): add the headline result and the plot (`results/<stage>/*.png`).
- To add result files to `main`, copy them from `results/all` (`git checkout FETCH_HEAD -- results/<stage>`).
  `results/.gitignore` keeps only csv / json / md / png.

### Step 4: deploy the Space

GitHub → repo Settings → Secrets and variables → Actions:

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | HF token with **write** access |
| Variable | `HF_SPACE_ID` | e.g. `<hf-username>/brink` (the Space is created if missing) |
| Secret (optional) | `HF_SPACE_TOKEN` | **read** token with access to AdvBench and HarmBench; installed as the Space's `HF_TOKEN` so live runs can load them |
| Variable (optional) | `HF_SPACE_PUBLIC` | `true` for public; anything else keeps it private |

Then Actions → "Deploy dashboard to Hugging Face Space" → Run workflow. It also runs after every green CI on
`main` and hourly, picking up new results from `results/all`.
- Only finished stages (with a `REPORT.md`) are published. With none, the Space shows clearly bannered
  simulated runs.
- Optional: set the Space hardware to **t4-small** (paid) so *Run Brink* runs Tiny Aya live. On the free CPU
  Space it uses `Qwen/Qwen2.5-0.5B-Instruct` (about 15-25 min per run).
- Check the Space by hand:
  - the *Run Brink* tab shows the recorded `live` run;
  - press Run and watch it finish;
  - *Gate your data* → "Load the example file" → Score;
  - the *Experiments* tab lists every stage.
- Details: `docs/DASHBOARD.md`, `scripts/deploy_space.py`.

### Step 5: publish artifacts to the Hub

```bash
python scripts/publish_hub.py --repo <hf-username>/brink-artifacts                     # dry run: lists files
```

```bash
python scripts/publish_hub.py --repo <hf-username>/brink-artifacts --yes --adapters rollback_guard
```

This publishes fingerprints, summaries, reports, plots and CSV logs, plus Brink-protected adapters by name.
It refuses drifted adapters and replies. Adapters exist only on the machine that trained them, so on Kaggle
run this at the end of the session, or the adapters are lost when it ends.

### Step 6: submit (by Nov 14)

- Submission text: `docs/SUBMISSION.md`, with the claims table, what was built and the limitations.
- Links: GitHub repo, HF Space, HF Hub artifacts repo.
- Toolkits used: CuratorKIT (BrinkSafetyGate), AlignTune (training in stage 04), SafeTune (ReSta and steering
  baselines), AuditKIT (red-team probes).
- Demo Day (Nov 22) script and 2-minute video script: `docs/SUBMISSION.md`. Start *Run Brink* on the GPU Space
  about 10 minutes before presenting, with the recorded run as the fallback.
- Repo About (set by hand on GitHub; the API was blocked):
  - description "Catch LLM safety collapse during fine-tuning, before it happens: gate, monitor, repair.
    Built on Tiny Aya and the Lexsi stack.";
  - topics `ai-safety`, `alignment`, `fine-tuning`, `llm`, `interpretability`, `lora`, `cohere`, `huggingface`.

## 8. Where things are configured

| What | Where |
|---|---|
| Stage sizes, models, datasets, thresholds | `configs/<stage>.yaml` |
| Alarm threshold `auto` | read from `results/go_nogo/summary.json` → `calibration.S3.threshold` (`brink/runner.py: resolve_threshold`) |
| Kaggle stage list, time budget | notebook cell 1 (`STAGES`, `HOURS`, `SESSION_BUDGET_H`) |
| Live-run defaults, allowed models on a Space | `brink/live/config.py` (`LiveConfig`, `allowed_models`, `BRINK_ALLOW_ANY_MODEL`) |
| Space contents and README | `scripts/build_space.py` |
| Deploy rules (CI-gated, tip of main only, finished stages only) | `scripts/deploy_space.py`, `.github/workflows/deploy-space.yml` |
