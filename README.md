# Brink

**Detect and prevent LLM safety collapse during fine-tuning, before it happens.**

Fine-tuning a model on harmless-looking data can quietly erode its safety. Today you only find out
afterwards, with an expensive behavioural eval. Brink watches the model's internal refusal mechanism
*while it trains*, sounds the alarm before the behaviour breaks, and filters out the training data that
causes the damage in the first place.

- **Gate:** scores training data for safety-drift risk before training (a CuratorKIT gate).
- **Monitor:** a training callback that probes the refusal direction and circuit every N steps, with a
  calibrated alarm.
- **Repair:** hands off to SafeTune (ReSta, SafeGrad, steering) only when needed.
- **Verify:** ground-truth safety, over-refusal and capability evals with AuditKIT and SafeTune.

Built on Cohere's Aya models with the Lexsi Labs open-source stack (AlignTune, CuratorKIT, SafeTune,
AuditKIT, CircuitKIT) for the Lexsi AI Safety & Alignment Hackathon 2026.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

## Status

| Experiment | What it tests | Status |
|---|---|---|
| [00 go / no-go](docs/GO_NOGO.md) | Fine-tuning breaks Tiny Aya's safety, and the probe signals track it | ready: `notebooks/kaggle_go_nogo.ipynb` |
| [01 data gate](docs/GATE.md) | The gate finds the (benign-looking) data that erodes safety, before training | ready: `notebooks/kaggle_gate.ipynb` |
| [02 repair](docs/REPAIR.md) | What to do when the alarm fires: halt, rollback, erosion guard, vs SafeTune post-hoc repair | ready: `notebooks/kaggle_repair.ipynb` |
| [03 multilingual](docs/MULTILINGUAL.md) | English fine-tuning erodes safety in Hindi and Tagalog too; English-only monitoring vs worst-language | ready: `notebooks/kaggle_multilingual.ipynb` |
| [04 pipeline](docs/PIPELINE.md) | The full Lexsi stack (CuratorKIT + gate → AlignTune + Brink) ships a safe Hindi assistant from contaminated data | ready |

**Monitor dashboard:** live alarm, signals, ground truth and gate results for any run. See [docs/DASHBOARD.md](docs/DASHBOARD.md).

**Using Brink in your own training:**
- Hugging Face Trainer, TRL and AlignTune: `brink.monitor.callback` (`BrinkCallback`, `attach_to_aligntune`).
- CuratorKIT pipelines: `brink.gate.curatorkit_gate.BrinkSafetyGate`.

**Release:**
- `scripts/build_space.py` packages the dashboard as a Hugging Face Space.
- `scripts/publish_hub.py` publishes fingerprints and results. It never publishes harmful text or drifted adapters.

**Run everything:** `notebooks/kaggle_run_all.ipynb` runs all five stages in order, saves results after each
stage, and resumes across Kaggle sessions.

**For the submission:**
- [docs/SUBMISSION.md](docs/SUBMISSION.md): write-up, Demo Day and video scripts.
- [docs/RELATED_WORK.md](docs/RELATED_WORK.md): prior work, and what Brink does and doesn't claim.
- [docs/UPSTREAM.md](docs/UPSTREAM.md): feedback for the Lexsi toolkits.

```bash
python -m pytest tests -q        # offline CPU tests of the whole pipeline (tiny random model)
```
