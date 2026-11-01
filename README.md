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

**Monitor dashboard:** live alarm, signals, ground truth and gate results for any run. See [docs/DASHBOARD.md](docs/DASHBOARD.md).

```bash
python -m pytest tests -q        # offline CPU tests of the whole pipeline (tiny random model)
```
