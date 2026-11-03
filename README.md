# Brink

[![CI](https://github.com/andrew1234-arch/brink/actions/workflows/ci.yml/badge.svg)](https://github.com/andrew1234-arch/brink/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Catch a language model's safety collapsing *during* fine-tuning, before it shows up in behaviour, and stop
it.**

Fine-tuning a safe model, even on mostly benign data, can quietly erode its refusals. Today you find out
afterwards, with an expensive behavioural evaluation, and repair the damage after the fact. Brink watches the
model's refusal mechanism while it trains, raises a calibrated alarm, acts on it, and filters out the training
data that causes the damage in the first place.

Built on Cohere's **Tiny Aya** with the **Lexsi Labs** open-source stack (CuratorKIT, AlignTune, SafeTune,
AuditKIT) for the Lexsi AI Safety & Alignment Hackathon 2026.

---

## How it works

Brink measures one quantity, **J**: the model's preference to refuse rather than comply on a fixed set of
probe prompts, relative to a *fingerprint* of the model taken before fine-tuning. J does three jobs:

```mermaid
flowchart LR
    FP[Fingerprint<br/>refusal direction, layer l*,<br/>refusal tokens, baselines] --> G & M & R
    D[(Training data)] --> G[Gate<br/>score each sample by cos ∇L, ∇J<br/>CuratorKIT BaseGate]
    G -->|riskiest samples dropped| T[Fine-tuning<br/>HF Trainer / TRL / AlignTune]
    T <--> M[Monitor<br/>probe J every N steps<br/>control-calibrated alarm]
    M -->|alarm| R[Repair<br/>rollback to last safe state<br/>+ erosion guard]
    R --> T
    T --> V[Verify<br/>ASR WildGuard · over-refusal ·<br/>task loss · AuditKIT red-team]
```

| Stage | What Brink does | Why it holds up |
|---|---|---|
| **Before training** | The **gate** scores each sample by the first-order change in J that one SGD step on it would cause | The predicted change matches a real SGD step to three significant figures (tested) |
| **During training** | The **monitor** probes J in one forward pass every few steps and raises an alarm | The threshold is calibrated only on safe fine-tuning; false alarms are tested leave-one-seed-out |
| **On alarm** | **Rollback** to the last safe LoRA state, then the **erosion guard** removes the part of each gradient that would lower J | A guarded step leaves J unchanged to first order while removing only 4–35 % of the gradient (tested) |
| **After training** | **Verify** on safety, over-refusal, task skill and red-team probes | Compared against safety-data mixing, a content-filter gate, ReSta and steering |

Refusal-direction drift and safety-aware data filtering are established in prior work. Brink claims the
**integrated, honestly evaluated system**, not those observations. See
[docs/RELATED_WORK.md](docs/RELATED_WORK.md).

## Quickstart

```bash
pip install -e ".[judge,dashboard,dev]"
```

```bash
make test        # offline CPU test suite: tiny random models, nothing downloaded
```

```bash
make demo        # the monitor dashboard on clearly labelled simulated runs
```

### Use it in your own fine-tuning

```python
from brink.monitor.callback import BrinkCallback
from brink.monitor.core import Monitor
from brink.probe import Prober, calibrate
from brink.repair import ErosionGuard, RepairController

# 1. Fingerprint the model before fine-tuning (harmful / harmless prompt lists: calibration and probe splits).
fp, _ = calibrate(model, tok, harmful_cal, harmless_cal, harmful_probe, harmless_probe)

# 2. Monitor + alarm + repair policy.
monitor = Monitor(
    Prober(fp, tok, harmful_probe, harmless_probe),
    "runs/my-finetune",
    probe_every=4,
    alarm_signal="S3",
    alarm_threshold=-3.0,
)  # use the control-calibrated threshold
controller = RepairController(monitor, "rollback_guard", guard=ErosionGuard(tok, fp, harmful_probe, harmless_probe))

# 3. Attach to any Hugging Face Trainer or TRL SFTTrainer.
trainer = SFTTrainer(model=model, args=args, train_dataset=ds, callbacks=[BrinkCallback(controller=controller)])
trainer.train()
```

- **AlignTune:** `attach_to_aligntune(aligntune_trainer, BrinkCallback(controller=...))`.
- **CuratorKIT:** add `BrinkSafetyGate(GateScorer(...), drop_frac=0.05)` as a step in a `curatorkit.Pipeline`.

## Experiments

| # | Experiment | Question |
|---|---|---|
| 00 | [go / no-go](docs/GO_NOGO.md) | Does the alarm fire before behaviour breaks, without false alarms on safe fine-tuning? (3 seeds) |
| 01 | [data gate](docs/GATE.md) | Does the gate find the data that erodes safety, including benign data a content filter misses? |
| 02 | [repair](docs/REPAIR.md) | Does rollback + erosion guard beat safety-data mixing, ReSta and steering? |
| 03 | [multilingual](docs/MULTILINGUAL.md) | Does English fine-tuning erode safety in Hindi and Tagalog, and does English-only monitoring miss it? |
| 04 | [pipeline](docs/PIPELINE.md) | Does the full Lexsi pipeline deliver a safe, useful Hindi assistant from contaminated data? |

**Run everything:**
- **Kaggle:** [`notebooks/kaggle_run_all.ipynb`](notebooks/kaggle_run_all.ipynb). It saves results after
  every stage and resumes across sessions.
- **Any GPU machine:** `make run-all`. About 12 h on a T4 x2 in total.

Each experiment writes a `REPORT.md` with an explicit verdict, and the dashboard reads every run.

**Results:** pending the GPU runs. They'll be summarised here and in [docs/SUBMISSION.md](docs/SUBMISSION.md).

## The Lexsi stack in Brink

| Toolkit | Role |
|---|---|
| **CuratorKIT** | `BrinkSafetyGate`, a native `BaseGate`; the pipeline experiment runs a full CuratorKIT pipeline |
| **AlignTune** | Training in the pipeline experiment, with Brink attached through its TRL backend |
| **SafeTune** | Post-hoc baselines: ReSta, and steering along the fingerprint's pre-drift direction |
| **AuditKIT** | Red-team probes on every final model, with attack resistance and over-refusal reported separately |

Issues found while integrating are written up in [docs/UPSTREAM.md](docs/UPSTREAM.md).

## Responsible release

Brink deliberately trains models whose safety breaks, in order to study that. So:
- Model replies to harmful prompts are never committed or published.
- Deliberately drifted adapters are never published; the release script refuses them.
- The dashboard shows rates and labels only.

Published: fingerprints, results, and Brink-protected adapters.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): components, design decisions, repository layout.
- [Related work](docs/RELATED_WORK.md): prior work, and what Brink does and doesn't claim.
- [Dashboard](docs/DASHBOARD.md): the monitor UI and Hugging Face Space.
- [Submission](docs/SUBMISSION.md): write-up, Demo Day and video scripts.
- [Upstream feedback](docs/UPSTREAM.md): for the Lexsi toolkits.

## Development

```bash
make install-all   # + CuratorKIT, AuditKIT, SafeTune (Python >= 3.12), AlignTune
```

```bash
make lint          # ruff check + format check (CI runs this and the tests on every push)
```

## Citation

See [CITATION.cff](CITATION.cff).

## License and acknowledgements

MIT ([LICENSE](LICENSE)). The Lexsi toolkits Brink integrates with carry their own licences: AlignTune is MIT;
CuratorKIT, SafeTune and AuditKIT are LSAL source-available.

Thanks to Lexsi Labs for the toolkits, Cohere Labs for Tiny Aya and the Aya datasets, and Hugging Face. Datasets:
AdvBench, HarmBench, Alpaca, LLM-LAT harmful-dataset, Aya red-teaming, Aya dataset. Judge: WildGuard.
