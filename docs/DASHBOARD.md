# Monitor dashboard

A Gradio app that shows what Brink sees during fine-tuning, for any experiment run: finished, or still
being written.

```bash
pip install -r app/requirements.txt          # gradio, plotly, numpy: no ML stack needed
```

```bash
python app/app.py                            # real runs under results/
```

```bash
python app/app.py --results path/to/runs     # any folder of runs
```

```bash
python app/app.py --demo                     # SIMULATED runs, clearly bannered, for trying the UI
```

## What it shows

| Area | Content |
|---|---|
| Status tiles | SAFE / DRIFTING / ALARM (icon + label), training step, ASR start → now, alarm step, behaviour-break step, lead time (or how late), probe-vs-eval cost |
| Probe signals | z-scores of S1 / S2 / S3 vs the reference model, the alarm threshold, the alarm and behaviour-break lines, and the lead-time span |
| Ground truth | ASR on held-out harmful prompts for every condition in the run |
| Layers | Diverging heatmap of the refusal-margin change at every layer over training (red = eroding), with the monitored layer marked |
| Across conditions | The chosen alarm signal for every condition, e.g. `pos` vs `control` |
| Data gate | Gate-score distribution for planted vs benign samples, and ASR of the top / random / bottom subsets |
| Tables | Alarm summary per condition and the eval rows with their z-scores, as text |

**The alarm is recomputed from the logs on every refresh.** Changing the signal, z threshold, number of
consecutive probes, or switching to CUSUM updates the alarm step and lead time immediately. That makes the
detection rule easy to show live on Demo Day.

The page never shows model replies to harmful prompts; only rates and labels.

## Live during training (Kaggle)

In `notebooks/kaggle_go_nogo.ipynb`, set `LIVE_DASHBOARD = True` before the run cell. It prints a public
`gradio.live` link; open it and tick **Live** to watch the signals and the alarm update as the model trains.
Locally, `python app/app.py --demo-live` shows the same behaviour with a simulated run.

## Hugging Face Space

**Automatic deploy (recommended).** `.github/workflows/deploy-space.yml` builds the Space and pushes it on every
push to `main` or `results/all`, or on demand from the Actions tab. It uses the real runs from `results/all`
once any exist, and the clearly bannered simulated runs until then. Each deploy replaces the previous results.

One-time setup, in the GitHub repo under **Settings → Secrets and variables → Actions**:

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | a Hugging Face token with **write** access |
| Variable | `HF_SPACE_ID` | e.g. `your-hf-username/brink` (the Space is created if it doesn't exist) |
| Variable | `HF_SPACE_PUBLIC` | `true` to make it public; anything else keeps it private |

Without the secret and variable, the workflow does nothing and says so in its log.

**By hand:**

```bash
python scripts/build_space.py --out build/space --demo --push your-hf-username/brink
```

This needs `HF_TOKEN` in the environment. Add `--public` for a public Space. The Space pins the Gradio version
the dashboard is tested with, and contains no model replies, tensors or adapters.

## Design notes

- Colours follow validated categorical slots (blue, orange, aqua), in both light and dark mode. Every line is
  direct-labelled and a table view exists, because the light-mode aqua is below 3:1 contrast. Status colours
  are only used for state, always with an icon and a label. There are no dual-axis charts.
- `brink/dashboard/` (runs, metrics, figures) has no Gradio dependency and is unit-tested. `app/app.py` is a
  thin UI layer.
