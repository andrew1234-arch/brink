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

The Space needs `app/app.py`, the `brink/` package and `app/requirements.txt`, plus the run folders to show
(`results/...`, small CSV / JSON files only). The app reads from `results/` by default; set `--results`
if the runs live elsewhere.

## Design notes

- Colours follow validated categorical slots (blue, orange, aqua), in both light and dark mode. Every line is
  direct-labelled and a table view exists, because the light-mode aqua is below 3:1 contrast. Status colours
  are only used for state, always with an icon and a label. There are no dual-axis charts.
- `brink/dashboard/` (runs, metrics, figures) has no Gradio dependency and is unit-tested. `app/app.py` is a
  thin UI layer.
