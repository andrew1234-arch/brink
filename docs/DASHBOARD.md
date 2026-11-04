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

**Automatic deploy (recommended).** `.github/workflows/deploy-space.yml` runs `scripts/deploy_space.py`:

- **After CI passes on a push to `main`:** deploys exactly the commit CI tested, as long as it is still the
  tip of `main`, so a slow CI run on an older commit can't roll the Space back. Untested code is never deployed,
  and CI runs for pull requests (including from forks) never trigger a deploy.
- **Hourly:** picks up new results from the `results/all` branch. That branch holds only results, so a push
  to it can't trigger a workflow. If neither the code nor the results changed, the hourly run only re-applies
  the visibility setting and doesn't fetch or upload anything.
- **Manually:** from the Actions tab. Manual runs have their own queue, so the hourly run never replaces one.

Results come **only** from the `results/all` branch, extracted to a separate folder; anything under `results/`
on `main` is ignored. The Space publishes **finished experiments only** (those with a `REPORT.md`), so a stage
that failed half-way never appears. Until at least one experiment has finished it shows the clearly bannered
simulated runs, never a mix of the two. If the results branch can't be read, for
example after a network or authentication error, the deploy **fails** rather than falling back to the demo, so
real results are never replaced by simulated ones. Each deploy mirrors the build exactly and removes anything
stale. Visibility is applied on every deploy, so changing `HF_SPACE_PUBLIC` takes effect next time.

One-time setup, in the GitHub repo under **Settings → Secrets and variables → Actions**:

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | a Hugging Face token with **write** access |
| Variable | `HF_SPACE_ID` | e.g. `your-hf-username/brink` (the Space is created if it doesn't exist) |
| Variable | `HF_SPACE_PUBLIC` | `true` to make it public; anything else keeps it private |

While `HF_SPACE_ID` is unset, the deploy job is skipped. While `HF_TOKEN` is unset, the deploy step is skipped
with a notice rather than failing.

**By hand:**

```bash
python scripts/build_space.py --out build/space --demo --push your-hf-username/brink
```

This needs `HF_TOKEN` in the environment. Add `--public` for a public Space. The Gradio version is pinned once,
in `app/requirements.txt`: CI tests that exact version and the Space deploys it. The Space contains no model
replies, tensors or adapters.

## Design notes

- Colours follow validated categorical slots (blue, orange, aqua), in both light and dark mode. Every line is
  direct-labelled and a table view exists, because the light-mode aqua is below 3:1 contrast. Status colours
  are only used for state, always with an icon and a label. There are no dual-axis charts.
- `brink/dashboard/` (runs, metrics, figures) has no Gradio dependency and is unit-tested. `app/app.py` is a
  thin UI layer.
