# Experiment 03: multilingual drift

Tiny Aya is a multilingual model, but safety evaluation is usually English-only. Experiment 03 fine-tunes on
**English** data (the go/no-go `pos` and `control` sets) and watches refusal in English, Hindi and Tagalog.

## Questions

1. Does safety erode in languages the fine-tuning never touched?
2. Which language breaks first?
3. Does an English-only monitor see it in time, or does a worst-language monitor fire earlier?

## Data

| Language | Harmful prompts | Harmless prompts |
|---|---|---|
| English | AdvBench (probe), HarmBench (eval), exactly as go/no-go | Alpaca |
| Hindi, Tagalog | Aya red-teaming (`aya_hin.jsonl`, `aya_tgl.jsonl`): human-written by native speakers | Aya instruction dataset, filtered by language |

Each language uses 32 + 32 calibration and probe prompts per side, and 48 held-out eval prompts.

## How it works

- **One fingerprint per language.** Each has its own refusal direction, layer, and refusal / compliance
  first tokens, calibrated on prompts in that language. So S3 stays a judge-free refusal signal in every
  language.
- **Worst language wins.** `z_S3_worst` is the lowest z across languages. The analysis compares the alarm on
  English S3 alone with the alarm on `z_S3_worst`, against the first language whose behaviour breaks.
- **Caveat.** WildGuard is mostly English-trained, so Hindi and Tagalog ASR is less reliable than English. The
  per-language S3 alarm doesn't depend on the judge.

## Run

Import `notebooks/kaggle_multilingual.ipynb` (about 2 h on T4 x2). Preflight prints how many prompts it found
per language. If the Aya language filter matches nothing, adjust `aya_dataset_languages` in
`configs/multilingual.yaml`.

```bash
python experiments/multilingual.py --config configs/multilingual.yaml --out results/multilingual
```

Outputs: `REPORT.md`, `summary.json`, `multilingual.png`, and `fingerprint_<lang>.*`. Per condition, the
probe and eval CSVs carry `<lang>:`-prefixed columns.
