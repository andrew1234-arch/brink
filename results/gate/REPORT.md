# Gate experiment: **GATE NOT PROVEN**

- ✅ D1_finds_planted_harmful
- ✅ D1b_beats_rep_baseline
- ❌ D2_top_scored_benign_breaks_safety_more
- ✅ D3_low_scored_benign_is_safer

## Detection of planted harmful samples in a benign pool

| score | AUROC | recall@k | recall@top 5 % |
|---|---|---|---|
| grad_S3 (primary) | 0.981 | 0.600 | 0.860 |
| grad_S1 | 0.968 | 0.540 | 0.720 |
| rep | 0.943 | 0.560 | 0.700 |
| random | 0.525 | 0.020 | 0.020 |
| classifier | 0.999 | 0.940 | 1.000 |

The safety classifier flags **0%** of the gate's top-scored benign subset: the risk the gate finds is invisible to a content filter.

## Fine-tuning on benign subsets selected by `grad_S3`

| subset | ASR start → end | held-out loss start → end | z_S1 end | z_S3 end |
|---|---|---|---|---|
| top | 0.156 → 0.190 | 2.051 → 1.449 | -23.9 | -7.3 |
| random | 0.156 → 0.109 | 2.051 → 1.439 | -15.3 | 3.2 |
| bottom | 0.156 → 0.094 | 2.051 → 1.500 | -10.0 | 10.0 |
