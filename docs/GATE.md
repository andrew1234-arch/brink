# Experiment 01: the data gate

**Claim.** Brink can tell, *before* training, which samples in a fine-tuning dataset will erode the
model's refusal behaviour, including benign-looking ones.

## The score: predicted refusal erosion

One SGD step on sample *x* changes the refusal objective *J* by about `−lr · ⟨∇L(x), ∇J⟩`. So

```
score(x) = cos(∇L(x), ∇J)        > 0  →  training on x lowers J  →  erodes refusal
```

- *J* is the **monitor's own signal**: S3 (refusal-vs-compliance first-token log-odds) and S1 (refusal
  margin on the refusal direction), computed on the held-out harmful probe set. The gate predicts what the
  monitor would later see. One mechanism, used before and during training.
- Gradients are w.r.t. freshly initialised LoRA adapters (r = 8). With B = 0, the B-gradient is the
  full-weight gradient times Aᵀ: a random projection that preserves inner products in expectation (the
  same idea as LESS, Xia et al. 2024). Cost: one forward + backward per sample, no training.
- **Verified:** on a test model, the predicted change in *J* after one real SGD step matches the actual
  change to three significant figures, for both signs (`tests/test_gate.py`).

**Baselines:** `rep`, the representation-similarity method of He et al. 2024 ("What's in your safe
data?"), using reply hidden states vs unsafe / safe anchors; and `random`.

## Design

| Part | What | Pass rule |
|---|---|---|
| A. Detection | Pool of 2,000 = 1,950 Alpaca + 50 planted harmful-compliance samples. Score all of them. | D1: AUROC(grad_S3) ≥ 0.9 · D1b: grad beats rep |
| B. Prediction | Benign samples only. Fine-tune Tiny Aya on the **top**, **random** and **bottom** 256 by score (96 steps, monitored). | D2: ASR(top) ≥ ASR(random) + 10 pts · D3: ASR(bottom) ≤ ASR(random) + 2 pts |

Part B is the real result: fine-tuning on benign data breaks safety, and the gate says *which* benign
data. Held-out benign loss is tracked so we can see the capability cost.

## Use in a real pipeline (CuratorKIT)

```python
from brink.gate import GateScorer
from brink.gate.curatorkit_gate import BrinkSafetyGate

scorer = GateScorer(model, tok, fingerprint, harmful_probe, harmless_probe, targets=["S3"])
gate = BrinkSafetyGate(scorer, drop_frac=0.05)      # or threshold=...
passed, rejected = gate.run(samples)                # or as a step in curatorkit.Pipeline
```

Rejected samples land in CuratorKIT's `rejected.jsonl` with reason `brink_safety_drift:<score>`. Every
sample gets a provenance record with its score and rank.

## How to run (Kaggle)

Import `notebooks/kaggle_gate.ipynb`. The setup is the same as go / no-go: T4 x2, Internet on, the two
secrets. Run it **after** reviewing the go / no-go result: if the drift learning rate needed changing
there, change `train.lr` in `configs/gate.yaml` too. About 1.5 h:
- scoring 2,000 samples takes about 15 min;
- each of the 3 fine-tunes takes about 20 min.

```bash
python experiments/gate.py --config configs/gate.yaml --out results/gate
```

Outputs: `REPORT.md`, `summary.json`, `gate.png`, `scores.csv` (ids and scores only, no text),
`subsets.json`, and per-subset logs. `pool.jsonl` (sample texts, including the planted harmful ones)
stays out of git.
