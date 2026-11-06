# Stage "circuit": CircuitKIT's refusal circuit vs the layer Brink monitors

**Question.** Brink reads refusal from one direction at one layer, l*, chosen by calibration. Is that where the
model's refusal mechanism actually is? CircuitKIT gives an independent answer.

**Method** (`experiments/circuit.py`, `configs/circuit.yaml`, about 0.6 h on one T4):
1. Load go / no-go's fingerprint: l*, Brink's per-layer refusal separation (Cohen's d), and Tiny Aya's refusal
   and compliance first tokens.
2. Register a CircuitKIT task: chat-formatted AdvBench prompts, with Brink's refusal token as the clean answer and
   its compliance token as the corrupt one (`SafetyPromptAdapter`, answer-contrastive pairing).
3. Run CircuitKIT discovery with EAP-IG, which scores every attention head and MLP by how much it drives refusal
   over compliance.
4. Compare (`brink/analysis/circuit.py`):
   - the share of the circuit's score mass in blocks upstream of l*. Brink reads the residual stream entering
     block l*, so that's what it can see;
   - the Spearman correlation between Brink's per-layer separation and the circuit mass written before each
     layer.

**Pass criterion.** At least half the circuit mass is upstream of l*, and the correlation is positive.

**Outputs.**
- `results/circuit/REPORT.md` and `summary.json`.
- `circuit_scores.json`, CircuitKIT's own artifact. Its `safety_units` / `layer_suggestions` can be read by
  SafeTune's CircuitKIT adapter.

**Caveats.**
- CircuitKIT marks its Tiny Aya support experimental.
- CircuitKIT pins its own `transformers` and `transformer-lens`, so the notebook installs it only when this
  stage starts, after every other stage.

**Result:** ⟦RESULT⟧
