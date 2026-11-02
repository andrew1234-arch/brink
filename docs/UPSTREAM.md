# Feedback for the Lexsi Labs toolkits

Issues and proposals found while building Brink on the Lexsi stack. Each was checked against the installed
package source (versions below). They're ready to file as GitHub issues; none have been filed yet.

| Toolkit | Version checked |
|---|---|
| AlignTune | 0.1.28 |
| CuratorKIT | 0.1.22 |
| SafeTune | 0.1.10 |
| AuditKIT | 1.1.3 |

---

## AlignTune: callbacks can't see the training step or stop training (TRL backend)

**What happens.** `TrainerBase.get_hf_callbacks()` wraps AlignTune callbacks in `HuggingFaceCallbackAdapter`.
The adapter forwards only `on_log`, `on_evaluate` and `on_save` to AlignTune's `CallbackHandler`, and passes
`state=None, control=None`
(`aligntune/core/callbacks/hf_adapter.py`: `self.callback_handler.on_log(self.aligntune_config, None, None, logs=logs)`).

**Consequences.**
- An AlignTune callback never gets `on_step_end`, `on_train_begin` or `on_pre_optimizer_step`.
- It can't read `state.global_step`.
- It can't stop training (`control.should_training_stop`).

So in-training monitors and gradient-level interventions can't be written as AlignTune callbacks. That
includes AlignTune's own `AlignmentAuditCallback` use case, which needs to act on drift.

**Suggested fix.** Forward every `TrainerCallback` event, with the real `state` and `control`, and return the
`control` the handler produces.

**Workaround we use.** Append our own Hugging Face callback to the list `get_hf_callbacks()` returns
(`brink.monitor.callback.attach_to_aligntune`).

## AlignTune: `pip install aligntune` fails without system ICU

`PyICU` is a hard dependency, and building it needs `libicu-dev` and `pkg-config`. In a clean Python 3.13
environment, `pip install aligntune` fails with *Failed to build 'PyICU'*. PyICU appears to serve the Indic
tokenizer-extension features. **Suggestion:** move it to an extra, e.g. `aligntune[indic]`.

## CuratorKIT: proposal, a safety-drift gate

`BaseGate` was easy to build on. The `(passed, rejected)` contract and provenance records fit naturally.
We'd like to contribute `BrinkSafetyGate` (`brink/gate/curatorkit_gate.py`). It scores each sample by the
first-order change in a refusal objective that one SGD step on it would cause, using gradients against fresh
LoRA adapters, and it rejects the riskiest share. It's verified against a real SGD step to three significant
figures (`tests/test_gate.py`). It would give CuratorKIT a pre-training safety check that complements the
hallucination, reward and diversity gates.

## SafeTune: steering hooks are invisible in `output_hidden_states` under transformers 5

**What happens.** transformers 5 records `hidden_states` with its own hook on each decoder layer, and that
hook runs before hooks registered later, such as `RefusalDirectionModel`'s steering hook. So after steering at
layer *i*, `hidden_states[i+1]` still shows the *unsteered* output; the shift only appears from
`hidden_states[i+2]` on. Generation and the actual forward computation are correctly steered.

**Why it matters.** Anything that measures the effect of steering through `output_hidden_states` will
underestimate it at the steered layer. That includes interpretability or monitoring run on a steered model.

**Suggestion.** Note this in the steering docs, or measure with a forward pre-hook on the next layer, as
`tests/test_repair.py::test_reference_steer_restores_projection` does.

## SafeTune: `import safetune` prints a kernels warning every time

On CPU-only machines, every import prints *"Failed to load CPU gemm_4bit_forward from kernels-community: No
module named 'kernels'…"*. **Suggestion:** log it at debug level, or warn once and only when a 4-bit path is
actually used.

## AuditKIT: `RedTeamResult.pass_rate` mixes opposite probe semantics

**What happens.** `pass_rate` averages over all probe/detector results. For `jailbreak`, `encoding` and
`prompt_injection`, a pass means *the attack failed*. For the `refusal` probe (benign prompts with
`RefusalDetector`), a pass means *the model did not refuse*: an over-refusal check. Averaging the two lets a
model that refuses everything and one that complies with everything score alike.

Results are also stored per *(prompt, detector)* pair. So `total_probes` counts detectors, not prompts, and
prompts with two detectors weigh double.

**Suggestion.** Report pass rates per probe family, attack resistance separately from over-refusal, and per
prompt. That's what we do in `brink/eval/audit.py`.
