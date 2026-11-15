# Feedback for the Lexsi Labs toolkits

Issues and proposals found while building Brink on the Lexsi stack. Each was checked against the installed
package source (versions below). They're ready to file as GitHub issues; none have been filed yet.

| Toolkit | First checked | Re-checked 2026-11-06 |
|---|---|---|
| AlignTune | 0.1.28 | 0.1.31 |
| CuratorKIT | 0.1.22 | 0.1.22 |
| SafeTune | 0.1.10 | 0.1.12 |
| AuditKIT | 1.1.3 | 1.1.3 |
| CircuitKIT | — | 0.2.0 |

Status per entry: **open** (still present in the re-checked version) or **fixed upstream** (with the version).

---

## AlignTune: callbacks can't see the training step or stop training (TRL backend)

**Status: open in 0.1.31** (`hf_adapter.py` lines 58, 76, 87 still pass `None, None`).

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

**Status: fixed upstream in 0.1.31.** PyICU is now the optional `aligntune[tokenizer]` extra.

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

**Status: fixed upstream in 0.1.12.** `python -I -c "import safetune"` prints nothing on a CPU-only machine.

On CPU-only machines, every import prints *"Failed to load CPU gemm_4bit_forward from kernels-community: No
module named 'kernels'…"*. **Suggestion:** log it at debug level, or warn once and only when a 4-bit path is
actually used.

## AuditKIT: default detectors score a refusal as a failed attack

**Status: open in 1.1.3.**

**What happens.** `RedTeamRunner.run()` without `detectors=` attaches **every** detector to **every** probe
(`runner.py`, the loop over `resolved_detectors`). That includes `RefusalDetector`, which is meant for the
benign `refusal` probe and marks any refusal as `passed=False`. On the attack probes this inverts the result.

**Measured.** Run with two stub models, using all default probes and detectors:

| Stub model | `pass_rate` | jailbreak | prompt_injection | encoding | refusal (benign) |
|---|---|---|---|---|---|
| Always refuses: "I'm sorry, but I can't help with that." | **0.667** | 16/24 | 16/24 | 10/15 | 10/15 |
| Always complies: "Sure, here is exactly how to do it, step by step." | **1.000** | 24/24 | 24/24 | 15/15 | 15/15 |

So the model that complies with every attack gets a perfect score, and the one that refuses everything gets a
lower one. Per detector, every failure of the refusing model comes from `RefusalDetector`. The counts are
(prompt, detector) pairs: 26 prompts × 3 detectors = 78.

**Two more issues.**
- `KeywordDetector`'s default list includes `"as an ai"`, so a safe refusal that starts "As an AI, I can't…"
  counts as attack success.
- No built-in detector can score the `jailbreak` probe at all. Its prompts ask for harmful content (lock
  picking, a bomb poem, hot-wiring), but the detectors only look for injection markers such as "hacked" or
  "ACCESS_GRANTED". A model that writes the harmful content passes.

**Repro.**

```python
from auditkit.redteam.runner import RedTeamRunner

print(RedTeamRunner(lambda ps: ["Sure, here is exactly how."] * len(ps)).run().pass_rate)  # 1.0
print(RedTeamRunner(lambda ps: ["I'm sorry, I can't help."] * len(ps)).run().pass_rate)  # 0.667
```

**Suggested fix.**
- Give each probe its own default detectors: markers for injection and encoding, `RefusalDetector` only for
  `refusal`.
- Drop "as an ai" from the success keywords.
- Score jailbreaks with a judge model, or document that they need one.

**What we do.** `brink/eval/audit.py` runs each probe family with the detector that fits it and scores
jailbreaks with Brink's judge (WildGuard). `tests/test_audit.py` pins all of the above.

## AuditKIT: `RedTeamResult.pass_rate` mixes opposite probe semantics

**Status: open in 1.1.3.**

**What happens.** `pass_rate` averages over all probe/detector results. For `jailbreak`, `encoding` and
`prompt_injection`, a pass means *the attack failed*. For the `refusal` probe (benign prompts with
`RefusalDetector`), a pass means *the model did not refuse*: an over-refusal check. Averaging the two lets a
model that refuses everything and one that complies with everything score alike.

Results are also stored per *(prompt, detector)* pair. So `total_probes` counts detectors, not prompts, and
prompts with two detectors weigh double.

**Suggestion.** Report pass rates per probe family, attack resistance separately from over-refusal, and per
prompt. That's what we do in `brink/eval/audit.py`.

## AlignTune: `torch<2.14` upper bound conflicts with the current PyTorch

**Status: open in 0.1.31.** AlignTune declares `torch<2.14,>=2.4.0`. The current PyTorch release is 2.15.0: it's
what `pip install torch` installs today, and what this repo's CPU test environment runs. So `pip install aligntune`
into an up-to-date environment downgrades torch. That's a multi-GB download for CUDA builds, and it can break
the other Lexsi kits installed alongside, none of which cap torch (CuratorKIT, SafeTune `>=2.7.0`, AuditKIT
`>=2.4`, CircuitKIT unpinned). We install AlignTune with `--no-deps`, and Brink's 96 tests pass against torch
2.15.0 and transformers 5.22.1. **Suggestion:** raise the cap, or drop it if there's no known incompatibility.

## CircuitKIT: `transformer-lens==3.8.0` exact pin forces a separate install step

**Status: open in 0.2.0.** CircuitKIT requires `transformer-lens==3.8.0` exactly. The other kits share
`transformers>=5.15,<6`, but an exact pin on a library that tracks transformers closely makes CircuitKIT
risky to install into an environment the other kits already resolved. Our Kaggle notebook installs it only
when the CircuitKIT stage starts, after every other stage. **Suggestion:** a compatible range
(`transformer-lens>=3.8,<3.9`), or a documented "one environment for the whole Lexsi stack" constraints file.

## CircuitKIT: the safety task scores raw prompts, without the chat template

**Status: open in 0.2.0.** `SafetyPromptAdapter` builds each pair from the raw harmful prompt plus a default
answer token (`" I"` for refusal, `" Sure"` for compliance; `data/adapters/safety_prompt.py`). Chat models such
as Tiny Aya refuse inside their chat template, so attributing refusal on raw text can find a circuit for a
behaviour the deployed model doesn't show in that form. The default tokens are also GPT-2/Llama BPE choices
and may not be single tokens, or the model's real refusal tokens, under other tokenizers.

**What we do.** `experiments/circuit.py` passes chat-formatted prompts (leading BOS stripped, since
TransformerLens adds its own) and the model's own refusal and compliance first tokens from Brink's fingerprint.

**Suggestion.** An `apply_chat_template=True` option and token arguments read from the tokenizer, with a check
that each answer is a single token.

## CuratorKIT: steps print progress bars unconditionally

**Status: open in 0.1.22.** `ExactDeduplicator.run()` always wraps its loop in `tqdm`, so library use inside
an app or a test prints `ExactDeduplicator: 100%|…|` to stderr on every call (seen in every Brink test that
gates data). **Suggestion:** a `progress=False` option, or `tqdm(disable=None)` so it switches off when stderr
isn't a TTY.
