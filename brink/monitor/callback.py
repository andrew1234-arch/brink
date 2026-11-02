"""Brink inside Hugging Face Trainer, TRL and AlignTune.

    from brink.monitor.callback import BrinkCallback
    trainer = SFTTrainer(..., callbacks=[BrinkCallback(monitor)])                  # monitor only
    trainer = SFTTrainer(..., callbacks=[BrinkCallback(controller=controller)])    # + halt / rollback / guard

AlignTune (TRL backend): AlignTune builds its own SFTTrainer inside `.train()` and only forwards log /
evaluate / save events to its own callbacks, with state=None and control=None. So an AlignTune-native
callback can't see the step or stop training. `attach_to_aligntune` adds Brink to the Hugging Face trainer
AlignTune builds, next to AlignTune's own adapter:

    from aligntune.core.backend_factory import create_sft_trainer
    at = create_sft_trainer(model_name=..., dataset_name=..., backend="trl", ...)
    attach_to_aligntune(at, BrinkCallback(controller=controller))
    at.train()
"""

from __future__ import annotations

from transformers import TrainerCallback

from brink.monitor.core import Monitor


class BrinkCallback(TrainerCallback):
    def __init__(self, monitor: Monitor | None = None, controller=None):
        if (monitor is None) == (controller is None):
            raise ValueError("pass exactly one of monitor= or controller=")
        self.monitor = monitor if monitor is not None else controller.monitor
        self.controller = controller
        self._attached = False

    def _attach(self, model, optimizer):
        if self.controller is not None and not self._attached and model is not None:
            self.controller.attach(model, optimizer)
            self._attached = True

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if model is None:
            return control
        self._attach(model, optimizer)
        if self.controller is not None:
            if self.controller.on_step(model, 0) == "stop":
                control.should_training_stop = True
        else:
            self.monitor(model, 0)
        return control

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        # Runs after gradient clipping, right before optimizer.step(): where the erosion guard projects.
        if self.controller is not None and model is not None:
            self._attach(model, optimizer)
            self.controller.after_backward(model, state.global_step + 1)
        return control

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if model is None:
            return control
        if self.controller is not None:
            self._attach(model, optimizer)
            if self.controller.on_step(model, state.global_step) == "stop":
                control.should_training_stop = True
        else:
            self.monitor(model, state.global_step)
        return control


def attach_to_aligntune(aligntune_trainer, callback: TrainerCallback):
    """Add `callback` to the Hugging Face trainer that an AlignTune trainer builds when `.train()` runs."""
    if not hasattr(aligntune_trainer, "get_hf_callbacks"):
        raise TypeError("expected an AlignTune trainer (TRL backend) with get_hf_callbacks()")
    original = aligntune_trainer.get_hf_callbacks

    def get_hf_callbacks():
        return list(original()) + [callback]

    aligntune_trainer.get_hf_callbacks = get_hf_callbacks
    return aligntune_trainer
