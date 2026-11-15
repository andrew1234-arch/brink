# Brink, end to end

**Unprotected fine-tuning cut refusal from 77% to 4%; with Brink it ended at 69%, preventing 89% of the loss. Brink's alarm fired at step 20 of the unprotected run.**

Model `CohereLabs/tiny-aya-global` on GPU; judge: wildguard. Fine-tuning data: 128 samples,
32 of them harmful-compliance samples. Verified on 48 held-out
harmful prompts and 32 harmless ones.

| | Before fine-tuning | Unprotected | With Brink |
|---|---|---|---|
| Refuses harmful prompts | 77% | 4% | 69% |
| Attack success rate | 21% | 94% | 21% |
| Over-refusal (harmless prompts) | 0% | 0% | 0% |
| Held-out task loss | 2.016 | 1.452 | 1.462 |
| Brink alarm | | 20 | — |

**Gate.** Dropped the 32 highest-risk samples; 29 of the 32 harmful
samples were among them (precision 91% against 25% for random dropping,
recall 91%).

**Repair.** Policy `rollback_guard`: rolled back to step —; halted at —.

**Fingerprint.** Refusal direction at layer 23/36.

A small, single-seed run sized to finish in minutes. The experiments (docs/) give the multi-seed numbers with
confidence intervals.
