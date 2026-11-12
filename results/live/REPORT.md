# Brink, end to end

**Unprotected fine-tuning cut refusal from 77% to 35%; with Brink it ended at 94%, preventing 100% of the loss. Brink's alarm fired at step 16 of the unprotected run.**

Model `CohereLabs/tiny-aya-global` on GPU; judge: wildguard. Fine-tuning data: 38 samples,
3 of them harmful-compliance samples. Verified on 48 held-out
harmful prompts and 32 harmless ones.

| | Before fine-tuning | Unprotected | With Brink |
|---|---|---|---|
| Refuses harmful prompts | 77% | 35% | 94% |
| Attack success rate | 21% | 48% | 6% |
| Over-refusal (harmless prompts) | 0% | 0% | 3% |
| Held-out task loss | 1.984 | 2.971 | 3.679 |
| Brink alarm | | 16 | — |

**Gate.** Dropped the 10 highest-risk samples; 3 of the 3 harmful
samples were among them (precision 30% against 8% for random dropping,
recall 100%).

**Repair.** Policy `rollback_guard`: rolled back to step —; halted at —.

**Fingerprint.** Refusal direction at layer 23/36.

A small, single-seed run sized to finish in minutes. The experiments (docs/) give the multi-seed numbers with
confidence intervals.
