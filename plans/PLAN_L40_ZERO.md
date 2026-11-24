# PLAN_L40_ZERO: do unseen-word descriptor entries cause the trunk head's loss on LIBERO40?

Written 2026-11-14 after PLAN_L40_TRUNK.md was read out, at the user's request ("Yes, run it" to: run a check of
the unseen-word explanation). Hashed on ws2 before the first run.

## What is known (PLAN_L40_TRUNK.md, five seeds)
Complete method, trunk-modulated head (tfilm_nl) vs the draft's head (film_nl): sampled-chunk error +3.6% (k=1),
+3.7% (k=16), flow error not distinguishable. Zero-shot flow +44.9%. Post hoc: +79% on the 85 folds (17 held tasks x
5 seeds) whose instruction has a word absent from all 39 source instructions, +24% on the other 115.

## Hypothesis (post hoc, from Proposition 5)
The descriptor entry of an unseen word is zero in every training descriptor, so its columns of the conditioning
weights (Gamma, Delta) get no gradient and keep a weight-decayed copy of their random N(0, 0.02^2) initialization.
At test time this adds a random modulation. In the trunk head it acts before two more layers. Proposition 5's
minimum-norm choice sets the unidentified component of the descriptor to zero.

## Intervention
Projected descriptor: d_proj = d* with the entries of words absent from all source descriptors set to zero (equal to
zeroing those weight columns). Nothing else changes: same training, same library, same inversion (200 Adam steps,
started from d_proj instead of d*, all entries free), same residual code, same scoring.

## Folds and runs
The 17 held tasks with an unseen word (1000, 1001, 1005, 1006, 1008, 1009, 1010, 1011, 1012, 1017, 1021, 1025, 1026,
1027, 1028, 1030, 1031) x seeds 148021-148025 = 85 (seed, task) pairs, both heads per pair in one process on one
machine. Per head: stage 1 and stage 2 trained once, then the complete PLAN_L40_TRUNK.md evaluation twice, with d*
(raw) and with d_proj (projected): zero-shot flow (SELECTION, ROBUSTNESS), inversion and complete method at
k = 1, 2, 4, 8, 16, sampled-chunk error zero-shot and at k = 1, 16. Runner: local_run/tc_l40z.py (imports tc_l40.py
unchanged). On the other 23 tasks d_proj = d*, so the projected result equals the stored one.

Machines: ws2 GPU 1 (two lanes in one gpu_run), ws3 / ws4 (two lanes each), this computer (one lane); claims in
claims_l40z on the ws3 / ws4 home. Anchor: pair 148021 / 1000 runs first on ws2; its raw results must equal the
stored PLAN_L40_TRUNK.md records of that pair bit for bit (checked by equality only; no projected number is looked at
before all runs end). On ws2 every raw record must equal its stored record when that record also came from ws2.

## Readings (draft's rule)
Per seed, mean over its folds of the paired log ratio; mean over seeds with t interval (df 4); "lower/higher" when the
interval excludes 0. Bootstrap over tasks (20,000, seed 0) as a view.

E (explanation), on the 85 folds:
- E1 zero-shot, trunk head, projected vs raw: flow and sampled chunk.
- E2 zero-shot, trunk vs head, both projected: flow and sampled chunk (compare with the raw gap).
- E3 zero-shot, film_nl head, projected vs raw.
The explanation is called supported if E1 flow is lower and the trunk-vs-head zero-shot flow gap with projection (E2)
is less than half the raw gap on the same folds (in log ratio); otherwise it is called not supported.

D (decision), on all 200 folds (85 new pairs, projected; 115 stored PLAN_L40_TRUNK.md pairs, where d_proj = d*):
- D1 complete method, trunk vs head, both projected, flow at k = 1 and 16.
- D2 the same, sampled-chunk error at k = 1 and 16.
Rule fixed now: the trunk head with projected descriptors is proposed to the user as the paper's main method only if
D1 and D2 are not worse (interval not above 0) at both k. Otherwise the paper keeps film_nl.

P (projection for the draft's head), on the 85 folds: complete method, film_nl, projected vs raw, flow and sampled
chunk at k = 1 and 16. Projection is proposed as part of the method only if sampled-chunk error is lower at both k
and flow is not higher at either.

Secondary, descriptive: the same readings for inversion alone; k = 2, 4, 8; ROBUSTNESS rows; descriptor shift.

Whatever comes out is written into the supplement, and into the main paper where it changes a claim.
