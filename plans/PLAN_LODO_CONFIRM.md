# PLAN_LODO_CONFIRM: registered confirmation of coordinates plus random columns on directions never trained

Written 2026-11-15 at the user's request ("跑坐标加随机列的登记确认实验"). Hashed on ws2 before the first run.

## What is known (all on seeds 148021-148025, leave-one-direction-out, MSAL's head)
PLAN_LODO_ADAPT.md: after MSAL's adaptation the coordinate descriptor (xy6) had 7.0% (k=1) / 6.9% (k=16) higher
sampled-chunk error than the harmonic one (sat), although zero-shot it was 62.6% lower. PLAN_LODO_DIM.md: xy6 plus
15 random columns (N(0, 0.7^2), drawn once per task with generator seed 20261114) was 2.4% / 3.1% lower than sat
after adaptation and 8.8% / 9.3% lower than xy6, at 6.4% higher zero-shot error than xy6. That finding came from a
test of width, not from a registered comparison of a candidate descriptor; this study confirms it as a recipe.

## Recipe under test (fixed now)
xy6r2 = the six coordinate columns plus 15 columns drawn once per task id from N(0, 0.7^2) with a new generator
seed, 20261116 (not the one of PLAN_LODO_DIM.md), so that neither the training seeds nor the draw were seen.

## Runs
Arms lodo_sat, lodo_xy6, lodo_xy6r2, all MSAL's head with the adaptation of PLAN_LODO_ADAPT.md unchanged
(tc_lodo_adapt.py, wrapped by local_run/tc_lodo_conf.py). New training seeds 148026, 148027, 148028, 148029, 148030;
16 held directions each = 80 (seed, direction) units, all three arms of a unit in one process on one machine
(ws2 GPU 1, ws3 / ws4, this computer; claims in claims_lodoc). 615 records (41 tasks x 3 arms x 5 seeds).
The pipeline is the one anchored bit for bit in PLAN_LODO_ADAPT.md and PLAN_LODO_DIM.md; a smoke run checks the
new arm before the study starts.

## Readings (draft's rule: per seed the mean over 41 tasks of the paired log ratio, t interval over the five seeds,
df 4; "lower/higher" only if the interval excludes 0; bootstrap over the 16 directions as a view)
Primary, complete method, sampled-chunk error at k = 1 and 16:
- K1 xy6r2 vs sat.
Secondary: K2 xy6r2 vs xy6 (complete method, sampled chunk, k = 1, 16); K3 zero-shot sampled chunk, xy6r2 vs sat
and xy6r2 vs xy6; K4 flow MSE for K1 and K2; K5 xy6 vs sat after adaptation (replication of the reversal); method vs
own zero-shot for each arm; descriptor shift.
Rule fixed now: the recipe is "confirmed" if K1 is lower at both k and xy6r2's zero-shot sampled-chunk error is lower
than sat's (K3); "partly confirmed" if K1 is lower at one k and not higher at the other, with K3 lower; "not
confirmed" otherwise. If confirmed, the paper may recommend coordinates plus trained random columns as the
descriptor for placements in untrained directions, stating that it was confirmed on new seeds and a new draw of the
columns, on one skill only, and not on leave-one-task-out folds or in closed loop.

## Limits
Offline, one skill and its 16-direction grid; leave-one-task-out folds and closed loop are not run.
