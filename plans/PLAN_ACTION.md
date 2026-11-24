# Sampled-action error of the two reference heads (exploratory) — written 2026-11-13 02:10, before any run

## Why
The draft's headline zero-shot numbers (MSAL's head 19.9% below the same readout without task input, 37.9%
below FiLM with a linear readout) are in flow MSE, the metric the draft itself shows to rank policies poorly in
closed loop (the simulated referee's first major concern). Sampled-chunk action error exists on the development
family only for MSAL's head (`film_nl`) and for the head trained on the task. This run supplies it for the two
reference heads so that the two headline comparisons can also be stated in action error.

## Runs
`plain_nl` (nonlinear readout, no task input) and `film` (FiLM, linear readout): the stored arms, unchanged code,
training seed 148021, leave-one-task-out over the 41 development tasks, 12,800 updates; flow MSE with the stored
scorer and sampled-chunk action error with `msal_ext.ideal.action_error` (same rows and noise draws as the
stored `film_nl` scores). Machines ws3 / ws4 through the shared queue (`nodes/tc_run_any.py`), `plain_nl` first.
If the night ends before `film` is complete, whatever is complete is reported and the rest is stated as not run.

## What is known now
All flow-MSE scores of both arms at five seeds (stored). No action error of either arm exists. The
trunk-conditioning result of seed 148021 is known; nothing of seeds 148022-148025 has been read.

## Comparisons (fixed now)
A1 `film_nl` vs `plain_nl` and A2 `film_nl` vs `film`, in sampled-action error, seed 148021: per task, the log
ratio; mean over the 41 tasks; 95% percentile bootstrap over tasks (20,000 resamples, generator seed 0); folds
won. Next to each, the same comparison in flow MSE at the same seed from the stored scores.
Check: the retrained arms' flow MSE against their stored ws2 flow MSE per fold (cross-machine reproducibility,
reported with the calibration of PLAN_5SEED.md). If a machine's calibration is outside +/-2%, the `film_nl`
side of A1 / A2 uses `film_nl` retrained on that machine for the folds concerned, or the comparison is reported
as not available.

## Limits
One training seed; the draft's unit of replication is the seed, so these are exploratory and are marked so
wherever they are written. Offline.
