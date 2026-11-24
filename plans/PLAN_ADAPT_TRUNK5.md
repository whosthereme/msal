# Adaptation on the trunk-modulated head, four more seeds — written 2026-11-14 00:50, before any run of these seeds

## What is known when this is written
PLAN_ADAPT_TRUNK.md ran at seed 148021 (41 tasks, both heads per fold on one machine) and was read on 2026-11-14:
complete method on `tfilm_nl` vs on `film_nl`, sampled-chunk error -17.8% [-20.6, -14.9] at k = 1 and -18.1%
[-21.1, -15.0] at k = 16 (bootstrap over tasks), flow MSE -5.6% [-9.1, -2.3] and -6.9% [-10.3, -3.5]; residual code
beyond inversion about -8% to -9% in flow MSE on both heads. That plan's rule (not worse at both k in both errors)
was met, the five-seed run was proposed to the user, and the user asked for it. No adaptation score of seeds
148022-148025 on `tfilm_nl` exists; for `film_nl` the stored flow-MSE scores of the combination study exist.

## Runs
`tc_combo.py` as at seed 148021 (same hashes for everything it calls), with the training seed now an argument:
seeds 148022, 148023, 148024, 148025, leave-one-task-out over the 41 development tasks, k in {1, 2, 4, 8, 16}, both
arms (`film_nl`, `tfilm_nl`) of a fold trained and adapted in one process on one machine. 164 fold pairs.
Queue of (seed, task) claims on the ws3 / ws4 shared home: this computer takes seeds in ascending order and tasks
in ascending order, the ws3 / ws4 lanes (two per machine) seeds and tasks in descending order. Which machine runs
a pair is therefore not chosen by its content; every record names its lane.

## Reading (fixed now)
The draft's rule: per seed, the mean over the 41 tasks of the paired log ratio; the mean over seeds as a percentage
with a 95% t interval over seeds; folds won. Better or worse only if the interval excludes zero.
- Primary, all five seeds (df 4): A1 complete method on `tfilm_nl` vs on `film_nl`, flow MSE (SELECTION) at k = 1 and
  k = 16; A2 the same in sampled-chunk error.
- Confirmation without the seed that prompted the extension: A1 and A2 on seeds 148022-148025 (df 3). If the two
  readings differ, the four-seed reading is the one reported as confirmed.
- Secondary, five seeds: residual code beyond inversion on each head (flow MSE and sampled-chunk error, k = 1, 16);
  inversion on `tfilm_nl` vs on `film_nl` (flow MSE); zero-shot `tfilm_nl` vs `film_nl` (both errors, same-machine
  replication of PLAN_5SEED.md); the complete method against its own zero-shot head on each head (both errors).
- Second view: per task the mean log ratio over seeds; 95% percentile bootstrap over the 41 tasks (20,000
  resamples, generator seed 0). Descriptive: per-seed values; A1 at k = 2, 4, 8; ROBUSTNESS rows.
- Check: retrained `film_nl` flow-MSE scores against the stored ws2 scores of the same seed (mean and per-fold SD).
If a seed is incomplete when the work ends, the readings are reported on the complete seeds with their number.
Every comparison is reported whatever the outcome.

## Limits
Offline, development family only, the draft's 21-column descriptor; no closed loop, no LIBERO.

Note on time: first hashed on ws2 at 00:47:51 (ws2 clock); the header time 00:50 was typed ahead of the clock.
