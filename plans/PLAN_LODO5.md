# Unseen direction, four more training seeds — written 2026-11-13 16:50, before any run of these seeds

## What is known when this is written
PLAN_LODO.md ran at seed 148021 on this computer (all five arms, 16 folds each). Read on 2026-11-13 16:30 with the
bootstrap over 16 directions: the metric descriptor `lodo_xy6` vs the draft's `lodo_sat` sampled-action error
-63.5% [-67.4, -58.0], flow MSE -62.3%; `lodo_xy6` vs no description `lodo_plain` action -63.0%, flow -21.3%;
`lodo_sat` vs `lodo_plain` action +1.4% (n.d.), flow +108.5%; `lodo_txy6` (trunk FiLM, 6 columns) vs `lodo_xy6`
action -18.4%, flow -13.1%. The decision to add seeds, and which arms, was made after seeing these numbers.
No score of seeds 148022-148025 exists for any arm of this design.

## Runs
Arms `lodo_plain`, `lodo_sat`, `lodo_xy6`, `lodo_txy6` exactly as in PLAN_LODO.md (`tc_lodo.py`, unchanged except
that the training seed is now an argument and the Windows DLL preload is skipped on Linux) at training seeds
148022, 148023, 148024, 148025: 16 leave-one-direction-out folds per arm and seed, 12,800 updates, the stored
recipe, each held task scored on its SELECTION rows (flow MSE, ROBUSTNESS flow MSE, sampled-chunk action error)
with the noise draws of its seed. 256 folds. `lodo_h9` is not extended.
Machines: every arm of one seed runs in one build family, so that every comparison is same-build. This
computer runs seeds 148022 and 148023, one process at a time, order sat, xy6, plain, txy6 per seed. ws3 / ws4
(one build family) run seeds 148024 and 148025 through a claim queue of (seed, arm) units, same order.

## Comparisons (fixed now)
L1 `lodo_xy6` vs `lodo_sat`; L2 `lodo_xy6` vs `lodo_plain`; L3 `lodo_sat` vs `lodo_plain`; L4 `lodo_txy6` vs
`lodo_xy6`. Primary metric: sampled-action error; then flow MSE (SELECTION), then ROBUSTNESS flow MSE.
Reading rule of the draft: per seed, the mean over the 41 held tasks of the paired log ratio; mean over seeds as a
percentage with a 95% t interval over seeds; tasks and directions won. Better or worse only if the interval
excludes zero.
- All five seeds (148021 included), df 4.
- Confirmation without the seed that prompted the extension: seeds 148022-148025 only, df 3. If the two
  readings differ, the four-seed reading is the one reported as confirmed.
- Second view: per task, the mean log ratio over seeds; 95% percentile bootstrap over the 16 directions
  (20,000 resamples, generator seed 0).
- Descriptive: per-seed values; context against the stored leave-one-task-out scores of the same seeds.
If a seed is incomplete when the work ends, the reading is reported on the complete seeds with their number.

## Limits
Offline, zero-shot, one simulated skill and its 16-direction grid. Descriptor inversion, the residual library and
closed loop are not run with these descriptors.

Note on time (2026-11-13, ws2 clock 16:28 when first hashed): the header time 16:50 was typed ahead of the clock.
