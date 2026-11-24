# Zero-shot to an unseen direction: leave-one-direction-out (exploratory) — written 2026-11-13 02:40, before any run

## Why
The development descriptor of the draft (21 columns: 16 harmonics of the target direction on the 16-direction
grid, 3 powers of the ring index, 2 interaction terms) spans every function of direction on that grid: a linear
map turns its angular block into a one-hot code of the 16 directions, and adjacent directions have cosine
similarity 0.03. In all 205 leave-one-task-out folds the direction and the ring of the held task both occur in
training, so those folds test a new combination of a seen direction and a seen ring. The scenario of the
introduction (placements given as coordinates) needs targets whose direction was never trained. In the
registered coverage study an unseen direction multiplied zero-shot error by about 1.33 at equal distance
(post hoc regression of 2026-11-12). This run tests an unseen direction directly and asks whether a metric
descriptor helps there.

## Design
Development family, language-blind features, training seed 148021, 12,800 updates, the stored recipe
(`run_loto.train_arm`). Sixteen folds: fold m holds out every task of direction m (2 or 3 tasks) and trains on
the other 38 or 39 tasks. Each held task is scored on its SELECTION rows with its own descriptor: flow MSE
(stored scorer) and sampled-chunk action error (`msal_ext.ideal.action_error`, stored draws).

Arms (the head is the zero-shot head of the draft, FiLM on h then the nonlinear readout, unless noted):
- `lodo_plain`  no task input (nonlinear readout): the reference for whether the description helps at all.
- `lodo_sat`    the 21-column descriptor of the draft (`saturated_v5`).
- `lodo_xy6`    6 columns: 1, cos, sin, rho, rho cos, rho sin (`physical_xy_radial_v5`); the target's x and y
         are linear in these, so this is the target coordinate plus the ring.
- `lodo_h9`     9 columns (`harmonic2_radial2_v5`): the above plus the second harmonic and rho squared.
- `lodo_txy6`   last, if time allows: the 6-column descriptor through FiLM inside the trunk (the position of
         `tfilm_nl`).
Feature sets are the repository's own (`lib/task_algebra_v5.py`); nothing is tuned.
Order of runs: plain, sat, xy6, h9, txy6. Whatever is complete when the night ends is reported.

## Comparisons (fixed now)
L1 `lodo_xy6` vs `lodo_sat`; L2 `lodo_xy6` vs `lodo_plain`; L3 `lodo_sat` vs `lodo_plain`; then h9 and txy6
against xy6. Each in sampled-action error and in flow MSE: per held task the log ratio; mean over the 41 tasks;
95% percentile bootstrap over the 16 directions (the tasks of one fold share a model; 20,000 resamples,
generator seed 0); tasks and directions won. Context from stored leave-one-task-out scores of the same seed:
`film_nl` and `plain_nl` on the same 41 tasks (how much harder an unseen direction is).

## Limits
One training seed, offline, the same 16-direction grid (an unseen direction lies 22.5 degrees from its
neighbours, 3 to 10 cm). Exploratory; marked so wherever it is written.
