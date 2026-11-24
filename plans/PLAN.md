# Trunk-conditioning control (exploratory) — written 2026-11-12, before any run

## Question
Every head in Table I of the MSAL draft receives the task description only after the shared trunk output h.
Does zero-shot accuracy on the development family change when the same description enters the trunk instead,
with the same nonlinear readout, the same descriptor and the same training recipe?

## Arms (all new; nothing tuned; initialisation as the existing FiLM maps, N(0, 0.02^2), bias-free)
The trunk is Linear(832, 512) - SiLU - Linear(512, 512) - SiLU - LayerNorm. The readout of every arm is the
existing one, a linear map plus the 1,472-unit SiLU network.
- `tin_nl`   description concatenated to the trunk input: P d (21 -> 512) is added to the first layer's
      pre-activation. Readout without task input (as `plain_nl`). 10,752 task-conditioned weights.
- `tfilm_nl` FiLM on the first trunk activation, u <- u * (1 + G d) + D d. Readout without task input.
      21,504 task-conditioned weights, the same number and the same operator as `film_nl`; only the
      position differs.
- `tboth_nl` `film_nl` unchanged (FiLM on h, then the readout) plus the trunk FiLM of `tfilm_nl`. 43,008.

## Data, folds, recipe (identical to the stored runs)
Development family, language-blind feature block, descriptor `saturated` (21 columns), leave-one-task-out over
the 41 source tasks, training seed 148021 only, 12,800 AdamW updates through `msal_exp.run_loto.train_arm`
unchanged (same batches and noise draws as every stored arm; only the weight initialisation stream differs).
No sealed task or row is read. No held-task TRAIN row is used.

## Scores (per fold, on the held task's SELECTION rows)
- flow MSE, the stored scorer (`run_loto.score_forward`, 8 paired draws); ROBUSTNESS rows as a second check;
- sampled-chunk action error (`msal_ext.ideal.action_error`, 10 Euler steps, 8 fixed draws), the same
  function, rows and seeds as the stored zero-shot `film_nl` scores of PREREG_IDEAL.

## Anchor (must pass before the arms run)
`film_nl`, seed 148021, held task 0, through the new runner: flow MSE must equal the stored 0.1416842844337225
and action error the stored 0.03402892626759734, both bit-exactly. Otherwise the study stops.
Self-check at start of every run: with its new conditioning weights set to zero, each arm must reproduce the
forward pass of the class it extends on random inputs (max abs difference <= 1e-6).

## Comparisons (fixed now)
Reference scores are the stored ones of the same seed and folds: `film_nl` and `plain_nl` (runs/n1n2), the
zero-shot action error and the head trained on all 41 tasks (ideal/out).
- C1 `tfilm_nl` vs `film_nl`   (position, capacity-matched)
- C2 `tin_nl`   vs `film_nl`   (the conventional input-concatenation baseline)
- C3 `tboth_nl` vs `film_nl`   (does conditioning the trunk add to MSAL's head?)
Each in flow MSE and in sampled-action error. Descriptive: each arm vs `plain_nl` (flow MSE), the share of the
gap from `plain_nl` to the trained head that each arm closes, folds won, results with and without the four
outlier tasks (1, 2, 31, 43) identified before this run.

## Reading rule
Per task, log ratio of the two scores; mean over the 41 tasks as a percentage change; 95% percentile bootstrap
over tasks (20,000 resamples, generator seed 0). "Better" or "worse" only if the interval excludes zero,
otherwise "not distinguishable". Every comparison is reported whatever the outcome.

## What this can and cannot show
One training seed. The draft's unit of replication is the training seed (five seeds, t interval), so a
task-level interval at one seed does not replace that study; it says whether a five-seed run is worth doing.
Offline only. Zero-shot only: descriptor inversion and the residual library are not rerun on these heads.

## Budget and rules
3 arms x 41 folds, about 2.5 min each: roughly 5 GPU-hours, about 2 hours wall with the three arms side by side.
GPU 1 only, through `upgrades/slots.sh` (`gpu_run train`); GPU 0 is never used. Writes only inside
`/home/ubuntu/work/manuscript6/trunkcond/`. The repo, runs and every other folder are read-only.
