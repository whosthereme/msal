# Registered test: zero-shot accuracy in the ideal regime (written 2026-11-11, before any run)

## Question
The paper should say how well MSAL's zero-shot head approximates an unseen task when conditions are ideal. The
existing results compare the zero-shot head only with weaker zero-shot heads (37.9% better than FiLM); they never
compare it with a head that was trained on the task. This test supplies that reference.

## Ideal regime (fixed here)
The development family: 41 tasks of one scripted skill that differ in the target position; the descriptor is a
function of that position, so it identifies the task; the other 40 tasks are available for training. LIBERO with
the bag-of-words descriptor is NOT in this regime (weak descriptor, heterogeneous tasks) and is not part of the test.

## What is known when this is written
All stored zero-shot scores of the modulated head (film_nl) and of the same readout without task input (plain_nl) on
the 205 folds (flow MSE). No head trained on all 41 tasks with the nonlinear readout has been scored per task, and no
sampled-action error has been computed on the development family.

## Runs (msal_ext/ideal.py; seeds 148021-148025; GPU 1 only)
- Seen head: per seed, film_nl trained with the unchanged loop and recipe (12,800 updates) on all 41 tasks.
- Zero-shot head: per fold, film_nl retrained on the other 40 tasks; its float32 SELECTION score must equal the
  stored one bit-exactly (a fold that does not is reported and marked).
- Scores on the held task's SELECTION rows (never used in training by either head):
  flow MSE (stored scoring, 8 paired draws; ROBUSTNESS rows as secondary) and sampled-chunk action error
  (10 Euler steps from tau = 1, 8 fixed noise draws identical for both heads, channel-weighted masked MSE against the
  demonstrated chunk, no clipping).

## Measures
For task t and seed s, excess e = E_zero / E_seen - 1 in each metric.
- Per task: geometric mean of E_zero / E_seen over the five seeds, minus 1.
- Pooled: per seed, geometric mean over the 41 tasks; mean and 95% t interval (df 4) of the log ratio over seeds.
- Counts: number of the 41 tasks whose per-task excess is at most 10%, and at most 25%.
Sampled-action error is primary (the closed-loop study found it to rank policies better than flow error); flow error
is reported next to it.

## Reading (fixed before the run)
- Strong: at least 31 of 41 tasks within 10% in sampled-action error. The paper may say that in this regime the
  zero-shot head faithfully approximates most unseen tasks, with the numbers.
- Moderate: not strong, but at least 21 of 41 tasks within 25%. The paper may say that it comes close to a trained
  head on most tasks, with the numbers; "faithfully" is not used.
- Otherwise: not supported. The paper says that even in this regime the zero-shot head stays clearly behind a head
  trained on the task, with the numbers.
If flow error falls in a different tier, both are reported and the claim uses the lower tier. The result is reported
in the paper whichever way it comes out, with the regime defined as above and the statement that it is offline
(no task success) and synthetic.

## Descriptive (no claim rests on these)
- Excess against the distance to the nearest training task (tasks split at the median distance).
- Share of the gap from the readout without task input (plain_nl, stored, flow MSE) to the seen head that the
  zero-shot head closes.
- Seen head's error per task, to show where a ratio is large only because the seen error is small.

## Stopping and exclusions
5 seen heads and 205 zero-shot heads; no early look changes the plan or the thresholds. A NaN fold is rerun once. If
the run cannot finish on all five seeds, the completed seeds are reported and the count of seeds is stated.

## Files
experiments/msal_ext/ideal.py, experiments/review/driver_ideal.sh (remote /home/ubuntu/work/manuscript6/ideal/),
analysis/ideal.py, results_remote/ideal/.
