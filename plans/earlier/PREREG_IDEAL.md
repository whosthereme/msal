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

## Execution and outcome (added 2026-11-11 after the run; nothing above was changed)
Launched 10:25 workstation time, done 13:16. Smoke passed first time. 5 seen heads, 205 of 205 zero-shot heads, every
one equal to its stored float32 score bit-exactly, no NaN, no rerun. Uploaded copy of this file before the run:
sha256 572e1222...6bdffe (results_remote/ideal/prereg). Analysis: analysis/ideal.py -> analysis/ideal.json.
Checked after the run: the splits are by episode (16 TRAIN, 4 SELECTION, 4 ROBUSTNESS episodes per task, no shared
episode), so the seen head has not seen the scored episodes.
- Sampled-action error (primary): zero-shot 0.0426, seen head 0.0071. Pooled excess +430% (interval +411% to +450%).
  Tasks within 10%: 0 of 41; within 25%: 0 of 41. Per task: minimum +194%, median +378%, maximum +2442%.
- Flow MSE: zero-shot 0.220, seen 0.057. Pooled excess +214% (+190% to +240%); 0 of 41 within 10% or 25%; per task
  minimum +80%, median +166%. ROBUSTNESS rows: +210% (+199% to +221%).
- Descriptive: nearer half of the tasks (27, ties at the median distance) median excess +367% in action error, farther
  half (14) +414%: distance to the nearest training task matters little. Of the flow-error gap from the readout
  without task input (0.261) to the seen head, the zero-shot head closes 0.20.
Reading by the registered rule: NOT SUPPORTED in both metrics. Even in the ideal regime as defined here, the
zero-shot head's error is about three times (flow) to five times (sampled actions) that of a head trained on the
task, on every one of the 41 tasks. For scale (stored results, flow MSE, not part of this registration): MSAL with 16
demonstrations lowers zero-shot error by 26.6% (to about 0.16) and readout fine-tuning at the development-selected
setting by about 74% (to roughly 0.06, the seen head's level of 0.057).

## Exploratory follow-up: coverage curve (added 2026-11-11, after the outcome; no new run)
analysis/ideal_coverage.py joins the seen heads above with the zero-shot scores of the registered coverage study
(PREREG_COV.md, same float32 SELECTION score, 205 folds per level). Zero-shot flow MSE against the seen head (0.057):
n = 8: 0.880 (+1380%); 16: 0.581 (+840%); 24: 0.401 (+515%); 32: 0.291 (+327%); 40: 0.220 (+214%).
The log ratio falls by 0.45, 0.43, 0.36, 0.31 per eight added tasks: steep and still falling at 40, slowing slightly.
No task is within 25% at any level (best task +80% at n = 40, +121% at n = 32). Extrapolation (not a result): a
line in n reaches +25% at about 58 tasks, a line in log2 n at about 116; the truth is probably between. Flow error
only; sampled-action error at n < 40 was not computed. Reading: in this family zero-shot accuracy is limited by the
number of training tasks, not saturated.
