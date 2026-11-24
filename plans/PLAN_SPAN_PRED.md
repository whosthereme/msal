# PLAN_SPAN_PRED: does the span share, computed from descriptions alone, predict zero-shot generalization?

Written 2026-11-16 after the user's go-ahead (reply to "Register and run the per-fold span-share analysis?":
"ws4恢复了。24098的。登记并可以开始跑。"). Hashed on ws2 before any score of Part B exists and before any Part A
correlation is computed. ws4: port 24098 closed the connection before key exchange; port PORT (its earlier port)
answers with the same ed25519 host key as ws3 (the key from the user's known_hosts), so ws4 is used through PORT.

## Why
The rewritten paper claims that whether a frozen policy generalizes zero-shot to a new task can be read off before
deployment from the share iota of the task's descriptor that lies in the span of the training descriptors
(Proposition 5). So far this rests on three contrasts (harmonic vs coordinates on untrained directions; LIBERO
instructions with and without unseen words). This study tests it per task, on stored scores (Part A), and
prospectively, with predictions fixed here before training (Part B).

## Span share (computed already, from descriptors only; trunkcond/compute_iota.py -> results/span/iota.json)
iota(d*, S) = ||P_S d*||^2 / ||d*||^2, S = row span of the fold's training descriptors. Reproduces the paper's 0.19
for the harmonic descriptor on leave-one-direction-out folds (mean 0.187, per task 0.076 to 0.344).
LIBERO bag of words, leave-one-task-out: mean 0.852, min 0.514; iota = 1 on 8 of 40 tasks; 17 tasks have an unseen
word, and 15 more have iota < 1 with all words seen (a combination outside the span).
Part B descriptors (coordinates xy6 plus alpha times a one-hot of the target direction, 16 columns; the held
direction's column is zero for every training task, so it is outside the span): mean iota 1 (alpha 0), 0.898 (0.5),
0.681 (1), 0.327 (2).

## Part A: stored scores (pre-specified now; the scores were seen before only in aggregate and, for LIBERO, split by
unseen words)
A1 LIBERO, 40 tasks: per task, the mean over seeds 148021-148025 of log(zero-shot flow MSE of MSAL's head with the
word descriptor / the same nonlinear readout without task input) on SELECTION rows (stored fields
zero_shot_float32.film_nl and .plain_nl, 04_data/raw/l40). Statistic: Spearman correlation between iota and that
mean log ratio. Prediction: negative (more of the description in the span, larger gain).
A2 Development, leave-one-direction-out, harmonic descriptor, 41 tasks: per task, the mean over the five seeds of
log(sampled-chunk error of lodo_sat / lodo_plain) (stored records results/local/out_lodo, results/final_nodes/out_lodo;
each seed's arms ran in one build family). Spearman with iota. Prediction: negative.
Interval: bootstrap over tasks (20,000 resamples, generator seed 0), 95% percentile. "Supported" only if the interval
lies below 0; "contradicted" if above 0; otherwise "not distinguishable". Secondary: robustness rows; flow MSE for A2;
A1 restricted to the 23 tasks without an unseen word (does iota predict beyond unseen words?).

## Part B: prospective (new training)
Leave-one-direction-out folds (16 directions, 41 tasks), seeds 148021-148025, MSAL's head (FiLM on h, nonlinear
readout) with four descriptors per (seed, direction) unit, all in one process on one machine:
lodo_xy6 (alpha 0), lodo_xy6id05, lodo_xy6id1, lodo_xy6id2 (xy6 + alpha * one-hot direction). Stage 1 only (12,800
updates), scored as tc_lodo.py: flow MSE (SELECTION, ROBUSTNESS) and sampled-chunk error (SELECTION) per held task.
Runner local_run/tc_span_pred.py (imports tc_lodo.py unchanged and registers the three arms).
Why identifiers: appending a task identifier to a description is a common way to condition multi-task policies; by
Proposition 5 the held direction's identifier column receives no learned weight, and by the span share the three
descriptors are predicted, before training, to be ordered alpha 0 < 0.5 < 1 < 2 in held-task error.
Primary B1: zero-shot sampled-chunk error, lodo_xy6id2 vs lodo_xy6 (draft's rule: per seed the mean over the 41
tasks of the paired log ratio, t interval over seeds, df 4). Prediction: higher. "Confirmed" if the interval lies
above 0; "contradicted" if below 0; otherwise "not distinguishable".
Secondary: B2 the same for alpha 1 and 0.5; B3 monotone order of the four arms' mean log ratios (descriptive);
flow MSE; per-direction view (bootstrap over the 16 directions); per-task Spearman of iota with log ratio across the
three identifier arms pooled.
Anchor: the lodo_xy6 arm of one unit per build family must equal the stored tc_lodo.py record of the same seed and
direction bit for bit (ws3/ws4: seed 148024 or 148025, results/final_nodes/out_lodo; this computer: seeds
148021-148023, results/local/out_lodo). ws2 has no stored zero-shot leave-one-direction-out record and is not anchored.
Machines: ws2 GPU 1 (two lanes, gpu_run), ws3 (four lanes), ws4 (four lanes, port PORT), this computer (one lane);
claims of (seed, direction) units in claims_span on the shared home. 80 units x 4 arms.

## Figure (if the readings allow)
Zero-shot gain against the same head without description, against iota: Part A tasks and Part B arms on one plot.

## Limits
One skill and its 16 directions for Part A2 and Part B; LIBERO flow MSE only (no stored zero-shot sampled-chunk
error); identifiers are a constructed variation of the descriptor, not a different task family. iota is a necessary
condition: it can predict where training cannot inform a task, not how accurate a reachable task will be.
