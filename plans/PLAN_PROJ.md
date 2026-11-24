# Projection control: what the in-span part of a held descriptor does (PLAN_PROJ) — written 2026-11-18, before any run

## Why
The paper states its second condition as "the descriptor of a new task must lie in the span of the training
descriptors" and reads the span share as a necessary condition. Proposition 2 proves less: training fixes the
head's response only to the in-span part P_S d* of a held descriptor; the rest passes through weights that
training did not determine. The third internal review (2026-11-18) pointed out that the reported experiments
do not separate two readings of the harmonic descriptor's failure on held directions (span share 0.19):
(i) the in-span part carries too little information, or (ii) the out-of-span part harms through undetermined
weights. The same holds for the identifier test (PLAN_SPAN_PRED), where raising alpha also scales the input of
weights that never received a gradient. This control evaluates the same heads on P_S d* instead of d*.

## Runs
Leave-one-direction-out folds of PLAN_LODO5 (16 directions, 41 tasks, seeds 148021-148025), MSAL's zero-shot
head (FiLM on h, nonlinear readout), stored training loop unchanged, 12,800 updates. Two descriptors:
- `lodo_sat`: the 21-column harmonic descriptor (as in PLAN_LODO5);
- `lodo_xy6id2`: six coordinate columns plus a direction one-hot of weight alpha = 2 (as in PLAN_SPAN_PRED).
Per (seed, direction) one process trains both heads. Nothing about training changes. Each held task is scored
twice with the same model, rows and noise draws:
- raw: the held descriptor d* (the stored scoring; repeats PLAN_LODO5 / PLAN_SPAN_PRED);
- projected: P_S d*, with P_S the orthogonal projector onto the row span of the fold's training descriptor
  matrix (singular value decomposition in double precision, singular values above 1e-9 of the largest kept).
Metrics: flow MSE on SELECTION and ROBUSTNESS rows (stored scorer arithmetic) and sampled-chunk error on
SELECTION rows (`msal_ext.ideal.action_error`), as in PLAN_LODO5. All runs on ws2 (GPU 1), four lanes.

## What is known now
All raw scores of both heads at the five seeds, and the scores of the head without description and of the
coordinate head on the same folds (PLAN_LODO5, PLAN_SPAN_PRED). Harmonic against none: sampled chunk +1.3%
(n.d.), flow +113%. Identifier alpha = 2 against coordinates: sampled chunk +25.0%, flow +27.5%. No head has
ever been evaluated on a projected descriptor.

## Comparisons (fixed now)
Per held task the log ratio of errors; per seed its mean over the 41 tasks; mean over the five seeds with a 95%
t interval (df = 4); tasks and directions on which the first arm is lower; for J1 and J4 also a percentile
bootstrap over the 16 directions (20,000 resamples, generator seed 0) on the seed-averaged log ratios.
- J1 harmonic projected vs harmonic raw. Primary metric: flow MSE (where the harm was seen); also sampled chunk.
- J2 harmonic projected vs no description (stored `lodo_plain`, same seeds). Both metrics.
- J3 harmonic projected vs coordinates (stored `lodo_xy6`, same seeds). Both metrics.
- J4 identifier projected vs identifier raw. Primary metric: sampled-chunk error (primary of PLAN_SPAN_PRED);
  also flow MSE.
- J5 identifier projected vs coordinates alone (stored `lodo_xy6` of PLAN_SPAN_PRED, same seeds). Both metrics.
J2, J3 and J5 pair new scores with stored scores of other processes and, for part of the folds, other machines
(per-fold retraining noise 5 to 10%, no bias in earlier checks); they are reported as such.
Check: the raw scores of this run against the stored ones per fold (bit-exact where the stored fold ran on
ws2; reported).

## Predictions and reading rules (fixed now)
- Prediction: J1 negative in flow MSE (the projected harmonic descriptor has lower flow error than the raw
  one) and J4 negative in sampled-chunk error.
- "The out-of-span part harms through weights that training did not determine" is supported for a descriptor
  if its primary comparison (J1 flow, J4 sampled chunk) is negative with the t interval excluding zero; not
  supported if the interval includes zero; contradicted if it is positive with the interval excluding zero.
- "The in-span part of the harmonic descriptor carries usable information about a held direction" is supported
  if J2 is negative in sampled-chunk error with the interval excluding zero; otherwise not shown.
- J3 and J5 are reported without a reading rule: they say how far the in-span part alone is from a descriptor
  that lies in the span and resembles the training descriptors.
Whatever the outcome, all five comparisons are reported in both metrics.

## Limits
One task family, one skill, offline. The projected descriptor is a linear combination of training descriptors
but need not resemble any of them, so a poor projected score does not show that the in-span part is useless
for another head. Evaluation only: no head is trained to expect projected descriptors.
