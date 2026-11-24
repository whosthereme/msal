# PLAN_TIN: input concatenation on LIBERO, and with MSAL's adaptation, on both task families

Written 2026-11-19 after the fourth internal review of the manuscript (the user: the experiments a referee is likely
to ask for should be done). Hashed on ws2 before any run of this plan. Changes after hashing are written below as
dated amendments before the runs they affect.

## Question
On the 41 placement tasks, concatenating the description to the trunk input predicted zero-shot at least as well as
MSAL's head (PLAN_HEADS5, C1: sampled-chunk error -4.1%). The manuscript's limitations say that this baseline was run
neither on LIBERO nor with adaptation. This plan runs both: (i) is input concatenation at least as accurate as
MSAL's head on LIBERO, zero-shot; (ii) does MSAL's adaptation (descriptor inversion, then the closed-form residual
code) work on the concatenation head, and is the complete method on it as accurate as MSAL?

## Arms
- `tin_nl` (development) / `l40_tin_nl` (LIBERO): `tc_arms.TrunkInputField`, unchanged: `P d` is added to the first
  trunk layer's pre-activation (the description concatenated to the trunk input), readout without task input
  (the `plain_nl` readout). LIBERO uses the stored word descriptor (`l40.word_descriptor`), the development
  family the stored 21-column harmonic descriptor.
- Adaptation on `tin_nl`, as PLAN_ADAPT_TRUNK did for the trunk-modulated head: stage 2 trains the residual
  library (16 atoms, the stored penalties and 12,800 steps) on the frozen head, the atoms reading the trunk
  activation computed at each task's own description; descriptor inversion is 200 Adam steps (learning rate 0.01)
  on `d` through the trunk; the residual code is the weighted least-squares solution at the inverted description.
  k in {1, 2, 4, 8, 16}; sampled-chunk error at k = 1 and 16.
- `l40_plain_nl` on LIBERO, zero-shot, retrained to obtain its sampled-chunk error (its stored runs have flow only).
- Reference, not retrained: the stored `film_nl` records of PLAN_ADAPT_TRUNK5 (`results/adapt5`, development) and of
  PLAN_L40_TRUNK (`results/l40`, LIBERO), which hold flow MSE and sampled-chunk error zero-shot and after adaptation
  for every seed and task. Amendment-style anchor: for held task index 0 of every seed (development) and task 1000
  of every seed (LIBERO), `film_nl` is retrained in the same process and must reproduce its stored ws2 score where
  the stored unit ran on ws2; the anchors are reported.

## Units, scores
Development: 41 tasks x seeds 148021-148025 = 205 units. LIBERO: 40 tasks x 5 seeds = 200 units. Training loop,
batches, 12,800 steps and scoring rows as stored (SELECTION rows for the development family, validation rows for
LIBERO; ROBUSTNESS / test rows as second check); flow MSE with eight paired noise draws; sampled-chunk error with ten
Euler steps and the stored function `ideal.action_error`. All units run on ws2 GPU 1 (never GPU 0) through
`gpu_run`; if the shared A800 server is reachable it may take units, and each record names its machine.

## Readings (fixed now)
Per-seed mean of the paired log ratio over tasks, mean over the five seeds as a percentage change, 95% t interval
(4 degrees of freedom); "lower" or "higher" only if the interval excludes zero, otherwise "not distinguishable".
Pairs use the stored `film_nl` record of the same seed and task; the subset of pairs whose stored record ran on the
same machine is reported next to the full set, and a reading is called only if both agree in sign or the same-machine
subset alone excludes zero.
- D1 (development, zero-shot): `tin_nl` vs `film_nl`, sampled chunk; flow. A replication of C1, not a new claim.
- D2 (development, complete method): `tin_nl` + inversion + residual code vs MSAL, at k = 1 and k = 16; sampled chunk
  (primary) and flow.
- D3 (development): on `tin_nl`, residual code beyond inversion, flow, k = 1 and 16 (is the library still useful?).
- L1 (LIBERO, zero-shot): `l40_tin_nl` vs `l40_film_nl`; sampled chunk (primary) and flow.
- L2 (LIBERO, what the description adds): `l40_tin_nl` vs `l40_plain_nl` and `l40_film_nl` vs `l40_plain_nl`, sampled
  chunk and flow. The manuscript's statement that the word description "did not help distinguishably" is a flow
  reading; this gives the sampled-chunk reading for both ways of entering before the nonlinear readout.
- L3 (LIBERO, complete method): `l40_tin_nl` adapted vs MSAL at k = 1 and 16; sampled chunk (primary) and flow.
- L4 (LIBERO): on `l40_tin_nl`, residual code beyond inversion, flow, k = 1 and 16.
Descriptive: every arm's mean errors; descriptor shift by inversion; folds won; ROBUSTNESS / test rows.

## Consequences fixed now
The manuscript reports D2, L1, L2 and L3 whatever they show, in the results and in the limitations (which now say
these were not run). If input concatenation with the same adaptation is lower than MSAL in sampled-chunk error on
both families, the paper says that MSAL's head has no accuracy advantage over concatenation and keeps MSAL only as
the method on which the conditions were studied. If it is higher on both, the paper says so and no longer calls
concatenation "at least as good" without qualification. Mixed or indistinguishable outcomes are reported as such.

## Cost
About 405 full units (4 to 5 minutes each with three lanes in parallel) and 200 zero-shot units (about 1.5 minutes
each): roughly 11 to 13 hours of ws2 GPU 1 with three lanes.
