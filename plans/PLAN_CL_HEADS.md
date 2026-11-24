# PLAN_CL_HEADS: the two conditions in closed loop (cluttered coordinate placement family)

Written 2026-11-19 after the fourth internal review of the manuscript: neither condition has closed-loop evidence
(PLAN_CL_COORD compared coordinates, a nine-column descriptor and no description, all on MSAL's head and, for the
two descriptors, all inside the span). Hashed on ws2 before any head of this plan is trained or rolled out. Changes
after hashing are written below as dated amendments before the runs they affect.

## Question
In closed loop, at targets in directions never trained: (1) does it matter where the description enters (condition
1), and (2) does a description that leaves the held target largely outside the span of the training descriptions
lose what a description inside the span gains (condition 2)?

## Family, folds, scenes (unchanged from PLAN_CL_COORD after its Amendment 1)
40 targets, four direction folds, seeds 148021-148023, the stored demonstrations (`libcoord/shards`), rollout
episodes 100-109 per held target and seed, the same scenes and the same flow-noise seeds (seed, task, episode), so
every new policy is paired episode by episode with the stored policies C0 (coordinates on MSAL's head) and N0 (no
description). Success as before (3 cm, resting, gripper open), 300 steps at most, 8 actions executed per chunk.

## New heads (zero-shot only; the codebase's training loop and 12,800 steps, unchanged)
All with the coordinate descriptor (six columns, span share 1 at every held target) unless stated:
- F: FiLM followed by a linear readout (`FiLMField`): description before a linear readout.
- M: softmax mixture of 16 nonlinear experts gated by the description (`DescriptorMoE`): description as weights
  after the nonlinearity.
- T: description concatenated to the trunk input (`TrunkInputField`): before the nonlinearity, like MSAL's head.
- G: MSAL's head with the full harmonic descriptor of this layout, 20 columns (constant, cos m theta for m = 1..8,
  sin m theta for m = 1..7, r, r^2, r cos theta, r sin theta; the development family's r^3 is omitted because with
  three rings it equals r). Span share of a held target, computed now from descriptors alone: mean 0.39, minimum
  0.04, maximum 0.96 (coordinates: 1 everywhere; training descriptors have rank 16 of 20).
Each artifact stores a probe; the rollout process rebuilds the head from its state dict and stops if it disagrees
with the probe by more than 1e-3. Offline flow MSE and sampled-chunk error of every head on its held targets are
recorded before any rollout.

## Rollouts
Policies F0, M0, T0, G0 on every held target: 40 targets x 10 episodes x 3 seeds x 4 policies = 4,800 episodes, ws2
GPU 1 through `gpu_run` rollout slots.

## Readings (fixed now)
Unit: the held target. Paired differences of success in percentage points against the stored policy named; two-level
bootstrap (targets, then episodes within target, seeds pooled), 10,000 resamples, generator seed [20261119, index];
"better" or "worse" only if the 95% interval excludes 0.
Condition 1:
- H1: C0 (MSAL's head) vs F0 (linear readout).
- H2: C0 vs M0 (mixture of experts).
- H3: T0 (input concatenation) vs C0.
- H4: each of F0, M0, T0 vs N0 (does the description carry in closed loop with that head at all?).
Condition 2:
- H5: G0 (harmonic, mostly outside the span) vs C0 (coordinates, inside), same head. Prediction from the span
  analysis and the offline result of the development family: G0 is worse than C0.
- H6: G0 vs N0. Prediction: not better.
Secondary: the per-target median final distance for H1 to H6 with the same bootstrap; success of G0 against the
held target's span share (Spearman over the 40 targets, bootstrap over targets).
Power, stated now: C0 succeeded in 10.9% and N0 in 3.2% of 1,200 episodes each, and the interval of their difference
was about +-3.4 points; differences between heads smaller than about 4 points will read "not distinguishable", and
the paper will then say that closed loop did not separate the heads, not that they are equal.

## Consequences fixed now
The manuscript reports H1 to H6 whatever they show. If H5 does not read "worse", the paper states that the second
condition was not confirmed in closed loop. If H1 and H2 read "not distinguishable", the paper states that the
offline ordering of heads was not resolved in closed loop at this success level. If T0 is better than C0, the paper
says so next to the offline result.

## Cost
48 head trainings of 4 to 5 minutes (about 1.5 hours with three lanes) and 4,800 rollout episodes (about 4 to 5
hours with three rollout lanes on ws2 GPU 1).
