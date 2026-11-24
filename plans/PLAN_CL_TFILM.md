# PLAN_CL_TFILM: trunk modulation against input concatenation in closed loop (cluttered coordinate placement family)

Written 2026-11-20 (evening). Reason: in the closed-loop study PLAN_CL_HEADS input concatenation (T0) placed more objects than
MSAL's head (C0), 16.8% against 10.9%, and it is also ahead offline on the placement tasks by about 4%. The author
asked whether any conditioning of ours does better than this baseline. One candidate exists and was fixed before this
plan: FiLM on the first trunk activation with the readout without task input (`tc_arms.TrunkFiLMField`, "trunk
modulation"). Offline on the development family it is below MSAL's head by 15.6% in sampled-chunk error (five seeds,
registered) and, read post hoc today from stored records, below input concatenation by 12.0% (-14.1 to -9.9; flow
-5.5%) (`results/tfilm_vs_tin.txt`). It has never been run in the cluttered family. Known against it: on LIBERO40 it
was 3.6% worse than MSAL's head offline and 5.5 points below it in closed loop at k = 1 (PLAN_L40_TRUNK,
PLAN_CL_TRUNK). Hashed on ws2 before any head of this plan is trained or rolled out; changes after hashing are
written below as dated amendments before the runs they affect.

## Question
At targets in directions never trained, zero-shot from coordinates: does modulation inside the trunk place more
objects than concatenation at the trunk input?

## Family, folds, scenes (unchanged from PLAN_CL_COORD after its Amendment 1 and PLAN_CL_HEADS)
40 targets, four direction folds, seeds 148021-148023, the stored demonstrations (`libcoord/shards`), rollout
episodes 100-109 per held target and seed, the same scenes and flow-noise seeds (seed, task, episode), 300 steps at
most, 8 actions executed per chunk, success as before (3 cm, resting, gripper open). Every new episode is therefore
paired with the stored episodes of T0, C0, N0, F0 and M0.

## New head (zero-shot only; the codebase's training loop and 12,800 steps, unchanged)
K: `tc_arms.TrunkFiLMField`'s forward pass with the six-column coordinate descriptor: the first trunk activation is
multiplied by 1 + Gamma d and shifted by Delta d (Gamma, Delta bias-free, N(0, 0.02^2), 2 x 512 x 6 weights), then the
second trunk layer and the readout without task input. Twelve heads (four folds, three seeds), trained on ws2 GPU 1
through `gpu_run train` (the stored T, F, M, G heads were trained on the A800 server and C, N on ws2 and the A800
server; training machine is not part of the pairing). Each artifact stores a probe; the rollout process rebuilds the
head from its state dict and stops if it disagrees with the probe by more than 1e-3. Offline flow MSE and
sampled-chunk error of every head on its held targets are recorded before any rollout.

## Rollouts
Policy K0 on every held target: 40 targets x 10 episodes x 3 seeds = 1,200 episodes, ws2 GPU 1 through `gpu_run`
rollout slots, script `lc_rollout3.py` (the loop of `lc_rollout2.py` with the rebuild of this head).

## Readings (fixed now)
Unit: the held target. Paired differences of success in percentage points; two-level bootstrap (targets, then
(seed, episode) units within target), 10,000 resamples, generator seed [20261121, index]; "better" or "worse" only if
the 95% interval excludes 0.
- H7 (primary): K0 vs T0 (trunk modulation against input concatenation).
- H8: K0 vs C0 (against MSAL's head).
- H9: K0 vs N0 (does the description carry with this head at all?).
Secondary: per-target median final distance for H7 to H9 with the same bootstrap; offline sampled-chunk error and
flow MSE of K against the stored T heads on the same held targets (per-seed mean of paired log ratios, t interval
over the three seeds).
Prediction, stated now: from the offline result, K0 is at least as good as T0. Power: T0 succeeded in 16.8% and C0 in
10.9% of 1,200 episodes and the interval of their difference was +1.8 to +9.8 points; a difference between K0 and T0
smaller than about 4 points will read "not distinguishable".

## Consequences fixed now
The result is reported whatever it shows. Only if H7 reads "better" may the manuscript say that a conditioning of
ours led input concatenation in closed loop, and it must then also say that trunk modulation was worse than MSAL's
head on LIBERO and that it was chosen for this test after concatenation was seen to lead. If H7 reads "not
distinguishable" or "worse", the manuscript keeps its present statement that input concatenation led in closed loop,
and adds this head to the heads it did not separate from, or was better than.

## Cost
12 head trainings of 4 to 5 minutes (about 1 hour, one training slot) and 1,200 rollout episodes (about 1 to 1.5
hours with three rollout lanes on ws2 GPU 1).
