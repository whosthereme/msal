# PLAN_LODO_ADAPT: MSAL's adaptation on directions never trained, harmonic vs coordinate descriptor

Written 2026-11-14 at the user's request ("跑坐标描述子接适配的实验"). Hashed on ws2 before the first run.

## What is known
Zero-shot, leave-one-direction-out, five seeds (PLAN_LODO.md + PLAN_LODO5.md): with the draft's 21-column
descriptor (`lodo_sat`) the description gave no gain over none in sampled-chunk error (+1.3%, n.d.); the six-column
coordinate descriptor (`lodo_xy6`) lowered it by 62.6% against the harmonic one (41/41 tasks). Only 19% of a held
harmonic descriptor lies in the training span, all of a coordinate one (Proposition 5). Adaptation was never run
with these folds or descriptors. Descriptor inversion moves the descriptor by gradient descent on the demonstrations
and may move a harmonic descriptor into the trained span, so the zero-shot gap may shrink after adaptation.

## Question
After MSAL's adaptation (descriptor inversion + residual code), is the coordinate descriptor still better than the
harmonic one on directions never trained, and does adaptation still help each of them?

## Runs
Arms `lodo_sat` and `lodo_xy6` exactly as in PLAN_LODO.md (both MSAL's head: FiLM on h, nonlinear readout;
registered by `tc_lodo.register`, unchanged). 16 held directions x seeds 148021-148025 = 80 (seed, direction)
units, both arms per unit in one process on one machine. Per unit and arm: stage 1 (run_loto.train_arm, 12,800
updates) and the residual library (hyb.train_library) trained once on the other directions' tasks; then for each
held task of the direction the stages of the development adaptation study (tc_combo.run_fold, stored hyb
functions, unchanged settings): zero-shot flow MSE (SELECTION, ROBUSTNESS, 8 noise draws), inversion (200 Adam
steps, lr 1e-2) from the arm's own held descriptor and closed-form residual code at k = 1, 2, 4, 8, 16
demonstrations (the held task's first k TRAIN episodes), sampled-chunk error on SELECTION rows (10 Euler steps,
8 draws) zero-shot and at k = 1, 16. 41 held tasks per seed and arm. Runner: local_run/tc_lodo_adapt.py.

Machines: ws2 GPU 1 (two lanes in one gpu_run), ws3 / ws4, this computer; claims of (seed, direction) units in
claims_lodoa on the ws3 / ws4 home. Anchor: unit (148021, direction 0) runs first on this computer, which ran the
stored PLAN_LODO.md seed 148021; its zero-shot flow MSE and sampled-chunk error must equal the stored records bit
for bit (equality only is checked; no adaptation number is looked at before all runs end). Every unit is
same-machine by construction.

## Readings (draft's rule)
Per seed, the mean over the 41 held tasks of the paired log ratio; mean over seeds with t interval (df 4);
"lower/higher" only if the interval excludes 0. Second view: per task the mean over seeds, 95% percentile bootstrap
over the 16 directions (20,000 resamples, generator seed 0).

Primary
- C1 complete method, xy6 vs sat: sampled-chunk error at k = 1 and 16.
- C2 complete method, xy6 vs sat: flow MSE (SELECTION) at k = 1 and 16.
Secondary
- C3 complete method vs its own zero-shot head, each arm: sampled chunk and flow at k = 1, 16.
- C4 residual code beyond inversion, each arm: flow at k = 1, 4, 16; sampled chunk at k = 1, 16.
- C5 inversion alone, xy6 vs sat: flow and sampled chunk at k = 1, 16.
- C6 zero-shot, xy6 vs sat (replication of PLAN_LODO5.md within this study).
Descriptive: k = 2, 4, 8; ROBUSTNESS rows; descriptor shift; per-seed values.

Consequence fixed now: the paper states that the coordinate descriptor's advantage on unseen directions holds
after adaptation only if C1 is lower at both k; if C1 is not distinguishable at either k it states that adaptation
closes the gap; if higher, that adaptation reverses it. C3 decides whether MSAL's adaptation helps on unseen
directions with each descriptor. Whatever comes out goes into the supplement, and into the main paper where it
changes a claim.

## Limits
Offline, one simulated skill and its 16-direction grid, MSAL's head only (the trunk-modulated head is not run).
