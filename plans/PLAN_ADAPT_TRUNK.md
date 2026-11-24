# Adaptation on the trunk-modulated head, one seed (exploratory) — written 2026-11-13 17:25, before any run

## Why
The five-seed study (PLAN_5SEED.md) found that applying MSAL's modulation to the trunk's first activation
(`tfilm_nl`) lowers zero-shot sampled-chunk error by 15.6% and flow MSE by 6.2% against MSAL's head (`film_nl`).
MSAL's adaptation stages (descriptor inversion, then the closed-form residual code) were only ever run on
`film_nl`. This run asks whether the complete method keeps the gain when its head is replaced, before any
decision to rerun it over five seeds. The user asked for one seed first.

## What is known now
All zero-shot scores of `tfilm_nl` and `film_nl` at seeds 148021-148025; the stored adaptation scores of MSAL
with `film_nl` (combination study, five seeds, flow MSE only). Nothing of adaptation on `tfilm_nl` exists.

## Runs
Seed 148021, leave-one-task-out over the 41 development tasks, k in {1, 2, 4, 8, 16} TRAIN demonstrations,
`tc_combo.py`. Two arms, run for every fold on the same machine (cross-build retraining is not bit-exact):
- `film_nl`: the stored combination fold (msal_ext/combo.py: stage 1 `film_nl`, stage 2 residual library with
  hyb.train_library, inversion with hyb.invert_descriptor, code with weighted_ls_code), calls unchanged.
- `tfilm_nl`: the same stages, seeds, offsets, steps and penalties on `tfilm_nl`. Two changes forced by the
  trunk placement, fixed now: (1) the atoms read the trunk activation computed at the descriptor in use (each
  source task's own descriptor in stage 2; the inverted descriptor when the code is solved and scored), so the
  field stays affine in the code at a fixed descriptor; (2) inversion differentiates through the trunk
  modulation (first activation cached, then modulation, rest of the trunk and readout at every step).
Scores per fold and k: flow MSE on SELECTION and ROBUSTNESS rows (stored scorer settings: 8 paired draws) for the
zero-shot head, inversion alone and MSAL (inversion + code); sampled-chunk error (msal_ext.ideal.action_error,
10 Euler steps, 8 draws) on SELECTION rows for zero-shot, and for inversion and MSAL at k = 1 and 16.
Machines: this computer runs held tasks 0-27 (21 tasks), ws3 / ws4 the other 20, each fold's two arms in one
process on one machine, after the leave-one-direction-out runs of PLAN_LODO5.md.

## Comparisons (fixed now)
Primary: A1 MSAL on `tfilm_nl` vs MSAL on `film_nl`, flow MSE (SELECTION) at k = 1 and k = 16; A2 the same in
sampled-chunk error. Secondary: residual code beyond inversion on `tfilm_nl` (MSAL vs inversion, k = 1 and 16,
flow MSE), and on `film_nl` as the replication check; inversion on `tfilm_nl` vs inversion on `film_nl`; zero-shot
`tfilm_nl` vs `film_nl` (same-machine replication of PLAN_5SEED.md at this seed). All other k: descriptive.
Reading: per task the log ratio, mean over the 41 tasks, 95% percentile bootstrap over tasks (20,000 resamples,
generator seed 0), tasks won. One seed: exploratory wherever written. Every comparison is reported.
Next step, fixed now: if A1 and A2 are not worse at both k, a five-seed run is proposed to the user; otherwise
the paper keeps MSAL on `film_nl` and reports this run as exploratory.

## Limits
Offline, one seed, development family only; no closed loop, no LIBERO.

Note on time: hashed on ws2 at 17:13:59 (ws2 clock); the header time 17:25 was typed ahead of the clock.

## Amendment 1 (before any run; only a 20-step smoke test of task 0 exists, its numbers meaningless)
Check done: on this computer, the `film_nl` arm of tc_combo.py reproduced msal_ext.combo.run_fold bit for bit
(zero-shot, inversion and combination at every k, 20 smoke steps, task 0). Distribution changed from a fixed
split to a claim queue on the ws3 / ws4 shared home: this computer claims tasks 0-27 first, then the rest from
the top; the ws3 / ws4 lanes claim tasks 29-54 first, then the rest. Each claim runs both arms of that fold in
one process on one machine, so the pairing rule is unchanged. Outputs per machine and lane.
