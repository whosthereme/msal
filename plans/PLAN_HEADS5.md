# Sampled-chunk error of every zero-shot head, and input concatenation, at five seeds (PLAN_HEADS5) — written 2026-11-18, before any run

## Why
Table I of the paper (ten zero-shot heads, 205 leave-one-task-out folds) is scored in flow MSE only, although
the paper itself argues that sampled-chunk error tracks closed-loop behaviour better. Sampled-chunk error
exists at five seeds only for MSAL's head (`film_nl`) and the trunk-modulated head, and at one seed
(exploratory, PLAN_ACTION and PLAN) for `plain_nl`, `film` and input concatenation (`tin_nl`). The third
internal review (2026-11-18) asked for (a) Table I in sampled-chunk error at five seeds and (b) input
concatenation, the most common way to condition a policy, as a full five-seed baseline.

## Runs
Development family, leave-one-task-out over the 41 tasks, training seeds 148021-148025, 12,800 updates, the
stored training loop and arm definitions unchanged (`msal_exp.run_loto.train_arm` with the registries of
`msal_ext.run_nl`, `msal_ext.run_nlb` and `tc_arms`). Eleven arms:
`plain`, `ctx_add`, `film`, `bilinear`, `sal` (linear readout); `plain_nl`, `moe_desc`, `sal_nl`, `sal_nlb`,
`film_nl` (nonlinear readout; the ten rows of Table I); and `tin_nl` (descriptor added to the first trunk
layer's pre-activation, i.e. concatenated to the trunk input; readout as `plain_nl`).
Per fold and arm: flow MSE on SELECTION and ROBUSTNESS rows with the stored scoring arithmetic, and
sampled-chunk error on SELECTION rows with `msal_ext.ideal.action_error` (ten Euler steps, eight noise
draws, the stored rows and seeds). For heads without a `forward_descriptor(state, x, t, d)` method the same
function is called through an adapter that returns the head's zero-shot velocity field: no task input for
`plain` / `plain_nl`; the held task's row for `ctx_add`; base plus atoms weighted by the zero-shot code
z = W^T d for `sal`, `bilinear`, `sal_nl`, `sal_nlb`.
Unit = (seed, held task). Two groups of arms, run in this order over all units so that the first group is
complete first: group A `film_nl, plain_nl, film, plain, tin_nl`; group B `sal_nlb, sal_nl, moe_desc, sal,
ctx_add, bilinear`. All arms of a unit run on one machine (ws2 GPU 1, or the A800 server); group B of a unit
runs on the machine that ran its group A.

## Checks (fixed now)
- On ws2 every arm's flow MSE must equal its stored score bit for bit (`runs/n1n2`, `trunkcond/out` for
  `tin_nl` at seed 148021), and `film_nl`'s sampled-chunk error its stored value (`ideal/out`); `tin_nl`'s
  sampled-chunk error at seed 148021 its stored value. Each record carries the flags; mismatches are listed.
- On the server the differences from the stored scores are reported per arm (retraining noise across builds).
- Before the lanes start: a smoke run of all eleven arms (20 updates), then one complete group-A unit on ws2
  (seed 148021, task 0) with every flag true.

## What is known now
Flow MSE of the ten Table I heads at five seeds (Table I). Sampled-chunk error: `film_nl` at five seeds; at
seed 148021 `film_nl` vs `plain_nl` -61.7%, vs `film` -17.1%, `tin_nl` vs `film_nl` -1.6% (interval -5.7 to
+2.6; flow +4.0%). From the stored flow scores (computed 2026-11-18, post hoc): `film` vs `plain` -10.1%,
`film_nl` vs `plain_nl` -19.9%, difference -10.9% [-13.0, -8.7]; `sal_nlb` vs `plain_nl` -7.3%; `sal_nl`
and `moe_desc` worse than `plain_nl`. Nothing is known about sampled-chunk error at seeds 148022-148025 for
any arm other than `film_nl` and the trunk-modulated head, nor at any seed for `plain`, `ctx_add`,
`bilinear`, `sal`, `moe_desc`, `sal_nl`, `sal_nlb`.

## Comparisons (fixed now)
Per fold the log ratio of errors; per seed its mean over the 41 folds; mean over five seeds with a 95% t
interval (df = 4); folds won; for H1 to H5 and C1 also a percentile bootstrap over the 41 tasks (20,000
resamples, generator seed 0) on seed-averaged log ratios. Each in sampled-chunk error (new) and flow MSE.
- Table: every head against `film` and against `film_nl` (Table I's two columns), in sampled-chunk error.
- H1 `film_nl` vs `plain_nl`: the description's gain before a nonlinear readout.
- H2 `film` vs `plain`: the description's gain with a linear readout.
- H3 H1 minus H2 (difference of the paired log ratios): is the gain larger before a nonlinear readout?
- H4 `film_nl` vs `film`.
- H5 `plain_nl` vs `film`: the nonlinear readout without task input against FiLM with a linear readout.
- H6 `sal_nlb`, `sal_nl`, `moe_desc`, each vs `plain_nl`: a code that enters after the nonlinearity.
- H7 `film_nl` vs `sal_nlb`.
- H8 `sal` vs `film`; `ctx_add` vs `film`; `bilinear` vs `film`.
- C1 `tin_nl` vs `film_nl` at five seeds, and at the four seeds 148022-148025 alone (seed 148021 is known).

## Reading rules (fixed now)
A difference is called lower or higher only if its t interval excludes zero; otherwise "not distinguishable".
- "The description helps more before a nonlinear readout than with a linear readout" holds in sampled-chunk
  error if H3 is negative with the interval excluding zero.
- "The readout, not the description, accounts for the gap to FiLM" holds in sampled-chunk error if H5 is
  negative with the interval excluding zero; if H5 is positive the paper must say that in sampled-chunk error
  FiLM with a linear readout is ahead of the nonlinear readout without task input.
- "MSAL's head is the most accurate zero-shot head in sampled-chunk error" holds only if every other arm,
  `tin_nl` included, is higher than `film_nl` with the interval excluding zero; otherwise the paper names the
  arms that are not distinguishable or lower.
- C1: input concatenation is "not distinguishable from", "less accurate than" or "more accurate than" MSAL's
  head according to the five-seed interval; the four-seed reading is given next to it as the confirmation.
Every comparison is reported in both metrics whatever its sign.

## If the runs are not complete
Group A is complete before group B starts on each machine. Whatever groups are complete at five seeds are
reported; a group that is incomplete is reported as not run, not from a subset of seeds or tasks.

## Limits
One task family, offline, zero-shot only. Sampled-chunk error is itself an offline measure. Server folds
differ from ws2 folds by retraining noise (per-fold SD 5 to 10% in earlier checks), which pairing within a
machine removes for the comparisons but not for the comparison with stored Table I values.

## Amendment 1 (2026-11-18, before any lane started)
State when written: the smoke runs passed on both machines, and the anchor unit (ws2, seed 148021, task 0, group
A) had run with every flag true: flow MSE of `film_nl`, `plain_nl`, `film`, `plain`, `tin_nl` equal to the
stored values bit for bit, and the sampled-chunk error of `film_nl` and `tin_nl` equal to theirs. No other
score of this study exists.
Change: on ws2, `film_nl` is retrained only for task 0 of each seed (five anchors). For the other ws2 folds
its stored ws2 scores are used (flow MSE `runs/n1n2`, sampled-chunk error `ideal/out`), which ws2 reproduces
bit for bit. On the server every unit retrains `film_nl`, as planned.
Reason: one training takes about 45 s on ws2's GPU 1 and the GPU is the bottleneck whatever the number of
lanes, so this removes about 9% of the ws2 trainings. No arm, fold, setting, comparison or reading rule
changes.
