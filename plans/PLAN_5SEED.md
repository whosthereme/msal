# Trunk-conditioning control, five-seed extension — written 2026-11-13, before any run of seeds 148022-148025

## What is known when this is written
Seed 148021 of PLAN.md is complete (123 folds). Against the stored `film_nl`, with a bootstrap over the 41 tasks:
`tfilm_nl` sampled-action error -14.8% [-19.1, -10.5], flow MSE -3.9% [-8.4, +0.9]; `tboth_nl` no better than
`tfilm_nl`; `tin_nl` not better than `film_nl`. The decision to run more seeds, and to run only `tfilm_nl`, was
made after seeing these numbers. No score of seeds 148022-148025 for any new arm exists.

## Runs
`tfilm_nl` (FiLM on the first trunk activation, readout without task input; `tc_arms.py` and `tc_run.py`
unchanged, same hashes as the seed-148021 run) at training seeds 148022, 148023, 148024, 148025: leave-one-task-out
over the 41 development tasks, 12,800 updates, the stored scorers. 164 folds.
Anchor, per seed, before its folds: `film_nl` on held task 0 through the same runner must reproduce the stored
flow MSE (runs/n1n2) and the stored sampled-action error (ideal/out) bit-exactly. If any of the four anchors
fails, no fold is run.

## Reading rule (the draft's own, Section V-C)
For each seed, the mean over the 41 folds of the paired log ratio of the two scores; the mean over seeds as a
percentage change with a 95% t interval over seeds; folds won. "Better" or "worse" only if the interval excludes
zero.
- Primary P1: `tfilm_nl` vs stored `film_nl`, sampled-chunk action error on SELECTION rows, all five seeds (df 4).
- Primary P2: the same in flow MSE on SELECTION rows, all five seeds (df 4).
- Confirmation without the seed that prompted this extension: P1 and P2 on seeds 148022-148025 only (df 3).
  If the five-seed and the four-seed readings differ, the four-seed reading is the one to report as confirmed.
- Secondary: flow MSE on ROBUSTNESS rows; `tfilm_nl` vs stored `plain_nl` (flow MSE); `tfilm_nl` and `film_nl`
  against the head trained on all 41 tasks (action error and flow MSE); the share of the flow-MSE gap from
  `plain_nl` to the trained head that each closes, per seed.
- Second view, because the seed interval does not cover task-to-task variation: per task, the mean log ratio over
  the five seeds; 95% percentile bootstrap over the 41 tasks (20,000 resamples, generator seed 0).
- Descriptive: per-seed values; the comparison without the four outlier tasks (1, 2, 31, 43).
Every comparison is reported whatever the outcome.

## What this can and cannot show
Offline and zero-shot only. Descriptor inversion and the residual library are not rerun on this head, and no
closed-loop episode is run, so the result does not say whether the complete method improves.

## Budget and rules
164 folds, the four seeds side by side on GPU 1: about 2.5 hours. GPU 1 only, through `upgrades/slots.sh`
(`gpu_run train`); GPU 0 is never used. Writes only inside `/home/ubuntu/work/manuscript6/trunkcond/`
(the shared disk is 99% full; this run writes well under 1 MB).

## Amendment 1 — 2026-11-13 00:40, change of machine (written before any ws4 score exists; no score of seeds 148022-148025 from ws2 has been looked at, only fold counts)
The user asked for the run to move from ws2 to ws3 / ws4. ws3 refuses SSH connections. ws4 is one shared
A800 80 GB (other jobs hold about 68 GB and 100% utilisation), PyTorch 2.12.1+cu126 (ws2: +cu130), driver 535
(ws2: 580), NumPy 2.2.6 on both. The model code, runner, data file (same SHA-256, verified by the loader) and
recipe are unchanged; `ws4/tc_run_ws4.py` only redirects ws2's two absolute paths for the stored scores.

- Folds finished on ws2 before the switch are kept and read exactly as planned (stored `film_nl` of the same
  machine, anchors bit-exact). The ws2 processes are stopped only after ws4 has trained and scored a fold.
- The bit-exact anchor rule cannot hold across machines, so on ws4 it is replaced by a calibration, fixed here:
  `film_nl` is retrained on ws4 for 24 folds, six held tasks at each of the four seeds: task 0 and the first
  five tasks of `numpy.random.default_rng(20261113).permutation(source_task_ids)` other than 0. For flow MSE and
  for sampled-action error, the mean over the 24 folds of log(ws4 score / stored ws2 score) is reported with a
  95% t interval (df 23), together with the largest single-fold difference.
- Rule: if both intervals lie inside [-2%, +2%], ws4 `tfilm_nl` folds are compared with the stored `film_nl`
  scores of the same fold and seed, and the calibration is reported next to the result. Otherwise `film_nl` is
  retrained on ws4 for every fold whose `tfilm_nl` score comes from ws4, and those folds are compared with the
  ws4 `film_nl`.
- Reported in addition: P1 and P2 on the ws2 folds alone and on the ws4 folds alone (per-seed means over the
  folds each machine ran), so a machine effect would be visible.
- ws4 runs at most two processes at once (about 12 GB of GPU memory is free there and belongs to other jobs).

## Amendment 2 — 2026-11-13 01:10, ws2 stopped; split over ws3 and ws4 (written before any ws3 / ws4 score exists; still no ws2 score of seeds 148022-148025 looked at, only which folds are present)
ws2 was stopped at a fold boundary at the user's request with 21 folds per seed (held tasks 0 1 2 4 5 7 8 10 11
13 14 15 16 17 18 20 21 23 24 26 27; 84 folds, anchors bit-exact). The other 20 tasks per seed (29 30 31 32 33
35 37 39 40 41 42 43 44 45 47 49 51 52 53 54; 80 folds) run on two machines the user named, reached at the
endpoints the user gave: ws3 (port PORT) and ws4 (port PORT). Both are single A800 80 GB GPUs already at
100% utilisation from the user's other jobs, with the same driver and the same private environment
(PyTorch 2.12.1+cu126) on a shared home directory. Amendment 1 holds, with these specifics:

- Seeds 148022 and 148023 run on ws3, seeds 148024 and 148025 on ws4, two processes per machine (one per seed),
  each seed's 20 tasks in order. Outputs are separate per machine (`out_ws3/`, `out_ws4/`).
- Calibration (the 24 folds of Amendment 1: held tasks 0, 15, 29, 7, 21, 39 at each seed) is run for each seed
  on the machine that trains that seed, 12 folds per machine, BEFORE that seed's `tfilm_nl` folds. The +/-2% rule
  is applied per machine (t interval over its 12 folds, df 11): a machine whose flow-MSE or action-error interval
  is not inside [-2%, +2%] has its `tfilm_nl` folds compared with `film_nl` retrained on that machine instead.
- Consequence to state with the result: the split is by held task, not random. The ws2 folds are the first 21
  tasks in id order and the ws3 / ws4 folds the last 20, so a per-machine reading also differs in tasks; the
  per-machine numbers are reported as a check, not as a machine comparison.

## Amendment 3 — 2026-11-13 01:55, shared queue on ws3 / ws4 (written before any ws3 / ws4 score has been read; only fold counts and seconds per fold were looked at)
Observed: with `nice -n 5` the lanes took about 490 s per fold on ws3 and received about 10% of a processor
core on ws4 (45 runnable threads on 8 cores), where no fold finished in 9 minutes. The ws4 lanes were stopped
with no record written. Changes, none of which touches training or scoring:
- Lanes run at normal priority (no `nice`), two per machine, and wait while less than 7 GB of GPU memory is free.
- The remaining `tfilm_nl` folds (20 tasks x 4 seeds) form one queue shared by both machines through claim
  directories (`nodes/tc_run_any.py`: the same `train_arm`, `score_forward` and `action_error` calls as
  `tc_run.py`; bookkeeping only), so a seed's folds may come from either machine. Every record names its lane.
- Calibration stays pinned: ws3 retrains `film_nl` on tasks 0, 15, 29, 7, 21, 39 for seeds 148022 and 148023,
  ws4 for seeds 148024 and 148025 (12 folds per machine; the folds ws3 already finished are kept). The +/-2% rule
  of Amendment 1 is applied per machine and decides the reference for every fold that machine produced.
- Added check: ws4 retrains one fold ws3 already ran (`film_nl`, seed 148022, held task 0). If the two scores
  are bit-identical the two machines are reported as numerically identical; the rule above is unchanged.

## Amendment 4 — 2026-11-13 02:45, a third machine (written before any score of this machine exists)
ws3 and ws4 finish about nine folds an hour together, so this computer (one RTX 4060 Ti 8 GB, Windows,
PyTorch 2.12.1+cu126, NumPy 2.2.6; `local_run/`, entry point `tc_run_local.py`: the unchanged `tc_run.py` with the
two stored-score paths redirected) is added as machine `local`. Same code (per-file checksums equal to ws2),
same data file (SHA-256 verified), same recipe. It runs one process at a time, in this order:
1. Calibration: `film_nl` on held tasks 0, 15, 29, 7, 21, 39 at seed 148022 and seed 148023 (12 folds).
   The +/-2% rule of Amendment 1 applies to this machine as to the others.
2. PLAN_ACTION.md's `plain_nl` at seed 148021 (41 folds), then `film` at seed 148021, so that ws3 / ws4 spend
   their time on the `tfilm_nl` folds; folds another machine already holds are skipped (the lanes on ws3 / ws4
   skip folds this machine has published to the shared home).
`tfilm_nl` folds are not run on this machine unless a later amendment says so.

Correction to Amendment 4 (2026-11-13 02:35): its header time should read 02:22, the clock of this computer when
it was written; the first calibration fold of machine `local` had finished by then, its score unread.

## Amendment 5 — 2026-11-13 02:40, same-machine pairs replace the stored reference off ws2
What is known now: the calibration scores of `film_nl` on ws3 (12 folds), ws4 (4 folds) and `local` (3 folds),
and that ws3 and ws4 reproduce each other bit-identically on one fold. No `tfilm_nl` score from seeds
148022-148025 has been read on any machine.
Calibration outcome so far, `film_nl` retrained versus the stored ws2 score of the same fold:
ws3 flow MSE -1.25% [-2.78, +0.30], action error -0.32% [-3.10, +2.54], per-fold SD 2.5% / 4.5%, no fold
bit-identical. The intervals are not inside +/-2%, so by the rule of Amendment 1 the folds run off ws2 are
compared with `film_nl` retrained on the machine that ran them. The per-fold spread (2.5% to 4.5%, single folds
up to 14%) is retraining noise between software builds (CUDA 12.6 here, 13.0 on ws2), not a bias I can detect.
Consequences, fixed now:
- Every `tfilm_nl` fold run off ws2 is paired with `film_nl` retrained for the same fold and seed on a machine
  of the same build family. ws3 and ws4 count as one machine (bit-identical on the duplicated fold; the folds
  already finished on both are compared again when the run ends and reported). `local` is its own machine.
- Split of the 80 remaining folds (20 tasks x 4 seeds): seed 148022 on ws3 / ws4 (`tfilm_nl` and `film_nl`,
  40 folds); seeds 148023, 148024, 148025 on `local`, one process at a time, each seed's `tfilm_nl` then its
  `film_nl` (120 folds). ws3 / ws4 continue their shared queue as a fallback; where a `tfilm_nl` fold exists
  on two machines, the one whose machine also has the paired `film_nl` fold is used, `local` first.
- The ws2 folds (first 21 tasks per seed) keep the stored ws2 `film_nl` as reference (bit-exact anchors).
- The pending calibration folds on ws4 and `local` are no longer needed for the rule and are reported as a
  description of cross-build retraining noise.
- If a seed's 20 pairs are incomplete when the night ends, P1 and P2 are reported on the complete seeds only,
  with the number of seeds stated; nothing is reported for a seed with missing pairs except its ws2 half,
  which is labelled as such.

## Amendment 6 — 2026-11-13 02:50, allocation for the rest of the night (no `tfilm_nl` score of seeds 148022-148025 has been read on any machine)
Throughput measured: `local` 153 s per fold (one process); ws3 about 450 s and ws4 about 700 s per fold with two
lanes each. All remaining work cannot finish tonight, so the order is fixed here.
- ws3 / ws4 (one build family, shared queue): seed 148022 on the 20 remaining tasks, `tfilm_nl` then `film_nl`;
  then, as a fallback, seed 148025 in the same way. The pinned calibration on ws4 is stopped (Amendment 5).
- `local`: (1) PLAN_LODO.md arms plain, sat, xy6; (2) PLAN_ACTION.md `plain_nl`, seed 148021; (3) seed 148023 on
  the 20 remaining tasks, `tfilm_nl` then `film_nl`; (4) seed 148024 likewise; (5) seed 148025 likewise;
  (6) the remaining arms of PLAN_LODO.md and PLAN_ACTION.md.
- Added co-primary, because not every seed will be complete: P1' and P2' are P1 and P2 restricted to the 21 held
  tasks that ws2 ran for every seed (ids 0 to 27: the eight-direction grid, directions 0 to 12 of 16), all five
  seeds, stored ws2 `film_nl` as reference, t interval over seeds (df 4). These 21 tasks were fixed by the order
  of the queue when ws2 was stopped, not chosen. P1 and P2 on all 41 tasks are reported on the seeds whose 20
  remaining pairs are complete, with the number of seeds stated (df = seeds - 1).

## Amendment 7 — 2026-11-13 02:58, schedule only (no analysis rule changes; no new-seed `tfilm_nl` score read)
Correction of a throughput figure: with two lanes each, ws3 finishes about 16 folds an hour and ws4 about 10,
not nine together. Schedule: ws3 / ws4 run the pairs of seed 148022, then 148025, then 148024. `local` runs the
pairs of seed 148023 first, then PLAN_LODO.md (plain, sat, xy6), then PLAN_ACTION.md (`plain_nl`), then the
remaining arms of both, then seed 148024 if ws3 / ws4 have not finished it. Pairing rule as in Amendment 5.

Note on times (2026-11-13 02:41 by this computer's clock): the header times of Amendments 4 to 7 were typed
ahead of the clock; all four were written between 02:22 and 02:41 in the order given. The record of what was
known when each was written is unaffected: no `tfilm_nl` score of seeds 148022-148025 has been read.

## Amendment 8 — 2026-11-13 02:45 (clock of this computer), final schedule for the night; analysis rules unchanged
So that the five-seed result completes as early as possible and every comparison is same-build:
- `local`: pairs (`tfilm_nl`, then `film_nl`, 20 remaining tasks) of seed 148023, then of seed 148024; then
  PLAN_LODO.md arms plain, sat, xy6.
- ws3 / ws4: pairs of seed 148022, then of seed 148025; then PLAN_ACTION.md at seed 148021: `plain_nl` (41
  tasks), then `film_nl` (41 tasks, the same-build partner PLAN_ACTION.md requires), then `film`.
Still no `tfilm_nl` score of seeds 148022-148025 has been read on any machine.
