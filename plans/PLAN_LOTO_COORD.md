# PLAN_LOTO_COORD: coordinates plus random columns on held tasks of trained directions (leave-one-task-out)

Written 2026-11-15 after the user's go-ahead (reply to "also test the recommended descriptor on leave-one-task-out
folds?": "ws4马上关闭了。用ws3吧。"). Hashed on ws2 before the first run.

## What is known
PLAN_LODO_CONFIRM.md (new seeds, new draw) confirmed, on directions never trained, that the six coordinate columns
plus 15 trained random columns (xy6r2; N(0, 0.7^2), generator seed 20261116) beat the draft's 21-column harmonic
descriptor after MSAL's adaptation (sampled chunk -4.9% / -5.3%) and zero-shot (-61.1%). Held tasks of trained
directions (the leave-one-task-out folds of the paper's main results) were never run with it.

## Question
On leave-one-task-out folds, is xy6r2 at least as accurate as the draft's harmonic descriptor, zero-shot and after
MSAL's adaptation? If so it can replace the harmonic descriptor as the paper's general default.

## Runs
Arms: film_nl (the draft's MSAL head with the harmonic descriptor, run_nl's own arm) and lodo_xy6r2 (same head,
xy6r2 descriptor; registered by tc_lodo_conf.py). Leave-one-task-out: 41 held tasks x seeds 148021-148025 = 205
(seed, task) units, both arms per unit in one process on one machine. Per arm: stage 1 on the other 40 tasks, the
residual library, then the adaptation of PLAN_LODO_ADAPT.md (inversion + residual code at k = 1, 2, 4, 8, 16;
sampled-chunk error on SELECTION rows zero-shot and at k = 1, 16). Runner: local_run/tc_loto_coord.py.
Machines: ws2 GPU 1 (two lanes), ws3 (four lanes; ws4 is being shut down), this computer (one lane); claims in
claims_lotoc on the shared home, reached through ws3. Anchor first, on ws3: the film_nl arm of one stored ws3 unit
(seed 148021) must equal the stored PLAN_ADAPT_TRUNK.md film_nl record bit for bit (equality only).

## Readings (draft's rule: per seed mean over the 41 tasks of the paired log ratio, t interval over seeds, df 4;
bootstrap over the 41 tasks as a view; "lower/higher" only if the interval excludes 0)
Primary: P1 complete method, xy6r2 vs film_nl, sampled-chunk error at k = 1 and 16; P2 zero-shot sampled-chunk
error, xy6r2 vs film_nl.
Secondary: flow MSE for P1 and P2; inversion alone; method vs own zero-shot for each arm; descriptor shift.
Rule fixed now: xy6r2 is proposed to the user as the paper's general default descriptor only if P1 is not higher at
either k and P2 is not higher; "better" is stated only where an interval lies below 0. Otherwise the recommendation
stays restricted to untrained placements, and this study is reported as it came out.

## Limits
Offline, one skill and its 41 tasks; closed loop is not possible for this family (no simulator).

## Amendment 1 (2026-11-15, before any score was read)
The first lane scripts looped over task numbers 0-40 instead of the 41 source task ids (0 1 2 4 5 7 8 10 11 13 14 15
16 17 18 20 21 23 24 26 27 29 30 31 32 33 35 37 39 40 41 42 43 44 45 47 49 51 52 53 54). Numbers that are not source
tasks stopped before writing a record (held task has no rows); ids above 40 were not reached. The lane loops were
stopped by PID (running units completed both arms), the scripts now loop over the source ids, and new lanes started.
No arm, fold, setting or reading changes.
