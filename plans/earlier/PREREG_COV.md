# Source coverage of MSAL's residual library — registered 2026-10-25, before any run of this study

## Purpose
The paper's limitation (sections_v2/discussion.tex): the residual code is fitted to the new task's own
demonstrations, and for the published library such target-supervised codes hurt held-task accuracy with 8 source
tasks and helped only with 23 to 36; MSAL's library was trained on 39 or 40 source tasks, and how its gain
depends on the number of source tasks was not measured. This study measures it on the development family. The
user approved it on 2026-10-25 and asked that it run after the closed-loop study.

## Design
Development family (v5, 41 tasks, language-blind features, source rows only; same data file as the NL, hybrid and
combination studies). Leave-one-task-out over all 41 held tasks, seeds 148021-148025.
Levels: n in {8, 16, 24, 32} source tasks. For held task h and seed s, the source tasks are the first n of a random
permutation of the other 40 tasks drawn with np.random.default_rng([20261025, s, h]) (levels nested within a fold;
different subsets across seeds and folds), passed to training in the order of source_task_ids() as in the stored runs. Level n = 40 (all other
tasks) is the stored combination study (PREREG_COMBO.md), reused unchanged.
Per fold, unchanged code (msal_ext.hyb / msal_ext.combo functions; new runner msal_ext/cov.py only selects the
subset): stage 1 trains film_nl, stage 2 the residual library (16 nonlinear atoms, free code table over the n
sources, orthogonality 0.02, gain-weighted group sparsity 0.001), each 12,800 AdamW updates; balanced batches use
512 // n rows per source task, as the existing loops do (so total rows per update stay about 480-512). The held task
is scored zero-shot, with the library's closed-form code alone (hybrid), by descriptor inversion and by MSAL
(inversion, then the closed-form code) for k in {1, 2, 4, 8, 16} of its TRAIN demonstrations, on SELECTION rows
(ROBUSTNESS secondary) with 8 paired noise draws.

## Anchor (before the study starts)
The runner is first run at n = 40 on the first held task of source_task_ids() with seed 148021; every score
(zero-shot, inversion and combination at every k, both splits) must equal the stored combination record
bit-exactly, and the film_nl anchor must equal the stored film_nl score. If not, the study does not start.

## Comparisons (paired log ratio per fold, mean per seed, 95% t interval over 5 seeds, df = 4)
For each level n in {8, 16, 24, 32} and, from the stored runs, n = 40:
- (V1, primary) MSAL vs inversion at k = 16: does the library add accuracy beyond inversion with n sources?
- (V2) the same at k = 1.
- (V3) library-only closed-form code vs zero-shot at k = 16.
- (V4) MSAL vs zero-shot at k = 16.
Trend (primary): per seed, the least-squares slope of the V1 fold-mean log ratio on log2(n) over the five levels;
mean slope with 95% t interval. Descriptive: zero-shot error at each n relative to n = 40.
Reading rule: "better" if the interval's upper bound < 0, "worse" if the lower bound > 0, else "not
distinguishable"; the slope is "the library's gain grows with coverage" if its interval lies below 0. Every
comparison is reported whatever the outcome; the paper's limitation sentence is replaced by the result.

## Runs and failure rules
Launcher remote_scripts/launch_cov.sh: starts only after the closed-loop study has finished (driver.log "driver
done" and no closed-loop process), runs the anchor, then 40 jobs (seed x level x half of the held tasks) on six
workers, three per GPU. Before each fold a runner on GPU 0 waits until no process other than this study's runners
is on GPU 0 (user rule of 2026-10-24: never compete with the other user's workload). NaN: the fold is retrained once
with the same seed; a second failure is reported, not replaced. Nothing is changed after the first run starts.
Output: /home/ubuntu/work/manuscript6/runs/cov/seed{S}_n{N}/folds.jsonl; analysis analysis/aggregate_cov.py.
