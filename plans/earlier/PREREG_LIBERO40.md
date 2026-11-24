# Confirmation on LIBERO40 — registered 2026-10-24 ~14:30, before any run on this pool

## Purpose
The method (FiLM-modulated nonlinear head + gradient descriptor inversion + closed-form residual atom-library code)
was designed on the 41 v5 LOTO folds (PREREG_NL.md, PREREG_HYB.md, PREREG_COMBO.md), so those results are
development evidence. This study tests the same method, unchanged, on a pool never used to develop it: the 40
heterogeneous LIBERO tasks of the G39 substrate (spatial, object, goal and long suites; human demonstrations;
different skills, objects and scenes). The pool was used before only for the coverage experiment (target-supervised
codes of the published head); no part of the new method has been run on it. It has no sealed split, so results are
"a different task family", not "held out once".

## Data
G39 substrate shards copied from the workspace (sha256 of all 40 shards verified against the source,
copy_check.txt). Language-blind frozen features (31,680-D), 16-step action chunks with all 7 action dimensions
active (DEAD_ACTION_DIMS = (), as for the earlier G39 runs). Splits as built: TRAIN (20-35 episodes per task) for
training and few-shot demonstrations, VAL (5-7 episodes) primary scoring, TEST (4-8 episodes) secondary.
Task descriptor: bag of the instruction's content words over the vocabulary of the 40 instructions (stop words:
the, a, an, and, it, of, to) plus a constant, as in the closed-loop study. Instructions are mapped to task ids by the
substrate's own rule (sorted suite, then sorted instruction), reconstructed from the LeRobot copy of the same
demonstrations; verified: episode counts agree for 40/40 tasks (1,693 episodes in both) and the sorted episode
lengths of every task agree within one sampling stride (msal_ext/libero40_tasks.py, libero40_tasks.json).
Words that occur only in a held task's instruction are never active in training; their weights stay at
initialization (registered as is).

## Method (identical settings to the development studies)
Leave-one-task-out over the 40 tasks, seeds 148021-148025. Per fold, trained on the TRAIN rows of the other 39 tasks
with the unchanged N1 loop (balanced batches, 12,800 AdamW updates, lr 2e-4, weight decay 1e-5):
- R: FiLM with the word descriptor, then linear head + readout MLP 512 -> 1472 -> SiLU -> 112 (film_nl);
- residual library on frozen R: 16 nonlinear atoms (512 -> 92 -> SiLU -> 112) + free source code table,
  orthogonality 0.02 and gain-weighted group sparsity 0.001 (as PREREG_HYB.md);
- zero-shot comparators: no task input + the same readout MLP (plain_nl); published-style FiLM with a linear head.
Adaptation for k in {1, 2, 4, 8, 16} TRAIN demonstrations of the held task (its first k episodes): descriptor
inversion (200 Adam steps, lr 1e-2, from d*); hybrid (closed-form residual code, ridge 0, 4 noise draws); and the
method, the combination (inversion, then the closed-form residual code at the inverted descriptor). Scored on the
held task's VAL rows with 8 paired noise draws (TEST secondary), double precision for the adaptation variants.

## Comparisons (paired log ratio per fold, mean per seed, 95% t interval over 5 seeds, df = 4)
Primary (the method's claims): (L1) combination k = 16 vs inversion k = 16; (L2) combination k = 4 vs inversion
k = 4; (L3) combination k = 16 vs zero-shot R.
Secondary: (L4) combination k = 1 vs inversion k = 1; hybrid and inversion vs zero-shot R for every k;
(L5) zero-shot R vs plain_nl (does the descriptor help zero-shot on heterogeneous tasks); (L6) zero-shot R vs
linear-head FiLM (does the nonlinear readout help); TEST rows for L1-L3.
Reading: "better" if the interval's upper bound < 0, "worse" if the lower bound > 0, else "not distinguishable".
The development finding "the library adds accuracy on top of inversion" is confirmed only if L1 and L2 are
"better"; every comparison is reported whatever the outcome.

## Runs and failure rules
Ten jobs (each seed split into two halves of 20 folds), six workers, three per GPU (features about 4.4 GB per
worker on the GPU). Code: msal_ext/l40.py (reuses msal_ext.hyb / combo functions and msal_exp.run_loto's
training and scoring unchanged; only the data source, the descriptor and the dead dimensions differ). NaN: the fold
is retrained once with the same seed; a second failure is reported, not replaced. Nothing is changed after the
first run starts.

## Execution note (2026-10-24 18:09, workstation clock)
Job seed 148025, tasks 1000-1019 stopped at 18:04 with a CUDA out-of-memory error on GPU 0 in its second fold
(during block construction, before any score of that fold was written): LIBERO tasks have more rows than the
development tasks, a worker peaked at about 9 GB, and GPU 0 also carried processes of another user. This is an
infrastructure failure, not a result. The job is restarted on GPU 1 with PYTORCH_CUDA_ALLOC_CONF=expandable_segments
and resumes after its one completed fold; no setting of the method, data or analysis changes.
