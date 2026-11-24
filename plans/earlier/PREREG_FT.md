# Fine-tuning baseline for few-shot adaptation — registered 2026-10-25, before any run of this study

## Purpose
MSAL adapts to a new task without changing any network weight: it inverts the head's task input and fits the
residual code in closed form. The standard alternative is to fine-tune the head's weights on the k
demonstrations; the paper has no such baseline. This study compares MSAL with fine-tuning on the development
family and on LIBERO40, with the fine-tuning setting chosen on the development family. The user approved it on
2026-10-25 and asked that it run after the coverage study (PREREG_COV.md).

## Design
Folds and heads: the stored combination study (development: 41 tasks, LOTO, seeds 148021-148025; PREREG_COMBO.md)
and the stored LIBERO40 confirmation (40 tasks, LOTO, the same seeds; PREREG_LIBERO40.md). Per fold the zero-shot head
R (film_nl, or l40_film_nl with the word descriptor) is retrained with the unchanged loop on all other tasks.
Anchor, every fold: before fine-tuning, R's float32 score must equal the stored film_nl score, and its
double-precision zero-shot score the stored record's, bit-exactly; MSAL's scores are then taken from the stored
records of the same fold.
Fine-tuning, for each k in {1, 2, 4, 8, 16} (the held task's first k TRAIN demonstrations, as in the stored runs):
- variants: "readout" (the linear map W and the readout MLP M; FiLM maps and trunk frozen) and "all" (every
  parameter of R);
- Adam (default betas, no weight decay), learning rate in {1e-4, 1e-3}, 400 steps; each step draws 512 rows with
  replacement from the demonstrations' rows with fresh noise and flow time (the training loss, descriptor fixed at
  the held task's d*); row sampling from np.random.default_rng([seed, held, k, variant, lr index, 7]);
- scored after 25, 100 and 400 steps with the stored runs' double-precision scoring at d* (eight paired noise
  draws; SELECTION/ROBUSTNESS rows for the development family, validation/test for LIBERO40).
That is 12 settings (variant x learning rate x steps) per k.

## Selection rule (fixed now)
For each k, the "development-selected" setting is the one with the lowest mean, over the 205 development folds,
of the log ratio of its SELECTION error to the zero-shot error. It is applied unchanged to LIBERO40.
On the development family the selected setting is chosen on the same rows it is scored on, which favours
fine-tuning; that comparison is development evidence. The LIBERO40 comparison is the clean test.

## Comparisons (paired log ratio per fold, mean per seed, 95% t interval over 5 seeds, df = 4)
Primary, LIBERO40: (F1) MSAL vs development-selected fine-tuning at k = 16; (F2) at k = 4; (F3) at k = 1.
Secondary: (F4) the same on the development family at every k (selection favours fine-tuning); (F5)
development-selected fine-tuning vs zero-shot and vs descriptor inversion, both families, every k; every setting
against zero-shot on both families (supplementary table); exploratory: MSAL vs the LIBERO40-best setting per k
(chosen on the scored rows, favours fine-tuning). Cost: seconds per adaptation for each setting and for MSAL.
Reading: "better" if the interval's upper bound < 0, "worse" if the lower bound > 0, else "not distinguishable".
Every comparison is reported whatever the outcome.

## Runs and failure rules
Code msal_ext/ft.py (imports the training, scoring and data code of msal_ext.hyb / combo / l40 unchanged).
Launcher remote_scripts/launch_ft.sh: starts only after the closed-loop and coverage studies have finished, runs
one anchor fold per family, then 20 jobs (family x seed x half of the held tasks) on six workers, three per GPU,
with the GPU 0 gate of the coverage study before each fold. NaN: the fold is rerun once; a job that exits with an
error (for example out of GPU memory) is restarted once and resumes after its last completed fold; a second
failure is reported, not replaced. Nothing is changed after the first run starts.
Output: /home/ubuntu/work/manuscript6/runs/ft/<job>/folds.jsonl; analysis analysis/aggregate_ft.py.
