# Registered follow-up: readout-only fine-tuning in closed loop (written 2026-11-11, before any run)

## Why
The offline grid of PREREG_FT.md contains readout-only fine-tuning. Read post hoc (analysis/ft_readout.py), the
readout fine-tuned at the setting selected on the development folds is more accurate than MSAL at every k on both
families (LIBERO40: MSAL's flow error 16-47% higher) and takes 11-49 s on a CPU. The closed-loop adaptation study
(PREREG_CLOSED_LOOP_ADAPT.md) ran only whole-head fine-tuning. This follow-up runs the readout-only variant on the
same eight held tasks and episodes.

## What is known when this is written
All offline results; all closed-loop results of PREREG_CLOSED_LOOP_ADAPT.md, its exploratory follow-ups and
PREREG_PERTURB.md (zero 0.25%, MSAL 16.0 / 13.8%, whole-head fine-tuning 41.0 / 59.2%, seen head 52.8%).
No readout-only policy has been rolled out.

## Policies (fixed before running)
For each held task h in {1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037} and k in {1, 16}: the zero-shot head R_h of
the closed-loop study (retrained with the same code and seed 148021; must equal the stored head bit-exactly), with
only its readout (film_head, film_mlp; 977,568 weights) fine-tuned on the first k TRAIN demonstrations exactly as in
msal_ext/ft.py, at the readout-only setting that the registered selection rule picks on the development folds:
- k = 1: Adam 1e-4, 100 steps (k1_readout_lr0.0001_s100)
- k = 16: Adam 1e-3, 400 steps (k16_readout_lr0.001_s400)
Each fine-tuned head must reproduce its stored offline SELECTION score (runs/ft, seed 148021) bit-exactly; a head
that does not is reported and its episodes are still run and marked.
Rollout: unchanged script, settings and seeds of Stage B (perturb/scripts/cl_rollout.py, N_exec 16, 10 flow steps,
episodes 0-49, LIBERO's unperturbed initial states, GPU 1 only), so every episode pairs with Stage B.

## Comparisons (cl_adapt.compare: paired, two-level bootstrap over tasks then episodes, 95% interval;
"better"/"worse" only if the interval excludes zero)
Primary
- R1: readout fine-tuning k=16 vs MSAL k=16
- R2: readout fine-tuning k=1 vs MSAL k=1
Secondary
- R3, R4: readout vs whole-head fine-tuning at k=16 and k=1
- R5, R6: readout fine-tuning vs zero-shot at k=16 and k=1
Descriptive: per-task success; success of the seen head for reference.

## Reading
- R1 and R2 better: a trainable readout beats MSAL in closed loop as well as offline; the paper says so.
- Not distinguishable or worse: the offline advantage of the readout does not carry to success; the paper says so.
Either way the result is reported, in VI-D/VI-E and supplementary Table S7 or S8, and the outcome does not change
MSAL's numbers. Perturbed layouts are not part of this follow-up.

## Stopping and exclusions
800 episodes; no early look changes the plan. Episodes that fail for infrastructure reasons are retried (up to 5
attempts per job); a job that cannot complete is reported, and its task is dropped from all comparisons.

## Files
experiments/msal_ext/cl_heads_ro.py (artifacts roft{h}.pt), experiments/review/driver_ro.sh (remote:
/home/ubuntu/work/manuscript6/readout_cl/), analysis/cl_readout.py, results_remote/readout_cl/.
