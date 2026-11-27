# PLAN_PENALTY: sensitivity of MSAL's adaptation to the two stage-2 penalty weights

Written 2026-11-27, before any run of this study. No directional hypothesis; the reading rule below is fixed here.

## Why
Equation (3) of the main paper weights the orthogonality penalty by 0.02 and the gain-weighted group penalty by
0.001. Both were fixed before any study (0.02 inherited from the training script of the earlier linear library,
0.001 declared in writing before its first run) and never tuned. Zero-shot prediction uses only the stage-1 head and
cannot depend on them. This study measures how much the adapted head depends on them.

## Design
- Development family, leave-one-task-out over the 41 source tasks, training seed 148021, MSAL's head (`tfilm_nl`).
- Per held task, stage 1 is trained once (as `tc_combo.run_fold`). Stage 2 (`tc_combo.train_library_t`) is then
  trained once per setting, from the same initialization and with the same batches:
  - S0 default: orthogonality 0.02, group 0.001
  - S1 orthogonality 0
  - S2 orthogonality 0.08 (four times the default)
  - S3 group 0
  - S4 group 0.004 (four times the default)
- Adaptation exactly as `tc_combo.run_fold` for `tfilm_nl`: shots k in `hyb.SHOTS`, descriptor inversion from the
  given descriptor, closed-form residual code. Descriptor inversion does not read the library, so it is computed once
  per k and shared by the five settings.

## Measures (per task and setting)
- flow error of the adapted head (inverted descriptor and code), selection and robustness splits, every k;
- sampled-chunk error of the adapted head at k = 1 and 16 (`ideal.action_error`, as stored);
- zero-shot flow error with zero code (sanity check: must agree across settings up to floating point);
- stage-2 final loss and the norm of each code slot over the source codes (number of active slots).

## Reading rule
For each non-default setting S1-S4 and k in {1, 16}: the paired change of the adapted sampled-chunk error against S0,
as 100(e^l - 1) with l the mean over the 41 tasks of the log ratio, and a 95% percentile bootstrap interval over tasks
(10,000 resamples, generator seed 0). The same for the adapted flow error (selection split) at every k. All readings
are reported in the supplement; the main text states the largest absolute change and its interval. No reading is
dropped. A setting whose stage 2 fails (non-finite loss twice) is reported as failed for that task.

## Machines and claims
Each task runs entirely on one machine. ws2 GPU 1 through `gpu_run train` (two processes in the slot) and this PC's
RTX 4060 Ti. Tasks are claimed by `mkdir` in ws2 `trunkcond/l40/claims_pen/`. Stop files: ws2
`trunkcond/l40/STOP_PEN`, this PC `local_run/STOP_PEN`. A smoke run (20 steps) on each machine precedes the main run.
Runner: `tc_penalty.py` (ws2 `trunkcond/l40/`, this PC `local_run/`). Analysis: `tc_penalty_analyze.py`.
