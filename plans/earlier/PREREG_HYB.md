# Hybrid method: modulated nonlinear head + closed-form residual atom library — registered 2026-10-24 ~10:00, before any run

## Provenance (honest)
Chosen by the user on 2026-10-24 after the registered nonlinear-library study (PREREG_NL.md): FiLM feeding a
nonlinear readout ("film_nl") was the most accurate zero-shot head; code-affine libraries gave up 11-20% zero-shot
accuracy against it, and even a closed-form code fitted from the target's 16 demonstrations did not recover it.
The hybrid keeps film_nl for zero-shot prediction and adds a shared atom library as a code-affine residual for
closed-form few-shot adaptation. It is designed after seeing these 41 LOTO folds, so every result here is
**development evidence**; a confirmation on data not used for development is required before it is the paper's
headline result.

## Method
Stage 1 (zero-shot model R): exactly the film_nl arm (FiLM modulation of the trunk activation, then linear head plus
readout MLP 512 -> 1472 -> SiLU -> 112), trained by the unchanged N1 loop with the same seed, so it must reproduce
the stored film_nl per-fold SELECTION score bit-exactly (anchor; any mismatch voids the fold).
Stage 2 (residual library): R frozen. K = 16 nonlinear atoms Psi_k (512 -> 92 -> SiLU -> 112, NonlinearAtomHead) on
R's trunk activation h, and a free code table Z (40 source tasks x 16, init N(0, 0.02^2)). Field
v = R(h, d) + sum_k z_{t,k} Psi_k(h), trained on the TRAIN rows of the 40 source tasks: balanced batches (12 rows
per task), 12,800 AdamW updates (lr 2e-4, weight decay 1e-5), masked flow MSE + 0.02 output orthogonality of the
atoms + 0.001 gain-weighted group sparsity on Z (gain detached) -- SAL's published recipe. Batches and noise use
their own seeds (no reuse of stage-1 draws). The field is exactly affine in z (Lean: field_code_additive).
Test, for the held task with descriptor d*:
- zero-shot: z = 0, i.e. R alone (= film_nl);
- few-shot, k in {1, 2, 4, 8, 16} demonstrations (the held task's first k TRAIN episodes): z* by the published
  closed-form weighted least squares (weighted_ls_code, ridge 0) on the residual target - R over those rows with 4
  noise draws; no parameter of R or of the library changes and no gradient step is taken.
All variants are scored on the held task's SELECTION rows with the 8 paired noise draws used by every arm
(ROBUSTNESS rows secondary), in double precision.

## Baselines and controls (same demonstrations, same scored tensors)
- Descriptor inversion: adapt R's own task input -- d initialized at d*, 200 Adam steps (lr 1e-2) on the same k
  demonstrations' blocks, then scored. Same order of adapted parameters (21 vs 16), but uses gradient steps.
- Random-atom control (k = 16 only): the same closed-form fit with an untrained library (same initialization),
  to show that training the library matters.

## Runs
Seeds 148021-148025, language-blind feature, all 41 folds, one job per seed (five workers). Code:
msal_ext/hyb.py (runner python -m msal_ext.hyb), reusing msal_exp.run_loto's data loading, training loop for
stage 1 and scoring tensors unchanged.

## Comparisons (paired log ratio per fold, mean per seed, 95% t interval over 5 seeds, df = 4)
Primary: (H1) hybrid few-shot k = 16 vs zero-shot R; (H2) hybrid few-shot k = 4 vs zero-shot R;
(H3) hybrid few-shot k = 16 vs descriptor inversion k = 16.
Secondary: the k curves of both adaptation methods; (H4) hybrid k = 16 vs random-atom control; (H5) hybrid
few-shot k = 16 vs the stored zero-shot nonlinear-base library (sal_nlb) and linear SAL; ROBUSTNESS rows for H1-H3.
Reading: "better" if the interval's upper bound < 0, "worse" if the lower bound > 0, else "not distinguishable".
Every comparison is reported whatever the outcome; nothing is changed after the first run starts.

## Failure rules
Stage-1 anchor mismatch voids that fold (reported). NaN in any stage: the fold is retrained once with the same
seed; a second failure is reported, not replaced.
