# Exploratory arm "sal_tied" — registered 2026-10-21 ~21:00 CST, before any run of it

## Why it exists (honest provenance)
Designed **after** seeing three-seed N1 results on the same 41 development folds: FiLM tied SAL
(−3.3%, CI −7.9…+1.4) with ~40× fewer task-conditioned parameters, the unpenalised SAL was +12% worse,
the unconstrained bilinear head ≈+30% worse. Algebra: FiLM + linear head = a library of 512 rank-one atoms
tied to the output map with a code linear in d. This arm combines the two constraints that worked: tied
atoms (FiLM) and a low-dimensional descriptor code with SAL's two penalties. Because it was chosen after
looking at these folds, it is reported as **exploratory**, whatever the outcome; no other variant will be
tried on these folds.

## Definition
Same trunk H, data, batches, noise draws, schedule and scoring as every N1 arm (msal_exp.run_loto, unchanged).
- code: z = W_cᵀ φ(t), φ = saturated_v5 (21-D), K = 16, W_c ~ N(0, 0.02²), no per-task parameters
- atoms tied to the base map B = [B_h, b]: Ψ_k = B_h (u_k ⊙ h + v_k), u_k, v_k ∈ R^512, init N(0, 0.02²)
- field: v̂ = B h̄ + Σ_k z_k Ψ_k  (code-affine → same nn/mean/oracle scoring path as SAL)
- loss: masked flow MSE + 0.02 · output orthogonality(Ψ) + 0.001 · gain-weighted group sparsity (gain detached)
- task-conditioned parameters: U, V (2 × 16 × 512) + W_c (21 × 16) = 16,720
- implementation: msal_ext/tied.py (new package; msal_exp/ and lib/ untouched), runner msal_ext/run_tied.py

## Runs
Seeds 148021–148025, language-blind feature, all 41 LOTO folds, SELECTION primary, ROBUSTNESS secondary.
Started only after the main N1/N2 queue has drained.

## Comparisons and reading (paired log ratio per fold, mean per seed, 95% t interval over 5 seeds)
1. sal_tied vs blind/sal; 2. sal_tied vs blind/film. Descriptive labels:
- "better" if the interval's upper bound < 0; "worse" if the lower bound > 0; otherwise "not distinguishable".
- Also reported: folds won per seed, worst fold, reuse-code baselines (nn, mean) as for SAL.
No outcome of this arm changes the reading of the registered N1/N2 arms.
