# Combination test: descriptor inversion + closed-form residual library — registered 2026-10-24 ~10:55, before any run

## Provenance (honest)
Proposed after an interim look at 70 of the 205 hybrid folds (PREREG_HYB.md): the closed-form residual code
improved zero-shot R by about 11% on every fold, but adapting R's own task input by gradient descent (descriptor
inversion) improved it by about 19% and beat the hybrid on about 4 of 5 folds. Question: do the two adaptations
add up, i.e. does the library still contribute accuracy once R's task input has been adapted? The user approved
the test. Designed after looking at these folds: development evidence, like the hybrid.

## Method (everything as in PREREG_HYB.md unless stated)
Per fold and seed, R (film_nl) and the residual library are retrained with the hybrid study's exact code and seeds,
so both are identical to the hybrid run's (stage-1 anchor against the stored film_nl score; the recomputed hybrid
k = 16 and inversion scores must equal the hybrid run's records, checked in the analysis).
Combination, for k in {1, 2, 4, 8, 16} demonstrations (the held task's first k TRAIN episodes):
1. descriptor inversion exactly as in the hybrid study (d from d*, 200 Adam steps, lr 1e-2, on the k demonstrations);
2. then the closed-form residual code z* (weighted_ls_code, ridge 0) on the same demonstrations, with the residual
   taken against R at the inverted descriptor;
3. score R(h, d_inv) + sum_k z*_k Psi_k(h) on the held task's SELECTION rows (8 paired draws; ROBUSTNESS secondary).
No parameter of R or of the library changes.

## Comparisons (paired log ratio per fold, mean per seed, 95% t interval over 5 seeds)
Primary: (C1) combination k = 16 vs descriptor inversion k = 16; (C2) k = 4; (C3) k = 1.
Secondary: combination vs zero-shot R for every k; combination vs hybrid for every k; ROBUSTNESS for C1-C3.
Reading: "better" if the interval's upper bound < 0, "worse" if the lower bound > 0, else "not distinguishable".
The library is said to add accuracy on top of inversion only if C1 and C2 are "better".

## Runs and failure rules
Seeds 148021-148025, all 41 folds, one job per seed, started after the hybrid jobs finish. Code:
msal_ext/combo.py (imports msal_ext.hyb unchanged). Stage-1 anchor mismatch voids a fold; NaN: retrain once.
