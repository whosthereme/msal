# Method upgrade: nonlinear shared atom library (SAL-NL) — registered 2026-10-24 ~00:45 CST, before any run

## Provenance (honest)
Chosen after the five-seed N1/N2 results on these 41 LOTO folds (FiLM matched or beat the linear library) and
after the closed-loop seed-1000 results. The hypothesis comes from earlier work on a different surface:
E37.1/E37.2 (v4 pool, seen tasks): the published library is exactly a rank-K weight factorization, because its
base and atom heads are linear maps of the same trunk activation; K per-atom MLPs (512 -> 92 -> SiLU -> 112)
were 25% more accurate at parameter parity, and the same head with the SiLU deleted tied the linear library.
E38.2 (breadth experiment): with nonlinear atoms the held-task code helped about twice as much. SAL-NL has never
been run on the v5 LOTO held-task surface. The 41 folds are development folds (already used by N1/N2); this is a
registered test on them, not a confirmation on fresh tasks. Reserved targets are not used (no sealed rows).

## Arms (language-blind s_V, descriptor phi = saturated_v5, K = 16 where applicable)
Everything else is the N1 harness unchanged: trunk H, batches, noise draws, 12,800 AdamW updates, scoring
(msal_exp.run_loto / msal_exp.arms imported unchanged; extension in msal_ext/nl.py, runner msal_ext/run_nl.py).
- sal_nl: the published SAL (StructuredCodeFieldV5) with atom_out replaced by NonlinearAtomHead (16 MLPs
  512 -> 92 -> SiLU -> 112, fan-in init; lib/nonlinear_atom_field.py, unchanged). Code z = W_c^T phi, both
  penalties (0.02 orthogonality, 0.001 gain-weighted group sparsity, gain detached), exactly as SAL.
  The field stays exactly affine in z, so it is scored like SAL (phi, nearest-source, mean, target-supervised codes).
  Task-conditioned parameters: 921,792 (atoms) + 336 (W_c) = 922,128 (linear SAL: 919,632).
- sal_nl_noreg: sal_nl without the two penalties.
- film_nl (capacity control): FiLM as in N1, then its linear head plus a readout MLP 512 -> 1472 -> SiLU -> 112
  (920,112 parameters; 1472 = 16 x 92 nonlinear units, same init scheme as the atom MLPs).
- plain_nl (capacity control): no task input; linear head plus the same readout MLP.
- moe_desc (mixture-of-experts control; added 00:55, still before any run, after the user noted that SAL-NL
  reads as "an encoder plus a bank of action experts"): the same 16 expert MLPs as sal_nl, mixed by a
  descriptor-routed softmax gate w = softmax(W_g phi) (W_g: 16 x 21, N(0, 0.02^2), so the initial gate is
  near uniform), v = B h_bar + sum_k w_k E_k(h); no penalties, no auxiliary load-balancing loss. Task-level
  routing from task features is the case the earlier ledger left untested (it tested routing by task index
  only). The field is not affine in a free code, so it is scored through the forward path with the held task's
  descriptor, like FiLM. Task-conditioned parameters: 921,792 + 336.
Stored N1 results at the same seeds and folds serve for sal, film and plain (identical batches and noise draws).
Anchor: arm sal run through run_nl at seed 148021 on the first fold must reproduce its stored N1 value
bit-exactly; otherwise the study is void until explained.

## Runs
Seeds 148021-148025, all 41 folds, SELECTION primary, ROBUSTNESS secondary. One job per (arm, seed): 25 jobs,
six workers (three per GPU), while the closed-loop study is paused.
Diagnostic (not a comparison): for every sal_nl fold, the relative residual of the field against the best single
affine map of the trunk activation at the mean source code (1,024 TRAIN rows), to show the arm uses its
nonlinearity (E37's V4 criterion: far above the float32 floor).

## Comparisons (paired log ratio per fold; mean per seed; 95% t interval over the 5 seeds, df = 4)
Primary: (1) sal_nl vs film; (2) sal_nl vs sal.
Capacity controls: (3) sal_nl vs film_nl; (4) sal_nl vs plain_nl.
Mixture-of-experts control: (8) sal_nl vs moe_desc (signed linear codes vs a softmax router over the same
experts).
Secondary: (5) sal_nl vs sal_nl_noreg; (6) within sal_nl, predicted code vs the better of the nearest-source and
mean codes; (7) ROBUSTNESS rows for (1)-(4); folds won per seed for all.
Reading per comparison: "better" if the interval's upper bound < 0, "worse" if the lower bound > 0, otherwise
"not distinguishable".
Method claim "the nonlinear library improves on FiLM because of its library structure" requires (1), (3) and (4)
all "better". If (1) is better but (3) or (4) is not, the gain is reported as readout capacity, not as a property
of the library. A claim that signed, descriptor-linear codes beat a mixture of the same experts requires (8)
"better"; otherwise the paper states that the two mix the experts equally well and that the library's
difference is its code-affine field (closed-form codes, reuse), not accuracy.
Every arm and comparison is reported whatever the outcome.

## Failure rules
A fold that produces NaN is retrained once with the same seed; a second failure is reported, not replaced.
No arm, setting or reading rule is changed after the first run starts. The registered N1/N2 arms and their
readings are not affected by this study.

## Addendum (2026-10-24 01:52, before any sal_nlb run): exploratory arm sal_nlb, added after an interim look
Interim look at about 01:50 (35 sal_nl folds over five seeds; 7 film_nl folds at seed 148021): sal_nl was ahead of
sal, film and plain on every fold (about -26%, -19%, -34%), but film_nl was ahead of sal_nl on 5 of 7 folds
(about +46% for sal_nl), apparently because sal_nl's only nonlinear capacity sits in code-weighted atoms while its
base map is linear. Hypothesis: a code-affine library needs task-independent nonlinear capacity in its base.
Arm sal_nlb: base = published zero-initialized linear map + readout MLP 512 -> 736 -> SiLU -> 112; atoms = 16 MLPs
512 -> 46 -> SiLU -> 112; code z = W_c^T phi; both penalties as SAL. Nonlinear width 736 + 16 x 46 = 1472 as
film_nl; parameters of base MLP + atoms + W_c about 922k (sal_nl 922,128; film_nl readout + FiLM 941,616).
Still exactly affine in z, scored like SAL (incl. the linear-map-residual diagnostic). Code: msal_ext/nlb.py,
runner msal_ext/run_nlb.py (nl.py and run_nl.py unchanged). Five jobs (seeds 148021-148025) fill the slots left
idle in the last round of the registered queue. Because it was designed after an interim look at these folds, it
is reported as exploratory whatever its outcome; no further variant will be added on these folds, and nothing in
the registered arms, comparisons or reading rules changes.
Comparisons (exploratory, same statistics): sal_nlb vs film_nl; sal_nlb vs sal_nl; sal_nlb vs plain_nl;
sal_nlb vs moe_desc; within sal_nlb, predicted vs better reuse code.
