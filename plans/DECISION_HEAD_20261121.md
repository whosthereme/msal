# Decision record: MSAL's head becomes the trunk-modulated head (2026-11-21)

Written before any text of the manuscript is changed for this decision. It is a record of a decision by the author,
not a study plan; no run depends on it.

## The decision
On 2026-11-21 the author decided that the method presented in the paper as MSAL modulates the first activation of the
trunk (`tfilm_nl`, `tc_arms.TrunkFiLMField`, "trunk modulation") instead of the trunk output h (`film_nl`, "modulation
at h", the head the paper has called MSAL's head so far). The variant that modulates h is kept in the paper under the
name MSAL-h, with every result obtained on it.

## What was registered against it
Three plans fixed consequences that this decision departs from. The departure is stated in the paper.
1. PLAN_L40_TRUNK (2026-11-14): "the trunk-modulated head is proposed to the user as the paper's main method only if
   B1 and B2 are not worse at both k; otherwise the paper keeps `film_nl` and reports this study as it came out."
   Outcome: B2 (sampled-chunk error after adaptation on LIBERO40) +3.6% at k = 1 and +3.7% at k = 16, both "higher";
   B1 (flow) not distinguishable. By the rule the paper keeps `film_nl`.
2. PLAN_CL_TRUNK (2026-11-14): "the paper's method stays film_nl (decided offline by PLAN_L40_TRUNK.md)". Outcome on the
   eight held LIBERO tasks: trunk against h, k = 1 -5.5 points (-11.5 to -0.0, "worse"), k = 16 -3.0 (not
   distinguishable).
3. PLAN_CL_TFILM (2026-11-20): only if H7 (trunk modulation against input concatenation in the cluttered family) read
   "better" might the manuscript say that a conditioning of ours led concatenation in closed loop. Outcome: 16.2%
   against 16.8%, -0.6 points (-4.2 to +3.2), not distinguishable. This consequence is kept: the paper does not say
   that MSAL led concatenation in closed loop.

## What the decision rests on (all known when it was taken)
- Development placement family, offline, five seeds: trunk against h, zero-shot sampled-chunk error -15.6% (-20.1 to
  -10.8), flow -6.2%; complete method -20.3% and -20.8% at k = 1 and 16 (registered, 5SEED and ADAPT_TRUNK5).
- Against input concatenation on that family, read post hoc from stored records on 2026-11-20: -12.0% (-14.1 to -9.9).
- Cluttered closed-loop family (registered, CL_TFILM): trunk 16.2% against h 10.9%, +5.3 points (+2.1 to +8.4);
  against concatenation not distinguishable.
- Known against it: every LIBERO result above, and LIBERO40 zero-shot, trunk against h: flow +44.9%, sampled chunk
  +8.7%.
The decision was taken after all of these results were seen, and after input concatenation was seen to lead the head
that modulates h on the placement tasks and in closed loop. It is a choice made with knowledge of the outcomes.

## What the paper must state
- That MSAL-h was the method when most studies were planned and run, which studies used it, and that MSAL's present
  head was fixed after the results were seen.
- The rule of PLAN_L40_TRUNK, its outcome, and that the choice of head departs from it.
- For every comparison in which both heads were run, both results; in particular that on LIBERO the trunk head was
  worse than MSAL-h zero-shot, after adaptation and in closed loop.
- That in closed loop MSAL's head was not distinguishable from input concatenation.
- Which studies were not run with MSAL's present head: the harmonic descriptor, the direction identifier, the
  projection control and the descriptor-width studies on held directions; the text embedding on LIBERO; the harmonic
  descriptor, the trained targets and adaptation in the cluttered family; fine-tuning references, source coverage
  and timing.

## Files
The manuscript before this decision stays in `manuscript_v3` and `submission`; the changed manuscript is written to
`manuscript_v4`.
