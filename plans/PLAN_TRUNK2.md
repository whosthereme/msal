# PLAN_TRUNK2: the studies that were run only on MSAL-h, repeated on MSAL's head (trunk modulation)

Written 2026-11-21 (evening), at the author's request ("保留并补跑"), after the decision of the same day that MSAL's
head is the trunk-modulated head (DECISION_HEAD_20261121.md). The manuscript (manuscript_v4) says in its limitations
which studies exist only for MSAL-h, the variant that modulates the trunk output h. This plan runs them on MSAL's
head. Hashed on ws2 before any head of this plan is trained; changes after hashing are written below as dated
amendments before the runs they affect.

MSAL's head everywhere below: `tc_arms.TrunkFiLMField`'s forward pass (the first trunk activation is multiplied by
1 + Gamma d and shifted by Delta d, Gamma and Delta bias-free with N(0, 0.02^2) initialization, then the second trunk
layer and the readout without task input), trained by the codebase's loop, 12,800 steps, unchanged.

## What is known when this is written
For MSAL-h: every result of the studies named below (manuscript_v4, supplement S7 and S9). For MSAL's head: the
development-family zero-shot and adaptation studies, the coordinates on held directions (-69.0% against no
description), LIBERO40 with words (zero-shot worse than no description, +5.4%; zeroing unseen-word entries lowers its
flow error by 25%), closed loop on LIBERO and in the cluttered family with coordinates (16.2%). None of the runs
below exists for MSAL's head.

## Part A: what the description encodes, held directions (development family, zero-shot, stage 1 only)
Per (seed, direction) unit, in one process on one machine, six heads trained on the tasks of the other 15
directions (seeds 148021-148025, 16 directions, 80 units, 480 trainings):
- `lodo_plain`: readout without task input (as stored).
- `lodo_tsat`: MSAL's head with the 21-column harmonic descriptor.
- `lodo_txy6`: MSAL's head with the six-column coordinate descriptor (the stored arm, class unchanged).
- `lodo_txy6id05`, `lodo_txy6id1`, `lodo_txy6id2`: MSAL's head with coordinates plus alpha times a one-hot of the
  target direction, alpha = 0.5, 1, 2 (the descriptors of PLAN_SPAN_PRED).
Scores per held task as in the stored studies: flow MSE on SELECTION and ROBUSTNESS rows, sampled-chunk error on
SELECTION rows (10 Euler steps, 8 draws). `lodo_tsat` and `lodo_txy6id2` are also scored on P_S d, the held
descriptor with its out-of-span part deleted (projector of PLAN_PROJ, tolerance 1e-9).
Runner `tc_lodo_t.py`. Anchor: where a unit runs on the machine type that holds the stored `lodo_txy6` or
`lodo_plain` record of the same seed (A800 server: seeds 148024, 148025), the new record is compared bit for bit with
the stored one and the count reported.

Readings (the main paper's rule: per seed the mean over the 41 held tasks of the paired log ratio, mean over seeds
with a 95% t interval, df 4; better or worse only if the interval excludes zero; bootstrap over the 16 directions as
second view, 20,000 resamples, generator seed 0). Sampled chunk is primary, flow is given beside it.
- A1 harmonic against no description. Prediction: not lower (the description gives no gain on a direction never
  trained, 19% of it lying in the span).
- A2 coordinates against no description (replication of the stored -69.0%) and A3 coordinates against harmonic.
  Prediction: both lower.
- A4 identifier at alpha = 2 against coordinates alone (primary of this part). Prediction: higher, and the three
  alphas in the order 0.5 < 1 < 2.
- A5 in-span part against the whole descriptor: harmonic, flow MSE (J1'); identifier at alpha = 2, sampled chunk
  (J4'). Prediction: both lower. Without reading rule: in-span harmonic against no description and against
  coordinates, in-span identifier against coordinates.
- Descriptive: each arm against the stored MSAL-h record of the same descriptor, seed and task (other process, for
  most units another machine).

## Part B: a pretrained text embedding as description (LIBERO40, zero-shot)
The descriptor and embedding file of PLAN_TEXT, unchanged (960 numbers, standardized per fold; span share 0.80 on
average). MSAL's head trained with it by the unchanged loop, 200 units (five seeds, 40 held tasks), scored on the
whole descriptor, on its in-span part and on the constant-only descriptor, flow MSE on validation and test rows and
sampled-chunk error on validation rows. Runner `tc_text_t.py`. Paired references, stored: MSAL's head with the word
descriptor (PLAN_L40_TRUNK records) and the readout without task input (PLAN_TIN records) of the same seed and task.
Units are assigned to the machine type of the stored word-descriptor record where that is ws2, the others to the
A800 server; readings are given on all pairs and on same-machine pairs.
- T1' embedding against no description. Prediction: not lower in sampled-chunk error (the word descriptor raised
  this head's error).
- T2' embedding against words on the same head. No prediction.
- T3' in-span part against the whole descriptor. Prediction: not higher.
- T4' whole descriptor against the constant-only descriptor. No prediction.

## Part C: the cluttered closed-loop family (ws2, folds, seeds, scenes, episodes 100-109 and flow-noise seeds of
## PLAN_CL_COORD, so that every episode is paired with the stored policies)
- C1, harmonic descriptor: MSAL's head with the 20-column harmonic descriptor of this layout (`lc_heads2.h20`, span
  share 0.18 at a held target), twelve heads, policy J0 on the held targets, 1,200 episodes.
- C2, trained targets: the stored coordinate heads of PLAN_CL_TFILM (K), policy K0 on the trained targets of two
  directions per fold (the targets of PLAN_CL_COORD's S3), 600 episodes.
- C3, one demonstration: for each stored K head (weights loaded from its artifact and checked against its stored
  probe), the residual library is trained on the frozen head (atoms reading the trunk output at each task's own
  descriptor) and each held target is adapted from its first demonstration (inversion through the trunk, closed-form
  code), the functions of PLAN_ADAPT_TRUNK unchanged; policy K1 on the held targets, 1,200 episodes.
Scripts `libcoord/lc_heads4.py`, `lc_rollout4.py`. Every rollout process rebuilds its head and stops if it disagrees
with the stored probe by more than 1e-3.
Readings: paired differences of success in points, two-level bootstrap (targets, then (seed, episode) units), 10,000
resamples, generator seed [20261122, index], better or worse only if the 95% interval excludes zero.
- H10 J0 against K0 (harmonic against coordinates on MSAL's head). Prediction: worse.
- H11 J0 against N0 (no description). Prediction: not better.
- H12 K1 against K0. Prediction: not distinguishable (as for MSAL-h).
- H13 K0 at held against trained targets (unpaired over targets, as S3). No prediction.
Secondary: median final distance for H10 to H12; offline flow MSE and sampled-chunk error of the J heads, and of K
after adaptation, on the held targets.

## Part D: CPU time of adaptation
One development fold (held task = first source task, seed 148021): MSAL's head and its library trained on the GPU,
then, in a CPU-only process with four threads, the two adaptation steps (inversion through the trunk, closed-form
code) timed three times at k = 1, 4, 16 (median), as `msal_ext/bench_cpu.py` does for MSAL-h; MSAL-h is timed again
in the same run as a control. Descriptive.

## Consequences fixed now
Every reading is reported whatever it shows. Where a statement of manuscript_v4 rests on MSAL-h only and this plan
gives the corresponding result for MSAL's head, the manuscript gives MSAL's result in the main text and keeps
MSAL-h's beside it or in the supplement. If a prediction above fails on MSAL's head, the manuscript says so and its
statement of the second condition is limited to what held on both heads. The limitation "not run with MSAL's head"
is removed only for the studies that completed.

## Machines and cost
A800 server (ws4): Parts A and B on four lanes. ws2 GPU 1 only, through `upgrades/slots.sh`: Part C (training slot
and rollout slots), Part D, then its share of Parts B and A. About 700 trainings and 3,000 rollout episodes; one
night. Units are claimed on the server's shared home (one claim directory per part).

## Amendment 1 (2026-11-21, 21:25; no score of this plan has been read)
The first start of Part C on ws2 (21:18) was stopped at 21:22 because of an error in the driver `lc4_run.sh`: the
slot wrapper `gpu_run` overwrites the shell variable `f`, which the driver used for the fold, so the second training
of a fold was called with a wrong fold number (24066). That call was stopped a few seconds after its start, before
any training step and before it wrote an artifact. Kept, because they were produced with correct arguments: the
adapted artifact of fold 0, seed 148021, and three rollout episodes of that fold (K1 on a held target, K0 on a
trained target), which the restarted lanes continue. The driver now names the fold `fd`. Nothing else changes.
Parts A and B on the A800 server were not affected (their lanes do not call `gpu_run`).

## Amendment 2 (2026-11-22, 17:34; no comparison of this plan has been computed or read)
Two faults of the night run, found at 17:30 when checking counts.
1. Part B ran on the wrong machine. The ws2 lanes started Part B at 23:39 while the four server lanes were still on
   Part A; the claim script let them take the ws2 list (93 units) and then the server's list, so all 200 units were
   claimed by ws2 and none ran on the A800 server. 157 finished (93 of the ws2 list, 64 of the server's list); 43 of
   the server's list failed when loading data onto GPU 1, which other processes had filled (CUDA out of memory, exit 1;
   one exit 137). Fix, restoring the assignment fixed above: all 107 units of the server's list are run on the A800
   server (four lanes, unchanged runner, new claim directory `claims_trunk2b`, output `outB2_*`). The readings of Part
   B use the 93 ws2 units from ws2 and the 107 server units from the server, as planned. The 64 ws2 records of
   server-list units are kept and reported only as a descriptive cross-machine repeat. While reading the error logs
   I saw the per-unit printout of some ws2 units (raw scores of single units, no paired reference); no comparison was
   computed.
2. Part C stopped one episode short of a file. The rollout K0, trained targets, seed 148023, fold 3 hung after
   episode 107 of its first target (18 of 40 episodes written, last write 00:55) and was stopped by PID at 17:33; the
   lane's retry loop restarts it, and the runner skips the episodes already written. Nothing else changes.

## Amendment 3 (2026-11-22, 19:28)
Correction of Amendment 2 and the last 13 units of Part B. Amendment 2 said that the 43 units that failed on ws2 all
belonged to the server's list. That was wrong (my comparison of the two lists was faulty): 30 belonged to the server's
list and 13 to ws2's list (seed 148021: tasks 1013, 1036, 1038; seed 148022: 1006, 1007, 1019, 1022, 1023, 1033, 1034,
1038, 1039; seed 148023: 1013). So ws2 finished 80 units of its own list and 77 of the server's. The server's 107 units
were rerun on the server as Amendment 2 says (all finished). The 13 missing units of ws2's list are now run on ws2
(GPU 1, through the slot wrapper, one process at a time, unchanged runner, output `outB3_ws2`).
Disclosure: the mistake was found only when the analysis of Part B was run after the pull, and that analysis printed
the Part B readings on the two seeds that were complete (148024 and 148025, 80 units, all from the server). These
readings were therefore seen before the plan's 200 units were complete. Nothing of the analysis is changed by this:
the unit assignment and every reading rule were fixed before (plan and Amendment 2), and the reported Part B readings
will be those on all 200 units. The two-seed printout is kept as `results/trunk2/trunk2_ab_partialB.txt`.
