# The trunk-modulated head on LIBERO40 — written 2026-11-14 (time of first hash recorded on ws2), before any run

## Why
On the development family the trunk-modulated head (`tfilm_nl`: the draft's modulation applied to the trunk's
first activation, readout without task input) beat the draft's head (`film_nl`) zero-shot and with the complete
method (inversion + residual code), over five seeds. The draft's confirmation family is LIBERO40 (40 tasks, human
demonstrations, 65-column bag-of-words descriptor); every LIBERO result so far uses `film_nl`. The user asked for
the LIBERO rerun with the trunk-modulated head, to decide whether it can become the paper's main method.

## What is known when this is written
The stored LIBERO40 study (`runs/l40`, five seeds, 200 folds, flow MSE only): `film_nl` zero-shot, inversion,
complete method, and the zero-shot references `plain_nl` and `film`. All development-family results of
`tfilm_nl`. No score of `tfilm_nl` on LIBERO40 exists, and no sampled-chunk error of any head on LIBERO40.

## Runs
`tc_l40.py`: per fold (held task, seed), in one process on one machine, two arms:
- `l40_film_nl`: the stored fold (msal_ext/l40.py: stage 1, hyb.train_library, hyb.invert_descriptor,
  weighted_ls_code), calls unchanged, without that file's two reference trainings (`plain_nl`, `film`).
- `l40_tfilm_nl`: the same stages on the trunk-modulated head with the word descriptor, with the two changes
  fixed in PLAN_ADAPT_TRUNK.md (atoms read the trunk activation at the descriptor in use; inversion
  differentiates through the trunk modulation), functions imported from `tc_combo.py` unchanged.
Seeds 148021-148025, the 40 tasks, k in {1, 2, 4, 8, 16}, 12,800 updates, all seven action channels, stored
settings. Scores: flow MSE on validation and test rows (8 paired draws) for zero-shot, inversion and the complete
method at every k; sampled-chunk error (msal_ext.ideal.action_error, 10 Euler steps, 8 draws) on validation
rows for zero-shot, and for inversion and the complete method at k = 1 and 16. 200 fold pairs.
Machines: ws2 GPU 1 only, through `upgrades/slots.sh` (three processes inside one `gpu_run train`); ws3 / ws4
(two lanes each); this computer (one process). One queue of (seed, task) claims; a pair's machine is decided by
the queue. The LIBERO feature shards are copied from ws2 to the other machines and checked by SHA-256.

## Anchor (ws2)
Before the queue starts, ws2 runs seed 148021, task 1000. Its `l40_film_nl` arm must reproduce the stored fold
bit for bit (zero-shot, inversion and complete method, validation and test rows, every k). If it does not, the
cause is found before anything else runs, and the outcome is reported. Every ws2 fold is compared with its
stored fold in the analysis and the number of bit-identical folds reported.

## Reading (fixed now)
The draft's rule: per seed, the mean over the 40 tasks of the paired log ratio; the mean over seeds as a
percentage with a 95% t interval (df 4); folds won. Better or worse only if the interval excludes zero.
- Primary: B1 complete method on `tfilm_nl` vs on `film_nl`, flow MSE (validation) at k = 1 and k = 16; B2 the same
  in sampled-chunk error.
- Secondary: zero-shot `tfilm_nl` vs `film_nl` (both errors); inversion alone, trunk vs head (flow MSE, k = 1, 16);
  residual code beyond inversion on each head (flow MSE at k = 1, 4, 16, the stored study's L1, L2, L4;
  sampled-chunk error at k = 1, 16); the complete method against its own zero-shot head on each head (both
  errors, k = 1, 16); zero-shot `tfilm_nl` against the stored `plain_nl` and stored `film` (flow MSE; chained
  through `film_nl`: same-machine log ratio plus the stored ws2 log ratio of the same fold).
- Second view: per task the mean log ratio over seeds; 95% percentile bootstrap over the 40 tasks (20,000
  resamples, generator seed 0). Descriptive: per-seed values; B1 at k = 2, 4, 8; test rows; B1 and B2 and the
  zero-shot comparison on the folds whose instruction words all occur in training and on the others (the
  draft's post hoc split; Proposition 5); retrained `film_nl` against stored scores per machine.
If the work ends with seeds incomplete, readings are given on the complete seeds with their number.
Every comparison is reported whatever the outcome.
Consequence fixed now: the trunk-modulated head is proposed to the user as the paper's main method only if B1
and B2 are not worse at both k; otherwise the paper keeps `film_nl` and reports this study as it came out.

## Limits
Offline. No closed loop: the closed-loop results of the draft stay those of `film_nl` whatever this study shows.

## Amendment 1 (after the anchor, before the queue starts; no other full-length LIBERO fold exists)
Anchor outcome: on ws2 the `l40_film_nl` arm of tc_l40.py reproduced the stored fold of seed 148021, task 1000,
bit for bit (zero-shot by both scorers, inversion and complete method at every k, validation and test rows). The
anchor pair counts as that fold's result (output `l40/out_ws2_anchor`), and its claim is entered by hand. Its
`l40_tfilm_nl` scores were written by the same process and have not been read. Data copies: the 40 shards on
ws3 / ws4 and on this computer have the SHA-256 digests of ws2. Measured on ws2 alone: 81 s per head.

## Amendment 2 (during the run; no score has been read)
With three processes on ws2 GPU 1, one ran out of GPU memory while scoring (each peaks near 7.9 GB on the 24 GB
card): the pair seed 148021, task 1003 stopped in its first arm and wrote no record. The third ws2 lane was
stopped after its running pair (three lanes also gave no more throughput than one process alone); ws2 continues
with two lanes. The claim of the failed pair was released so that another lane reruns it. Nothing else changes.
