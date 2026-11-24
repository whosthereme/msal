# PLAN_LODO_DIM: does descriptor width explain why the harmonic descriptor adapts better?

Written 2026-11-15 at the user's request ("跑维度检验，给坐标描述子补随机列"). Hashed on ws2 before the first run.

## What is known (PLAN_LODO_ADAPT.md, five seeds, leave-one-direction-out, MSAL's head)
Zero-shot the six-column coordinate descriptor (xy6) had 62.6% lower sampled-chunk error than the 21-column
harmonic one (sat). After MSAL's adaptation the order reversed: xy6 +7.0% (k=1) and +6.9% (k=16) in sampled-chunk
error, flow not distinguishable. MSAL lowered sat's error by 69.7% and xy6's by 13.4% (k=1). Inversion moved the sat
descriptor by 3.0 on average and the xy6 descriptor by 1.4. Post hoc idea: 21 entries give inversion more freedom.

## Arms (all MSAL's head, the adaptation of PLAN_LODO_ADAPT.md unchanged)
- lodo_xy6: the coordinate descriptor, rerun.
- lodo_xy6r: xy6 plus 15 random columns, N(0, 0.7^2) (the coordinate columns' SD is about 0.7), drawn once per
  task id from a fixed generator (seed 20261114). These columns vary across source tasks, so their weights train.
- lodo_xy6z: xy6 plus 15 columns that are zero for every task. Their weights get no gradient and keep a decayed
  random initialization, as do the harmonic descriptor's held-direction entries; they add only inversion freedom.
Both variants have 21 columns, like sat. Runner: local_run/tc_lodo_dim.py (wraps tc_lodo_adapt.py unchanged).

## Runs
The 80 (seed, direction) units of PLAN_LODO_ADAPT.md, three arms per unit in one process, each unit on the same
machine type as in that study (ws2: 36 units; ws3 / ws4, which are bit-identical to each other: 33; this computer:
11), so that the xy6 rerun can reproduce the stored xy6 records bit for bit and the stored sat records become exact
pairs. Every xy6 record is checked for bit identity with the stored one (equality only); a unit that fails is excluded
from comparisons with sat and reported.

## Readings (draft's rule: per seed mean over the 41 tasks of the paired log ratio, t interval over seeds, df 4;
bootstrap over the 16 directions as a view)
Primary, complete method, sampled-chunk error at k = 1 and 16:
- W1 xy6r vs xy6; W2 xy6z vs xy6;
- W3 xy6r vs sat; W4 xy6z vs sat (stored sat, pairs whose xy6 was bit-identical).
Secondary: the same in flow MSE; zero-shot (both errors) of xy6r and xy6z vs xy6; method vs own zero-shot for each
arm; inversion alone; descriptor shift.
Rule fixed now, for each variant separately: "width explains the reversal" if the variant is lower than xy6 (W1/W2)
at both k and not higher than sat (W3/W4) at both k; "width explains part of it" if lower than xy6 at both k but
higher than sat at either; "not supported" otherwise.

Whatever comes out goes into the supplement, and into the main paper where it changes a claim.

## Limits
Offline, one skill and its 16-direction grid. A wider descriptor also changes the head's initialization; that
confound is shared with any width change and is not separated here.
