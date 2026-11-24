# PLAN_CL_TRUNK: the trunk-modulated head in closed loop on LIBERO

Written 2026-11-14 at the user's request ("跑新设计的闭环实验"). Hashed on ws2 before the first rollout.

## Scope
Of the two designs found after the method was fixed, only the trunk placement can be tested in closed loop: the
coordinate descriptor belongs to the development family, whose demonstrations come from a stored dataset with no
simulator in this codebase. This study rolls out the trunk-modulated head (tfilm_nl) on LIBERO, in the setting of
the registered closed-loop study (PREREG_CLOSED_LOOP_ADAPT.md) so that its episodes pair with the stored ones.

## What is known
Offline on LIBERO40 (PLAN_L40_TRUNK.md), after MSAL's adaptation the trunk head had 3.6% (k=1) and 3.7% (k=16)
higher sampled-chunk error than film_nl, flow error not distinguishable; zero-shot flow error +45%. Stored closed
loop (film_nl, eight held tasks, 50 episodes each): zero-shot 0.2%, MSAL k=1 16.0%, MSAL k=16 13.8% success.

## Heads
For each of the eight held tasks of the stored study (1000, 1019, 1020, 1023, 1025, 1030, 1033, 1037), seed 148021:
the trunk head and its residual library retrained on the other 39 tasks and adapted with MSAL at k = 1 and 16 by
local_run/cl_heads_t.py, which repeats tc_l40.run_fold's calls in order. Every offline record must equal the stored
PLAN_L40_TRUNK.md record of the same task (all eight were produced on ws2) bit for bit; a head that does not is
reported and not rolled out.

## Rollouts
trunkcond/cl/cl_rollout_t.py = cl_adapt/cl_rollout.py plus a rebuild of the trunk head, checked against the
artifact's probe outputs (max abs difference reported; > 1e-3 stops the run). Policies tzero, tmsal_k1, tmsal_k16;
episodes 0-49 per task; N_exec 16, 10 Euler steps, LIBERO's fixed initial states, ten settling steps, the suites'
step limits, the noise generator seeded per task and episode as in the stored study. 1,200 episodes, ws2 GPU 1,
through gpu_run (rollout slots; head training in the train slot).

Replay check, before the trunk rollouts: the stored film_nl artifact of task 1000, policy msal_k1, episodes 0-4,
through this script. If every replayed episode equals the stored one in success and steps, the stored film_nl
episodes are used as paired references. If any differs, film_nl's zero, msal_k1 and msal_k16 are rerun for all
eight tasks within this study and used instead (recorded as an amendment).

## Comparisons (two-level bootstrap as the stored study: tasks, then episodes within task; 10,000 resamples;
generator seed [20261115, index]; "better/worse" only if the interval excludes 0)
Primary: T1 tmsal_k16 vs film msal_k16; T2 tmsal_k1 vs film msal_k1.
Secondary: T3 tzero vs film zero; T4 tmsal_k16 vs tzero; T5 tmsal_k1 vs tzero.
Descriptive: trunk policies vs the stored fine-tuned and seen-task heads; per-task success.

Consequence fixed now: the paper's method stays film_nl (decided offline by PLAN_L40_TRUNK.md). The closed-loop
result is reported in the supplement and, in the main paper's closed-loop section, in one sentence; if T1 or T2 reads
"better", the paper states that closed loop favoured the trunk head despite its offline result.
