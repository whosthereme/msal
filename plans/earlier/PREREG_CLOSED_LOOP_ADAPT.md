# Closed-loop test of MSAL adaptation on held-out LIBERO tasks — registered 2026-11-01, before any rollout

## Purpose
Every adaptation result in the paper is offline (held-task flow MSE). This study asks whether the offline gains
carry over to task success when the frozen-feature head drives the robot in LIBERO: does adapting the fixed head
to a held-out task from k demonstrations (MSAL: descriptor inversion + closed-form residual code, no weight
change) raise its success over the zero-shot head, and how does it compare with fine-tuning the head on the same
demonstrations? The user approved the study on 2026-11-01 ("do 2 and 3").

## Policy (fixed now)
- Observation -> feature: the frozen observation contract (smolvla_state_contract.raw_policy_observation, copied
  unchanged) and the frozen SmolVLA extractor used to build the LIBERO40 substrate: base weights (sha256
  7cd549ac..., my own cached copy of lerobot/smolvla_base; identical hash to the builder's), the builder's config and
  preprocessor (copied from the substrate's builder checkpoint; its adapter is not used), bfloat16 autocast,
  extract_task_independent_spatial_state_feature, rounded to float16 as in the substrate. LIBERO cameras 256x256.
- Head -> actions: the flow field is integrated from Gaussian noise at tau = 1 to tau = 0 with N Euler steps to a
  16-step chunk of raw LIBERO actions (the substrate's actions are raw RLDS actions), clipped to [-1, 1]; the
  first N_EXEC actions are executed, then the policy replans. Noise seeds depend only on (task, episode), so all
  policies of a task see identical initial states and identical noise streams.
- Episodes: LIBERO's fixed initial states (episode e uses init state e), 10 settling steps with the gripper open,
  maximum steps 220 / 280 / 300 / 520 (spatial / object / goal / long), success = the environment's done flag.
- Heads are rebuilt from saved state dicts and must reproduce the saving process's outputs on a stored probe
  (max abs difference <= 1e-3), otherwise the run aborts.

## Stage A — feasibility gate (seen tasks; no held-out task is involved)
R_all = l40_film_nl trained on all 40 tasks (unchanged loop, seed 148021), conditioned on each task's own
descriptor.
A1 (settings): N_EXEC in {8, 16} with N = 10, on 8 tasks (two per suite: the first two task ids of each suite)
x 10 episodes (init states 0-9); the setting with the higher success is kept (ties: 8).
A2 (gate): R_all with the kept setting on all 40 tasks x 20 episodes (init states 0-19).
Gate: Stage B runs only if A2's mean success over the 40 tasks is >= 20% AND at least 8 tasks reach >= 40%.
If the gate fails, the study stops and is reported as: the frozen-feature head is not a usable closed-loop policy
in this setting, so its offline adaptation gains cannot be tested in closed loop here.

## Stage B — held-out tasks (only if the gate passes)
Held tasks: the 8 tasks with the highest A2 success (ties broken by lower task id), restricted to at most 3 per
suite (then the next eligible task is taken). Selection uses only seen-task success of R_all, never any held-out
result.
Per held task h (msal_ext/cl_heads.py): R_h and its residual library are retrained on the other 39 tasks
exactly as in the LIBERO40 study (seed 148021); MSAL and fine-tuning (whole head, Adam 1e-4, 100 steps for k = 1,
400 for k = 16: the development-selected setting) are run on h's first k TRAIN demonstrations with the seeds of the
registered studies. Anchors: R_h's zero-shot scores and the MSAL and fine-tuning selection scores must equal the
stored offline records bit-exactly; a mismatch is reported (it does not stop the study, since GPU kernels can
differ), and the rollout uses the heads actually trained here.
Policies per held task, 50 episodes each (init states 0-49), k in {1, 16}:
zero (R_h at d*), msal_k1, msal_k16, ft_k1, ft_k16; plus the reference R_all at h's descriptor ("seen").
That is 8 tasks x 6 policies x 50 = 2,400 episodes.

## Comparisons (paired on identical initial states and noise; success in percentage points)
Primary: (C1) msal_k16 vs zero; (C2) msal_k1 vs zero.
Secondary: (C3) msal_k16 vs ft_k16; (C4) msal_k1 vs ft_k1; (C5) ft_k16 vs zero; (C6) ft_k1 vs zero.
Descriptive: every policy against seen (how much of the seen-task success each adaptation recovers).
Statistics: mean difference over the 8 tasks x 50 episodes, 95% interval by a two-level bootstrap (tasks, then
episodes within task; 10,000 resamples, seed 20261101), as in analysis/closed_loop.py. Reading: "better" if the
interval lies above 0, "worse" if below, else "not distinguishable". One training seed; the unit of inference is
the episode within task, so the result does not generalize over training seeds, and the paper says so.
Every comparison is reported whatever the outcome.

## Compute and execution rules
GPU 1 only (user rule: never compete with other workloads on GPU 0). Head training on GPU 1 (~2 min per head);
rollouts in up to 4 parallel processes on GPU 1. Nothing is changed after the first Stage B rollout starts;
crashed episode runs resume after the last completed episode. Code: experiments/msal_ext/cl_heads.py
(offline env), experiments/closed_loop_adapt/cl_rollout.py (LeRobot env); outputs under
/home/ubuntu/work/manuscript6/runs/cl_adapt/.

## Execution notes (written before the first Stage A episode)
- 2026-11-01 13:05: another workload (four jobs) started on both GPUs (~20 GB each) and four CPU cores. The user
  decided (13:10) to share GPU 1 rather than wait. Consequences, none of which changes a computed value:
  head training keeps the 4.3 GB feature block in host memory and moves only each batch to the GPU
  (msal_ext/cl_heads.py HostFeatureCache; same CUDA noise generator, so batches are identical to
  DeviceCache); the rollout keeps SmolVLA's language model and action expert in host memory and puts only the
  modules the feature uses (vision encoder, connector, state projection) on GPU 1; rollouts run 3 in parallel.
- 13:19: seen head R_all trained (402 s). 13:21: smoke test, one episode of task 1030 outside the study
  (deleted): success in 79 steps, rebuilt head within 4.8e-6 of its probe. Stage A starts automatically after
  the CPU benchmark (runs/bench_cpu) finishes, so the two do not compete for cores.
- 13:40: the first Stage A launch was void: an argument-order bug in cl_stage_a.sh passed the task id as N_EXEC,
  so every rollout exited at argument parsing (0 episodes run; logs kept in runs/cl_adapt/void_1340_argbug/).
  Fixed, a guard added (the N_EXEC choice requires all 160 A1 episodes), and Stage A restarted at 13:41.
- 13:42: with 3 parallel rollouts GPU 1 ran out of memory next to the other workload (one A1 job failed at start-up,
  before any episode). Restarted with 2 parallel rollouts, PYTORCH_CUDA_ALLOC_CONF=expandable_segments, and up to 3
  attempts per job; a rollout resumes after its last completed episode, so episodes already written are kept.
- 14:00: A1 complete (160 episodes): N_EXEC=16 37.5%, N_EXEC=8 32.5% -> N_EXEC=16 kept by the registered rule.
- 14:13-14:15: the other workload grew (GPU 0 20.7 -> 22.1 GB) and A2 task 1003 failed three start-ups with GPU
  out-of-memory (no episode written). A2 continues with ONE rollout at a time (cl_stage_a2.sh), skipping episodes
  already written; the gate is evaluated only when all 800 A2 episodes exist.
- 14:22: the other workload left both GPUs (GPU 1: 3.5 GB used, all ours). Drivers rewritten around one helper
  (cl_common.sh): two rollouts at a time, each started only when GPU 1 has >= 3 GB free, up to 5 attempts, resume
  after the last written episode. A2 restarted 14:23 (tasks 1000-1006 complete); Stage B queued behind the gate.
- 14:56: A2 complete (800 episodes): 29.6% mean (long 16.5, goal 26.5, object 44.0, spatial 31.5), 14 tasks >= 40%
  -> gate PASS. Held tasks by the registered rule: 1019 1025 1000 1020 1023 1030 1033 1037 (gate.json).
- 15:05-16:24: eight held heads trained (453-793 s each, host-memory feature block). Every head reproduced all six
  stored anchors exactly (float32 film_nl selection score, zero-shot, MSAL k=1 and k=16, fine-tuning k=1 and k=16
  against the l40 and ft records). Stage B rollouts (2 at a time) started 16:24; no failure at start-up so far.
- 18:35: Stage B complete, 2400 of 2400 episodes; every file has episodes 0-49 once, the registered artifact,
  N_EXEC=16, 10 flow steps, probe error < 1e-5. Audit before analysis: the rollout head computes base + sum_k z_k
  Psi_k(h) with base = film_nl(h, d), the same expression weighted_mse scores offline (target eps - a); LIBERO has no
  dead action dimensions; each held id matched exactly one LIBERO task by instruction. analysis/cl_adapt.py ->
  analysis/cl_adapt.json. Post hoc (not registered): analysis/cl_offline_link.py relates offline error to success.
- 2026-11-02 (found after the study): the rollout configuration copied from the substrate builder
  (expert0_processor/config.json "vlm_model_name", policy_preprocessor.json "tokenizer_name") pointed to the SmolVLM2
  snapshot inside another user's folder (/home/ubuntu/other_workspace/paper6/models/...), so each rollout process READ
  that snapshot's config/processor/tokenizer files (no weights; nothing written). All copies were repointed to the
  identical snapshot (same revision 7b375e1b...) in our own cache; originals kept as *.orig. Check: task 1019 msal_k16
  episodes 0-1 re-run with the fixed configuration repeat their recorded outcomes exactly (success at 98 and 80
  steps), and an strace of the run shows no file opened under /home/ubuntu/other_workspace. No result changes.
