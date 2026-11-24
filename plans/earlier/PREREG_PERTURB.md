# Registered study: closed-loop adaptation under perturbed start states (PREREG_PERTURB.md)

Written 2026-11-02 before any perturbed closed-loop episode. Chosen by the user ("c") after the upgrade pilots
(experiments/upgrades/SUMMARY.md) showed that replaying the nearest demonstrated chunk beats every code-based adapter
on LIBERO's own start states, which come from the same distribution as the demonstrations.

## Question
Does the advantage of demonstration replay over the adapted head shrink when the scene departs from the demonstrated
start states, i.e. does the head's generalization matter there?

## Perturbation (experiments/perturb/perturb_lib.py, precompute.py)
After LIBERO loads fixed initial state e of a held task, every movable object (env.objects_dict; fixtures stay) is
shifted in the table plane by an offset drawn uniformly from a disc of radius r; objects closer than 5 cm in the
plane (stacked or nested) form one group and share one offset. Levels: L0 unperturbed, L1 r = 4 cm, L2 r = 8 cm.
Calibration (calib.json, no policy involved): object positions vary by about 1-5 cm (range) across LIBERO's 50
initial states, so L1 is at the edge of the natural variation and L2 about twice it. A draw is accepted if, after
200 passive physics steps, every object's in-plane drift differs from the unperturbed scene's settling drift by at
most 1.5 cm, its height from the unperturbed resting height by at most 2 cm, its tilt by at most 20 degrees, no
two groups end up closer than min(their original distance, 10 cm) (shifts create no new contacts), and the task is
not already done; otherwise redrawn (generator seeded by
(20261102, task, episode, level), at most 300 tries). The accepted offsets for every (task, episode 0-49, level) are
fixed in perturbations.json BEFORE any policy runs; an episode without an accepted draw is excluded at that level for
every policy. The rollout gets the perturbed scene before settling, exactly as it gets the unperturbed one.

## Policies (unchanged code and artifacts from the earlier studies; launcher run_perturbed.py patches only
## LIBERO's set_init_state to apply the stored offsets)
| policy | what | source |
|---|---|---|
| msal_k1, msal_k16 | MSAL (registered method) | cl_rollout.py, runs/cl_adapt/held{h}.pt |
| bf_k16 | MSAL's 81 numbers refit to the sampled chunk (behavfit pilot) | bf_rollout.py, upgrades/behavfit/art/held{h}.pt |
| ret_replay_k1, ret_replay_k16 | nearest demonstrated chunk, no head (retrieval pilot) | ret_rollout.py, banks bank_h{h}_k{k}.npz, L2 |
| ft_k16 | fine-tuned head (reference, needs training) | cl_rollout.py, held{h}.pt |
| seen | head trained on the task (reference) | cl_rollout.py, seen.pt |
8 held tasks (1019 1025 1000 1020 1023 1030 1033 1037), episodes 0-49, N_EXEC 16, 10 flow steps, policy noise seed
1_000_003*h + e and init state e as before, so every perturbed episode is paired with the same episode at L0 and
across policies. L1 and L2: 7 policies x 8 tasks x 50 episodes x 2 levels = 5,600 episodes (minus exclusions). L0:
the existing episodes of these policies (Stage B; retrieval pilot and confirmation; behavfit pilot), not re-run.

## Comparisons (analysis/cl_perturb.py; two-level bootstrap over tasks then episodes, 10,000 resamples,
## seed 20261102; better/worse only if the 95% interval excludes zero; episodes excluded at a level are dropped from
## every contrast that uses that level)
Primary (difference in differences, paired per episode):
- P1: [(ret_replay_k16 - msal_k16) at L2] - [(ret_replay_k16 - msal_k16) at L0]. Negative = replay's advantage
  over the head shrinks under perturbation.
- P2: the same with ret_replay_k1 and msal_k1.
Secondary:
- S1-S2: P1 and P2 at L1.
- S3: ret_replay_k16 - msal_k16 at L2 (does the head overtake replay?); S4: ret_replay_k1 - msal_k1 at L2.
- S5: bf_k16 - ret_replay_k16 at L2, and its difference in differences vs L0.
- S6: change from L0 to L2 for each policy (drop in success).
Descriptive: success per policy, level and task; ft_k16 and seen as references (do perturbed tasks remain solvable by
a head that saw the task?); fraction of excluded episodes per task and level.

## Reading
The head's generalization is supported only if P1 or P2 is negative with its interval below zero. A crossover (S3 or
S4 below zero) would be stronger evidence. If P1 and P2 are not distinguishable from zero, replay's advantage is not
shown to depend on matched start states at these perturbation sizes.

## Execution notes
(appended below with times)
- 2026-11-02 ~11:40, before any policy episode: the first table (minimum group separation 5 cm) produced, in a visual
  check (snapshot.py, task 1030 episode 0 at L2), a bowl resting partly on the ramekin. The separation rule was
  changed to "no pair closer than min(original distance, 10 cm)" (stated above) and the table regenerated. No
  policy had been run on either table.
- 12:05: regenerated table perturbations.json (sha256 c4470bcc915563120f39ad4f422cfbc7b62891294d64f7ab5c07b53a91acd498):
  all 50 episodes accepted at both levels for every task except 1030 at L2 (48/50; episodes 14 and 27 excluded);
  re-applying stored offsets reproduces the accepted scene exactly (max error 0). Snapshots (smoke/snap2_*.png) show
  no new contacts. Smoke test (smoke.sh, task 1019 episode 0 at L2, one per rollout script; not part of the results):
  all three scripts ran, probe checks passed. Rollout scripts in perturb/scripts are byte-identical (md5) to those of
  the earlier studies. Driver started 12:07:53 (4 rollout slots, jobs.txt: level 2 first, then level 1).
- 14:34: driver done, 5,586 episodes (= 5,600 - 2 excluded episodes x 7 policies), no retries or failed jobs. Every
  file holds exactly the accepted episodes of its (task, level), N_EXEC 16, 10 flow steps, probe errors < 1e-5.
  analysis/cl_perturb.py -> analysis/cl_perturb.json. Primary: P1 -34.5 pp [-51.8, -15.3], P2 -32.2 [-51.5, -12.3],
  both negative. S3 replay16 - msal16 at L2 +17.3 [+8.3, +28.4] (no crossover); S4 +4.9 [-4.2, +16.2] n.d.
  Reading note: every demonstration-fitted policy (replay, fine-tuning, the head trained on the task) lost 35-38
  points from L0 to L2; MSAL lost 4-6 (n.d.) from a much lower level, so part of the difference in differences is a
  floor effect.
- 2026-11-02 (found after the study): the rollout configuration copied from the substrate builder
  (expert0_processor/config.json "vlm_model_name", policy_preprocessor.json "tokenizer_name") pointed to the SmolVLM2
  snapshot inside another user's folder (/home/ubuntu/other_workspace/paper6/models/...), so each rollout process READ
  that snapshot's config/processor/tokenizer files (no weights; nothing written). All copies were repointed to the
  identical snapshot (same revision 7b375e1b...) in our own cache; originals kept as *.orig. Check: task 1019 msal_k16
  episodes 0-1 re-run with the fixed configuration repeat their recorded outcomes exactly (success at 98 and 80
  steps), and an strace of the run shows no file opened under /home/ubuntu/other_workspace. No result changes.
