# PLAN_CL_COORD: zero-shot placement at unseen coordinates, closed loop, cluttered LIBERO scenes

Written 2026-11-16 at the user's request ("在 LIBERO 里搭一个带坐标描述的放置任务族，用来做闭环检验"; the scene "比较杂乱遮挡较多
比较复杂", with "许多不同的模型"). Hashed on ws2 before any head is trained or any policy is rolled out. Changes after
hashing are written below as dated amendments before the runs they affect.

## Question
The paper's claim is interpretable zero-shot generalization with all weights frozen: a new task is reached by
writing its descriptor, not by training. Offline, the coordinate descriptor reached unseen directions with 62% lower
error than the harmonic one. The stored closed-loop study (LIBERO, word descriptor) found zero-shot success near 0.
This study asks, in closed loop: given only a target's coordinates, does the frozen MSAL head place an object at
targets whose direction it never trained on?

## Task family (built and piloted before this plan; code in trunkcond/libcoord)
- 40 targets on the table around the start point (-0.05, 0): ring radii 0.10 / 0.16 / 0.22 m; 8 directions on the
  inner ring (even 16-direction indices), 16 on the other two (lc_tasks.py). Task id = 1000 + target index.
- The task is "place the object at the target position"; the target is invisible; the same sentence for all tasks.
- Scenes, per (task, episode), from a generator seeded by both (lc_scene.py): the object to place drawn from 10
  Google Scanned Objects bowls and ramekins; up to 10 low (height <= 7 cm) and 6 tall clutter objects drawn from 126
  other scanned objects (120 drawn at random from the 1030 of kevinzakka/mujoco_scanned_objects, plus 4 mugs and 2
  misfits removed from the place set), random yaw, kept 4 cm clear of the start and the target; tall objects also
  6 cm clear of the straight carry path (they occlude the front camera, not the arm). Poses are set after reset.
- Cameras: front (agentview) and wrist, 256x256; Franka arm with LIBERO's OSC_POSE controller.
- Success: the object's centre within 3 cm of the target, resting on the table (z < 0.905 m), with the gripper
  command open; the episode ends at the first step that satisfies all three. The final distance is also recorded.
- Pilot (not scored): the scripted demonstrator with a rim inset of 16 mm placed 36 of 40 scenes; all failures but one
  came from one white ramekin (0/6 over two insets), which was then moved to the clutter pool.

## Demonstrations
Scripted, from privileged state (lc_script.for_vessel, rim inset 16 mm). During execution Gaussian noise
(sd 0.15, in action units, translation only) is added to the executed action; the recorded label is the clean
scripted action. Episodes in which the script fails are dropped. Per task, episodes are generated in index order
from 0 until 20 succeed (at most 40 attempts). The first 14 successful episodes form the training split, the next
3 validation, the last 3 test (the codebase's splits 0/1/2). Rows every 4th step; label = the next 16 clean actions
with a validity mask (the substrate's chunking); features = the frozen SmolVLA extractor and observation contract of
cl_adapt/cl_rollout.py (same weights hash, rounded to float16). Rollout scenes use episodes 100-109 and never occur in
the demonstrations.

Gate G0: if the scripted success rate over all attempts is below 80%, or any task has fewer than 20 successes, the
study stops and is reported as infeasible at that point.

## Heads (MSAL, nothing trained at deployment)
Descriptors, as in the development family, with the radius index scaled to [-1, 1] (r = index - 1):
- coordinates C (xy6): const, cos t, sin t, r, r cos t, r sin t;
- harmonic H (h9): const, cos t, sin t, cos 2t, sin 2t, r, r^2, r cos t, r sin t;
- none N: the plain head without a descriptor (it cannot know the target; the floor).
Folds: four direction folds F0-F3; fold f holds every target in directions f, f+4, f+8, f+12 (12, 8, 12, 8 targets)
and trains on the rest. For each fold and seed (148021, 148022, 148023): MSAL stage 1 (film_nl for C and H,
plain_nl for N; the codebase's training loop and 12,800 steps, unchanged) and, for C and H, stage 2 (the residual
library of 16 atoms, unchanged). MSAL with one demonstration (k = 1, per held target: descriptor inversion and the
closed-form code, unchanged) for C and H. Each artifact stores a probe; the rollout process rebuilds the head and
stops if it disagrees by more than 1e-3.

## Policies and rollouts
Per held target: C0 (C zero-shot), H0, N0, C1 (C with MSAL k = 1), H1. Seen reference per fold: C0 and H0 on the
training targets in directions f+2 and f+10. Episodes 100-109 per target and seed; N_exec 8, 10 Euler steps, 300
steps at most; the flow noise generator seeded by (seed, task, episode) so all policies of an episode share their
noise and their scene. The scripted demonstrator is run on every rollout scene (its success = the scene's ceiling).
40 targets x 10 episodes x 3 seeds x 5 policies = 6,000 held episodes, plus the seen reference. ws2 GPU 1 through
gpu_run (rollout slots; training in the train slot).

## Readings (fixed now)
Unit: the held target. Paired differences of success between policies; two-level bootstrap (targets, then episodes
within target with the three seeds pooled), 10,000 resamples, generator seed [20261116, index]; "better" or "worse"
only if the 95% interval excludes 0.
- P1: C0 vs N0. Does the coordinate descriptor carry an unseen target in closed loop with frozen weights?
- P2: C0 vs H0. Coordinates vs harmonic at unseen directions.
Secondary: S1 C1 vs C0; S2 H1 vs H0; S3 C0 held vs C0 seen (generalization gap, unpaired bootstrap over targets);
S4 the final distance (per-target median), P1 and P2 comparisons with the same bootstrap.
Descriptive: success by ring, fold and object; on the scenes the script solves; offline flow and action error of
each head on its held targets; the scripted ceiling.

## Consequences fixed now
The main paper's closed-loop section reports P1 and P2 whatever they show. If P1 is not "better", the paper states
that, in this family, the coordinate descriptor did not carry unseen targets in closed loop. If P2 reads "worse",
the paper states that closed loop reversed the offline ordering. The supplement reports every reading above.

## Amendment 1 (2026-11-16, after the first demonstrations, before any head was trained or any policy rolled out)
Gate G0 triggered: over all 845 attempts the script succeeded on 796 (94%), but target 1024 (x = 0.17, y = 0, the
point farthest from the robot) reached only 16 successes in 40 attempts (40%; its neighbour 1039 at x = 0.15: 83%).
A diagnostic (lc_diag24.py, 10 episodes) showed the cause: the carry phase timed out with the end effector stalled at
x = 0.146 m at carry height, the arm's reach; every miss ended 3.0-3.5 cm short of the target, just outside the
3 cm radius. Clutter and objects were not involved. Change: the target layout moves back and contracts, centre
(-0.08, 0) instead of (-0.05, 0), ring radii 0.08 / 0.13 / 0.18 m instead of 0.10 / 0.16 / 0.22 m, so the farthest
target lies at x = 0.10 m. Adjacent targets stay at least 5.1 cm apart (more than the 3 cm success radius). Nothing
else changes: the same scene generator, objects, script, noise, splits, folds, arms, policies and readings. The first
demonstrations are kept (shards_v0) and not used; all demonstrations are regenerated and gate G0 is applied again
unchanged. The supplement reports this amendment and its trigger.
