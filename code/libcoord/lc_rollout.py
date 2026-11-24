import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, "/home/ubuntu/work/manuscript6/cl_adapt")
import numpy as np
import torch
import torch.nn.functional as F

import lc_objects
import lc_scene
import lc_script
import lc_tasks
from cl_rollout import CHUNK, ACTION, Head, feature, load_extractor, sample_chunk

MAX_STEPS, N_EXEC, FLOW_STEPS, RIM_IN = 300, 8, 10, 0.016
SUCCESS_RADIUS, REST_Z = 0.03, 0.905
INSTRUCTION = "place the object at the target position"
TASKS = {1000 + t["task_id"]: t for t in lc_tasks.tasks()}
ARM = dict(C="lc_xy6", H="lc_h9", N="lc_plain")


class PlainHead(Head):
  def velocity(self, s, x, t, d=None, z=None):
    p = self.sd
    n = s.shape[0]
    h = self.hidden(s, x, t)
    out = F.linear(h, p["plain_head.weight"], p["plain_head.bias"])
    out = out + F.linear(F.silu(F.linear(h, p["plain_mlp.inner.weight"], p["plain_mlp.inner.bias"])),
      p["plain_mlp.outer.weight"], p["plain_mlp.outer.bias"])
    return out.reshape(n, CHUNK, ACTION)


def held_ids(fold):
  return sorted(i for i, t in TASKS.items() if t["direction_16"] % 4 == fold)


def seen_ref_ids(fold):
  return sorted(i for i, t in TASKS.items() if t["direction_16"] in ((fold + 2) % 16, (fold + 10) % 16))


def build(arts, policy, task, device):
  letter, k = policy[0], policy[1]
  art = arts[letter]
  pr = art["probe"]
  if letter == "N":
    head, d, z = PlainHead(art["head"]), torch.zeros(1, 1), None
    ref = art["probe_v_plain"]
  elif k == "0":
    head, d, z = Head(art["head"]), art["descriptors"][task], None
    ref = art["probe_v"][task]
  else:
    ad = art["adapt"][task]
    head, d, z = Head(art["head"], art["library"]), ad["d"], ad["z"]
    ref = ad["probe_v"]
  with torch.no_grad():
    v = head.velocity(pr["state"], pr["x"], pr["t"], d, z)
  err = float((v - ref).abs().max())
  if err > 1e-3:
    raise RuntimeError("%s task %d: rebuilt head disagrees with its probe (%.2e)" % (policy, task, err))
  return head.to_(device), d.to(device), (None if z is None else z.to(device)), err


def placed(obs, obj, target, grip):
  p = obs[obj + "_pos"]
  return bool(np.linalg.norm(p[:2] - target) < SUCCESS_RADIUS and p[2] < REST_Z and grip < 0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fold", type=int, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--heads", type=Path, required=True)
  ap.add_argument("--tasks", choices=("held", "seen"), required=True)
  ap.add_argument("--policies", required=True)
  ap.add_argument("--episodes", default="100-109")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--only-task", type=int, default=0)
  args = ap.parse_args()
  a, b = (int(v) for v in args.episodes.split("-"))
  policies = args.policies.split(",")
  from libero.libero.envs import OffScreenRenderEnv
  catalog = lc_objects.register()
  arts = {}
  for letter in sorted(set(p[0] for p in policies if p != "script")):
    arts[letter] = torch.load(args.heads / ("%s_f%d_s%d.pt" % (ARM[letter], args.fold, args.seed)), weights_only=False)
  extractor, pre = load_extractor(args.device)
  done = set()
  if args.out.exists():
    for line in args.out.read_text().splitlines():
      if line.strip():
        r = json.loads(line)
        done.add((r["policy"], r["task"], r["episode"]))
  tasks = held_ids(args.fold) if args.tasks == "held" else seen_ref_ids(args.fold)
  if args.only_task:
    tasks = [args.only_task]
  for tid in tasks:
    task = TASKS[tid]
    built = {p: build(arts, p, tid, args.device) for p in policies if p != "script"}
    for ep in range(a, b + 1):
      todo = [p for p in policies if (p, tid, ep) not in done]
      if not todo:
        continue
      path, scene, meta = lc_scene.write(task, ep, catalog, HERE / "bddl_roll")
      env = lc_scene.make_env(path, camera_heights=256, camera_widths=256)
      obj = meta["vessel"] + "_1"
      target = np.asarray(scene["target"])
      for policy in todo:
        t0 = time.time()
        obs = lc_scene.reset(env, ep, scene)
        success, steps, grip = False, 0, -1.0
        if policy == "script":
          script = lc_script.for_vessel(scene["vessel"], scene["target"], rim_in=RIM_IN)
          while steps < MAX_STEPS and not success:
            act = script(obs)
            obs, _, _, _ = env.step(act.tolist())
            steps += 1
            grip = act[6]
            success = placed(obs, obj, target, grip)
          probe_err = None
        else:
          head, d, z, probe_err = built[policy]
          gen = torch.Generator(device="cpu").manual_seed(1_000_003 * (args.seed % 100_000) + 1_009 * tid + ep)
          while steps < MAX_STEPS and not success:
            chunk = sample_chunk(head, feature(extractor, pre, obs, INSTRUCTION, args.device), d, z, gen, FLOW_STEPS)
            for j in range(N_EXEC):
              act = np.clip(chunk[j], -1.0, 1.0)
              obs, _, _, _ = env.step(act.tolist())
              steps += 1
              grip = float(act[6])
              success = placed(obs, obj, target, grip)
              if success or steps >= MAX_STEPS:
                break
        p = obs[obj + "_pos"]
        rec = dict(fold=args.fold, seed=args.seed, task=tid, ring=task["radius_index"], direction=task["direction_16"],
          set=args.tasks, policy=policy, episode=ep, vessel=meta["vessel"], success=success, steps=steps,
          final_dist=round(float(np.linalg.norm(p[:2] - target)), 4), final_z=round(float(p[2]), 4),
          seconds=round(time.time() - t0, 1), probe_max_abs=probe_err)
        with args.out.open("a") as fh:
          fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
      env.close()


if __name__ == "__main__":
  main()
