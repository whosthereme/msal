import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import torch.nn.functional as F

import lc_rollout
import lc_objects
import lc_scene
from cl_rollout import ACTION, CHUNK, Head, feature, load_extractor, sample_chunk, time_embedding
from lc_rollout import FLOW_STEPS, INSTRUCTION, MAX_STEPS, N_EXEC, TASKS, held_ids, placed

ARM, POLICY = "lc_tfilm_xy6", "K0"


class TrunkFiLMHead(Head):
  def velocity(self, s, x, t, d, z=None):
    p = self.sd
    n = s.shape[0]
    s = (s.float() - p["state_mean"]) / p["state_std"]
    sh = F.layer_norm(F.silu(F.linear(s, p["state_encoder.0.weight"], p["state_encoder.0.bias"])), (512,),
      p["state_encoder.2.weight"], p["state_encoder.2.bias"])
    ah = F.silu(F.linear(x.reshape(n, -1).float(), p["action_encoder.0.weight"], p["action_encoder.0.bias"]))
    h = torch.cat([sh, ah, time_embedding(t.float())], dim=-1)
    dd = d.expand(n, -1)
    h = F.silu(F.linear(h, p["trunk.0.weight"], p["trunk.0.bias"]))
    h = h * (1.0 + F.linear(dd, p["trunk_gamma.weight"])) + F.linear(dd, p["trunk_beta.weight"])
    h = F.silu(F.linear(h, p["trunk.2.weight"], p["trunk.2.bias"]))
    h = F.layer_norm(h, (512,), p["trunk.4.weight"], p["trunk.4.bias"])
    out = F.linear(h, p["plain_head.weight"], p["plain_head.bias"])
    out = out + F.linear(F.silu(F.linear(h, p["plain_mlp.inner.weight"], p["plain_mlp.inner.bias"])),
      p["plain_mlp.outer.weight"], p["plain_mlp.outer.bias"])
    return out.reshape(n, CHUNK, ACTION)


def build(art, task, device):
  pr = art["probe"]
  head, d = TrunkFiLMHead(art["head"]), art["descriptors"][task]
  with torch.no_grad():
    v = head.velocity(pr["state"], pr["x"], pr["t"], d, None)
  err = float((v - art["probe_v"][task]).abs().max())
  if err > 1e-3:
    raise RuntimeError("K0 task %d: rebuilt head disagrees with its probe (%.2e)" % (task, err))
  return head.to_(device), d.to(device), err


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fold", type=int, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--heads", type=Path, required=True)
  ap.add_argument("--episodes", default="100-109")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--only-task", type=int, default=0)
  ap.add_argument("--probe-only", action="store_true")
  args = ap.parse_args()
  a, b = (int(v) for v in args.episodes.split("-"))
  art = torch.load(args.heads / ("%s_f%d_s%d.pt" % (ARM, args.fold, args.seed)), weights_only=False)
  tasks = held_ids(args.fold)
  if args.only_task:
    tasks = [args.only_task]
  if args.probe_only:
    worst = max(build(art, tid, "cpu")[2] for tid in tasks)
    print("probe check passed: %d targets, max abs %.2e" % (len(tasks), worst))
    return
  from libero.libero.envs import OffScreenRenderEnv
  catalog = lc_objects.register()
  extractor, pre = load_extractor(args.device)
  done = set()
  if args.out.exists():
    for line in args.out.read_text().splitlines():
      if line.strip():
        r = json.loads(line)
        done.add((r["task"], r["episode"]))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  for tid in tasks:
    task = TASKS[tid]
    head, d, probe_err = build(art, tid, args.device)
    for ep in range(a, b + 1):
      if (tid, ep) in done:
        continue
      path, scene, meta = lc_scene.write(task, ep, catalog, HERE / "bddl_roll")
      env = lc_scene.make_env(path, camera_heights=256, camera_widths=256)
      obj = meta["vessel"] + "_1"
      target = np.asarray(scene["target"])
      t0 = time.time()
      obs = lc_scene.reset(env, ep, scene)
      success, steps, grip = False, 0, -1.0
      gen = torch.Generator(device="cpu").manual_seed(1_000_003 * (args.seed % 100_000) + 1_009 * tid + ep)
      while steps < MAX_STEPS and not success:
        chunk = sample_chunk(head, feature(extractor, pre, obs, INSTRUCTION, args.device), d, None, gen, FLOW_STEPS)
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
        set="held", policy=POLICY, episode=ep, vessel=meta["vessel"], success=success, steps=steps,
        final_dist=round(float(np.linalg.norm(p[:2] - target)), 4), final_z=round(float(p[2]), 4),
        seconds=round(time.time() - t0, 1), probe_max_abs=probe_err)
      with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
      print(json.dumps(rec), flush=True)
      env.close()


if __name__ == "__main__":
  main()
