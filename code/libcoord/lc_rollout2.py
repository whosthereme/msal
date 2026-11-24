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

ARM = dict(F="lc_film_xy6", M="lc_moe_xy6", T="lc_tin_xy6", G="lc_h20")


class FilmLinearHead(Head):
  def velocity(self, s, x, t, d, z=None):
    p = self.sd
    n = s.shape[0]
    h = self.hidden(s, x, t)
    dd = d.expand(n, -1)
    g = h * (1.0 + F.linear(dd, p["film_gamma.weight"])) + F.linear(dd, p["film_beta.weight"])
    return F.linear(g, p["film_head.weight"], p["film_head.bias"]).reshape(n, CHUNK, ACTION)


class MoEHead(Head):
  def velocity(self, s, x, t, d, z=None):
    p = self.sd
    n = s.shape[0]
    h = self.hidden(s, x, t)
    base = F.linear(h, p["base_out.weight"], p["base_out.bias"]).reshape(n, CHUNK, ACTION)
    inner = F.silu(torch.einsum("kch,bh->bkc", p["atom_out.w1"], h) + p["atom_out.b1"][None])
    experts = (torch.einsum("koc,bkc->bko", p["atom_out.w2"], inner) + p["atom_out.b2"][None]).reshape(n, -1, CHUNK,
      ACTION)
    w = torch.softmax(F.linear(d.expand(n, -1), p["gate.weight"]), dim=-1)
    return base + torch.einsum("bk,bktd->btd", w, experts)


class TinHead(Head):
  def velocity(self, s, x, t, d, z=None):
    p = self.sd
    n = s.shape[0]
    s = (s.float() - p["state_mean"]) / p["state_std"]
    sh = F.layer_norm(F.silu(F.linear(s, p["state_encoder.0.weight"], p["state_encoder.0.bias"])), (512,),
      p["state_encoder.2.weight"], p["state_encoder.2.bias"])
    ah = F.silu(F.linear(x.reshape(n, -1).float(), p["action_encoder.0.weight"], p["action_encoder.0.bias"]))
    h = torch.cat([sh, ah, time_embedding(t.float())], dim=-1)
    h = F.silu(F.linear(h, p["trunk.0.weight"], p["trunk.0.bias"]) + F.linear(d.expand(n, -1), p["desc_in.weight"]))
    h = F.silu(F.linear(h, p["trunk.2.weight"], p["trunk.2.bias"]))
    h = F.layer_norm(h, (512,), p["trunk.4.weight"], p["trunk.4.bias"])
    out = F.linear(h, p["plain_head.weight"], p["plain_head.bias"])
    out = out + F.linear(F.silu(F.linear(h, p["plain_mlp.inner.weight"], p["plain_mlp.inner.bias"])),
      p["plain_mlp.outer.weight"], p["plain_mlp.outer.bias"])
    return out.reshape(n, CHUNK, ACTION)


CLASS = dict(F=FilmLinearHead, M=MoEHead, T=TinHead, G=Head)


def build(arts, policy, task, device):
  art = arts[policy[0]]
  pr = art["probe"]
  head, d = CLASS[policy[0]](art["head"]), art["descriptors"][task]
  with torch.no_grad():
    v = head.velocity(pr["state"], pr["x"], pr["t"], d, None)
  err = float((v - art["probe_v"][task]).abs().max())
  if err > 1e-3:
    raise RuntimeError("%s task %d: rebuilt head disagrees with its probe (%.2e)" % (policy, task, err))
  return head.to_(device), d.to(device), err


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fold", type=int, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--heads", type=Path, required=True)
  ap.add_argument("--policies", default="F0,M0,T0,G0")
  ap.add_argument("--episodes", default="100-109")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--only-task", type=int, default=0)
  ap.add_argument("--probe-only", action="store_true")
  args = ap.parse_args()
  a, b = (int(v) for v in args.episodes.split("-"))
  policies = args.policies.split(",")
  assert all(p in ("F0", "M0", "T0", "G0") for p in policies), policies
  arts = {}
  for letter in sorted(set(p[0] for p in policies)):
    arts[letter] = torch.load(args.heads / ("%s_f%d_s%d.pt" % (ARM[letter], args.fold, args.seed)), weights_only=False)
  tasks = held_ids(args.fold)
  if args.only_task:
    tasks = [args.only_task]
  if args.probe_only:
    worst = max(build(arts, p, tid, "cpu")[2] for tid in tasks for p in policies)
    print("probe check passed: %d heads x %d targets, max abs %.2e" % (len(policies), len(tasks), worst))
    return
  from libero.libero.envs import OffScreenRenderEnv
  catalog = lc_objects.register()
  extractor, pre = load_extractor(args.device)
  done = set()
  if args.out.exists():
    for line in args.out.read_text().splitlines():
      if line.strip():
        r = json.loads(line)
        done.add((r["policy"], r["task"], r["episode"]))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  for tid in tasks:
    task = TASKS[tid]
    built = {p: build(arts, p, tid, args.device) for p in policies}
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
        head, d, probe_err = built[policy]
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
          set="held", policy=policy, episode=ep, vessel=meta["vessel"], success=success, steps=steps,
          final_dist=round(float(np.linalg.norm(p[:2] - target)), 4), final_z=round(float(p[2]), 4),
          seconds=round(time.time() - t0, 1), probe_max_abs=probe_err)
        with args.out.open("a") as fh:
          fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
      env.close()


if __name__ == "__main__":
  main()
