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
from lc_rollout import FLOW_STEPS, INSTRUCTION, MAX_STEPS, N_EXEC, TASKS, held_ids, placed, seen_ref_ids

ARTIFACT = dict(J0="lc_tfilm_h20", K1="lc_tfilm_xy6a", K0="lc_tfilm_xy6a")
SETS = dict(J0=("held",), K1=("held",), K0=("seen",))


class TrunkFiLMLibHead(Head):
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
    v = out.reshape(n, CHUNK, ACTION)
    if z is not None:
      L = self.lib
      inner = F.silu(torch.einsum("kch,bh->bkc", L["w1"], h) + L["b1"][None])
      atoms = (torch.einsum("koc,bkc->bko", L["w2"], inner) + L["b2"][None]).reshape(n, -1, CHUNK, ACTION)
      v = v + torch.einsum("k,bktd->btd", z.float(), atoms)
    return v


def build(art, policy, task, device):
  pr = art["probe"]
  if policy == "K1":
    ad = art["adapt"][task]
    head, d, z, ref = TrunkFiLMLibHead(art["head"], art["library"]), ad["d"], ad["z"], ad["probe_v"]
  else:
    head, d, z, ref = TrunkFiLMLibHead(art["head"]), art["descriptors"][task], None, art["probe_v"][task]
  with torch.no_grad():
    v = head.velocity(pr["state"], pr["x"], pr["t"], d, z)
  err = float((v - ref).abs().max())
  if err > 1e-3:
    raise RuntimeError("%s task %d: rebuilt head disagrees with its probe (%.2e)" % (policy, task, err))
  return head.to_(device), d.to(device), (None if z is None else z.to(device)), err


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", choices=sorted(ARTIFACT), required=True)
  ap.add_argument("--set", choices=("held", "seen"), required=True)
  ap.add_argument("--fold", type=int, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--heads", type=Path, required=True)
  ap.add_argument("--episodes", default="100-109")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--only-task", type=int, default=0)
  ap.add_argument("--probe-only", action="store_true")
  args = ap.parse_args()
  assert args.set in SETS[args.policy], (args.policy, args.set)
  a, b = (int(v) for v in args.episodes.split("-"))
  art = torch.load(args.heads / ("%s_f%d_s%d.pt" % (ARTIFACT[args.policy], args.fold, args.seed)), weights_only=False)
  tasks = held_ids(args.fold) if args.set == "held" else seen_ref_ids(args.fold)
  if args.only_task:
    tasks = [args.only_task]
  if args.probe_only:
    worst = max(build(art, args.policy, tid, "cpu")[3] for tid in tasks)
    print("probe check passed: %s, %d targets, max abs %.2e" % (args.policy, len(tasks), worst))
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
    head, d, z, probe_err = build(art, args.policy, tid, args.device)
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
        set=args.set, policy=args.policy, episode=ep, vessel=meta["vessel"], success=success, steps=steps,
        final_dist=round(float(np.linalg.norm(p[:2] - target)), 4), final_z=round(float(p[2]), 4),
        seconds=round(time.time() - t0, 1), probe_max_abs=probe_err)
      with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
      print(json.dumps(rec), flush=True)
      env.close()


if __name__ == "__main__":
  main()
