import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, "/home/ubuntu/work/manuscript6/cl_adapt")
import numpy as np
import lc_objects
import lc_scene
import lc_script
import lc_tasks

NEED, MAX_ATTEMPTS, NOISE, RIM_IN = 20, 40, 0.15, 0.016
STRIDE, CHUNK, MAX_STEPS = 4, 16, 500
SPLIT = [0] * 14 + [1] * 3 + [2] * 3
INSTRUCTION = "place the object at the target position"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tasks", required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--smoke-need", type=int, default=0)
  args = ap.parse_args()
  global NEED
  if args.smoke_need:
    NEED = args.smoke_need
    assert "SMOKE" in args.out.name
  args.out.mkdir(parents=True, exist_ok=True)
  import torch
  from cl_rollout import feature, load_extractor
  from libero.libero.envs import OffScreenRenderEnv
  catalog = lc_objects.register()
  extractor, pre = load_extractor(args.device)
  T = lc_tasks.tasks()
  for ti in [int(x) for x in args.tasks.split(",") if x.strip()]:
    task = T[ti]
    tid = 1000 + task["task_id"]
    if (args.out / ("task%d.npz" % tid)).exists():
      continue
    t0 = time.time()
    rows = dict(state_features=[], action_chunks=[], action_valid_masks=[], task_ids=[], episode_indices=[],
      step_indices=[], split_codes=[])
    episodes, n_ok = [], 0
    for ep in range(MAX_ATTEMPTS):
      if n_ok == NEED:
        break
      path, scene, meta = lc_scene.write(task, ep, catalog, HERE / "bddl_demo")
      env = lc_scene.make_env(path, camera_heights=256, camera_widths=256)
      obs = lc_scene.reset(env, ep, scene)
      obj = meta["vessel"] + "_1"
      script = lc_script.for_vessel(scene["vessel"], scene["target"], rim_in=RIM_IN)
      rng = np.random.default_rng([20261116, task["task_id"], ep])
      feats, clean, n = [], [], 0
      while not script.done and n < MAX_STEPS:
        if n % STRIDE == 0:
          feats.append(feature(extractor, pre, obs, INSTRUCTION, args.device).cpu().numpy().astype(np.float16)[0])
        a = script(obs)
        clean.append(a.copy())
        a[:3] = np.clip(a[:3] + rng.normal(0.0, NOISE, 3), -1, 1)
        obs, _, _, _ = env.step(a.tolist())
        n += 1
      ok = lc_script.success(obs, scene["target"], obj=obj)
      env.close()
      episodes.append(dict(episode=ep, success=bool(ok), steps=n, vessel=meta["vessel"], start=meta["start"],
        n_clutter=len(meta["clutter"]), clutter=[c["key"] for c in meta["clutter"]]))
      if not ok:
        continue
      clean = np.asarray(clean, dtype=np.float32)
      for j, f in enumerate(feats):
        s = j * STRIDE
        chunk = np.zeros((CHUNK, 7), np.float32)
        mask = np.zeros(CHUNK, np.float32)
        m = min(CHUNK, n - s)
        chunk[:m], mask[:m] = clean[s:s + m], 1.0
        rows["state_features"].append(f)
        rows["action_chunks"].append(chunk)
        rows["action_valid_masks"].append(mask)
        rows["task_ids"].append(tid)
        rows["episode_indices"].append(ep)
        rows["step_indices"].append(s)
        rows["split_codes"].append(SPLIT[n_ok])
      n_ok += 1
    info = dict(task=task, task_id=tid, successes=n_ok, attempts=len(episodes), episodes=episodes,
      seconds=time.time() - t0, noise=NOISE, rim_in=RIM_IN)
    (args.out / ("task%d.json" % tid)).write_text(json.dumps(info, indent=1))
    if n_ok < NEED:
      print("task %d: only %d successes in %d attempts (gate G0)" % (tid, n_ok, len(episodes)), flush=True)
      continue
    np.savez(args.out / ("task%d.npz" % tid), **{k: np.asarray(v) for k, v in rows.items()})
    print("task %d: %d/%d successes, %d rows, %.0fs" % (tid, n_ok, len(episodes), len(rows["task_ids"]),
      time.time() - t0), flush=True)


if __name__ == "__main__":
  main()
