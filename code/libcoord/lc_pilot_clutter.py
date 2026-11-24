import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import lc_objects
import lc_scene
import lc_script
import lc_tasks
from libero.libero.envs import OffScreenRenderEnv

RIM_IN, EP, OUT = float(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
NOISE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
catalog = lc_objects.register()
with OUT.open("w") as f:
  for task in lc_tasks.tasks():
    path, scene, meta = lc_scene.write(task, EP, catalog, HERE / "bddl_pilot")
    env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=64, camera_widths=64)
    env.seed(EP)
    env.reset()
    obs = lc_scene.pin(env, scene)
    obj = meta["vessel"] + "_1"
    script = lc_script.for_vessel(scene["vessel"], scene["target"], rim_in=RIM_IN)
    rng = np.random.default_rng([EP, task["task_id"], 7])
    n, phase_max = 0, 0
    while not script.done and n < 500:
      a = script(obs)
      if NOISE:
        a[:3] = np.clip(a[:3] + rng.normal(0.0, NOISE, 3), -1, 1)
      obs, _, _, _ = env.step(a.tolist())
      n += 1
    ok = lc_script.success(obs, scene["target"], obj=obj)
    err = float(np.linalg.norm(obs[obj + "_pos"][:2] - np.asarray(scene["target"])))
    rec = dict(task=task["task_id"], episode=EP, vessel=meta["vessel"], rim_in=RIM_IN, noise=NOISE, success=ok, err=round(err, 4),
      steps=n, phase=script.phase, n_clutter=len(meta["clutter"]))
    f.write(json.dumps(rec) + "\n")
    f.flush()
    print(rec, flush=True)
    env.close()
