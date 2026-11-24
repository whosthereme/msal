import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import lc_objects, lc_scene, lc_script, lc_tasks
catalog = lc_objects.register()
task = lc_tasks.tasks()[int(sys.argv[1])]
for ep in range(int(sys.argv[2])):
  path, scene, meta = lc_scene.write(task, ep, catalog, HERE / "bddl_pilot")
  env = lc_scene.make_env(path, camera_heights=64, camera_widths=64)
  obs = lc_scene.reset(env, ep, scene)
  obj = meta["vessel"] + "_1"
  s = lc_script.for_vessel(scene["vessel"], scene["target"], rim_in=0.016)
  ph, lens, carry_end = 0, {}, None
  n = 0
  while not s.done and n < 500:
    before = s.phase
    obs, _, _, _ = env.step(s(obs).tolist())
    n += 1
    lens[before] = lens.get(before, 0) + 1
    if before == 4 and s.phase == 5:
      carry_end = (np.round(obs[obj + "_pos"][:2] - scene["target"], 3), np.round(obs["robot0_eef_pos"], 3))
  p = obs[obj + "_pos"]
  print(ep, meta["vessel"], "ok", lc_script.success(obs, scene["target"], obj=obj), "err", round(float(np.linalg.norm(p[:2] - scene["target"])), 4),
    "phases", lens, "carry_end(obj-target, eef)", carry_end, flush=True)
  env.close()
