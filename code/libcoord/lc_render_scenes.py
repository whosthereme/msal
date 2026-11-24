import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import lc_objects
import lc_scene
import lc_tasks
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

catalog = lc_objects.register()
T = lc_tasks.tasks()
rows = []
for ti, ep in ((0, 0), (12, 1), (30, 2), (37, 3), (21, 4), (5, 5)):
  path, scene, meta = lc_scene.write(T[ti], ep, catalog, HERE / "bddl_pilot")
  print("task", ti, "episode", ep, "vessel", meta["vessel"], "clutter", len(meta["clutter"]), flush=True)
  env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256, camera_widths=256)
  env.seed(ep)
  env.reset()
  obs = lc_scene.pin(env, scene)
  key = meta["vessel"] + "_1"
  print("   vessel at", np.round(obs[key + "_pos"], 3), "start", meta["start"], "target", np.round(meta["target"], 3), flush=True)
  rows.append(np.concatenate([obs["agentview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=1))
  env.close()
Image.fromarray(np.concatenate(rows, axis=0)).save(HERE / "pilot_scenes.png")
print("written pilot_scenes.png")
