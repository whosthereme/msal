import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import lc_tasks

from libero.libero.envs import OffScreenRenderEnv

T = lc_tasks.tasks()
print("tasks:", len(T))
bddl = lc_tasks.write_bddl(T[0], HERE / "bddl")
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
obs = env.reset()
inner = env.env
sim = inner.sim
print("obs keys:", sorted(obs.keys()))
print("eef pos", np.round(obs["robot0_eef_pos"], 4), "quat", np.round(obs["robot0_eef_quat"], 4))
print("gripper qpos", np.round(obs["robot0_gripper_qpos"], 4))
bid = inner.obj_body_id["akita_black_bowl_1"]
print("bowl body pos", np.round(sim.data.body_xpos[bid], 4))
print("table offset", getattr(inner, "workspace_offset", None), "table full size",
  getattr(inner, "table_full_size", None))
print("controller", type(inner.robots[0].controller).__name__, "output_max",
  getattr(inner.robots[0].controller, "output_max", None))
geoms = [sim.model.geom_id2name(g) for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]
print("bowl geoms", geoms[:6])
for name in ("main_table_target_region", "main_table_bowl_init_region"):
  try:
    sid = sim.model.site_name2id(name)
    print(name, "site pos", np.round(sim.data.site_xpos[sid], 4))
  except Exception as e:
    print(name, "no site", e)
for _ in range(5):
  obs, r, done, info = env.step([0.0] * 6 + [-1.0])
print("after 5 no-op steps: bowl", np.round(sim.data.body_xpos[bid], 4), "done", done)
print("check_success", inner._check_success())
img = obs["agentview_image"]
print("image", img.shape, img.dtype)
from PIL import Image
Image.fromarray(img[::-1]).save(HERE / "probe_agentview.png")
Image.fromarray(obs["robot0_eye_in_hand_image"][::-1]).save(HERE / "probe_wrist.png")
env.close()
print("probe ok")
