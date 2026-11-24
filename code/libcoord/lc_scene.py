import json
import math
from pathlib import Path

import numpy as np

import lc_tasks

N_LOW, N_TALL = 10, 6
KEEP = 0.04
LOW_MAX_H = 0.07
WORK = dict(x=(-0.30, 0.20), y=(-0.32, 0.32))
TALL_AREA = dict(x=(-0.28, 0.26), y=(-0.30, 0.30))
CORRIDOR = 0.06


NOT_VESSEL = ("Mug", "Wagon", "Threshold_Ramekin_White")


def is_vessel(e):
  return e["kind"] == "place" and not any(w in e["name"] for w in NOT_VESSEL)


def radius(e):
  return 0.5 * math.hypot(e["size_x"], e["size_y"])


def episode_scene(task, episode, catalog, seed=20261116):
  rng = np.random.default_rng([seed, int(task["task_id"]), int(episode)])
  place = [e for e in catalog if is_vessel(e)]
  pool = [e for e in catalog if not is_vessel(e)]
  low = [e for e in pool if e["height"] <= LOW_MAX_H]
  tall = [e for e in pool if e["height"] > LOW_MAX_H]
  vessel = place[int(rng.integers(len(place)))]
  start = np.array(lc_tasks.CENTER) + rng.uniform(-0.01, 0.01, size=2)
  target = np.array([task["x"], task["y"]])
  keep = [(start, radius(vessel) + KEEP), (target, radius(vessel) + KEEP)]
  chosen = []

  def seg_dist(p):
    d = target - start
    t = float(np.clip(np.dot(p - start, d) / max(float(np.dot(d, d)), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (start + t * d)))

  def put(entry, box, clear, corridor=False):
    r = radius(entry)
    for _ in range(1000):
      p = np.array([rng.uniform(*box["x"]), rng.uniform(*box["y"])])
      if corridor and seg_dist(p) < r + radius(vessel) + CORRIDOR:
        continue
      if all(np.linalg.norm(p - q) > r + rr for q, rr in keep) and all(
          np.linalg.norm(p - c["xy"]) > r + c["r"] + clear for c in chosen):
        chosen.append(dict(entry=entry, xy=p, r=r, yaw=float(rng.uniform(-math.pi, math.pi))))
        return True
    return False

  for e in rng.choice(len(tall), size=N_TALL, replace=False):
    put(tall[e], TALL_AREA, 0.01, corridor=True)
  for e in rng.choice(len(low), size=N_LOW, replace=False):
    put(low[e], WORK, 0.01)
  return dict(vessel=vessel, clutter=chosen, target=target.tolist(), start=start.tolist(),
    vessel_yaw=float(rng.uniform(-math.pi, math.pi)))


def make_env(path, **kw):
  from libero.libero.envs import OffScreenRenderEnv
  from robosuite.utils.errors import RandomizationError
  for attempt in range(50):
    try:
      return OffScreenRenderEnv(bddl_file_name=str(path), **kw)
    except RandomizationError:
      np.random.seed(7919 + attempt)
  raise RuntimeError("LIBERO could not place the objects of %s" % path)


def reset(env, episode, scene, settle=25):
  from robosuite.utils.errors import RandomizationError
  for attempt in range(50):
    env.seed(episode + 100_000 * attempt)
    try:
      env.reset()
      break
    except RandomizationError:
      continue
  else:
    raise RuntimeError("LIBERO could not reset episode %d" % episode)
  return pin(env, scene, settle)


def pin(env, scene, settle=25):
  base = env.env
  objs = base.objects_dict
  poses = [(scene["vessel"]["key"] + "_1", scene["start"], scene["vessel_yaw"])]
  counts = {}
  for c in scene["clutter"]:
    k = c["entry"]["key"]
    counts[k] = counts.get(k, 0) + 1
    poses.append(("%s_%d" % (k, counts[k]), c["xy"], c["yaw"]))
  for inst, xy, yaw in poses:
    j = objs[inst].joints[0]
    q = base.sim.data.get_joint_qpos(j).copy()
    q[0], q[1] = xy[0], xy[1]
    q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    base.sim.data.set_joint_qpos(j, q)
    base.sim.data.set_joint_qvel(j, np.zeros(6))
  base.sim.forward()
  obs = None
  for _ in range(settle):
    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
  return obs


def bddl(task, scene):
  regions, objects, init = [], [], []

  def region(name, x, y, half=0.005, yaw=None):
    yaw_txt = "" if yaw is None else "\n          (:yaw_rotation (\n              (%.4f %.4f)\n            )\n          )" % (yaw, yaw + 1e-3)
    regions.append("      (%s\n          (:target main_table)\n          (:ranges (\n              (%.4f %.4f %.4f %.4f)\n            )\n          )%s\n      )"
      % (name, x - half, y - half, x + half, y + half, yaw_txt))

  v = scene["vessel"]
  region("vessel_init_region", scene["start"][0], scene["start"][1], half=0.01)
  region("target_region", scene["target"][0], scene["target"][1], half=lc_tasks.TARGET_HALF)
  objects.append("    %s_1 - %s" % (v["key"], v["key"]))
  init.append("    (On %s_1 main_table_vessel_init_region)" % v["key"])
  counts = {}
  for i, c in enumerate(scene["clutter"]):
    k = c["entry"]["key"]
    counts[k] = counts.get(k, 0) + 1
    inst = "%s_%d" % (k, counts[k])
    region("clutter_%d_region" % i, c["xy"][0], c["xy"][1], yaw=c["yaw"])
    objects.append("    %s - %s" % (inst, k))
    init.append("    (On %s main_table_clutter_%d_region)" % (inst, i))
  return ("(define (problem LIBERO_Tabletop_Manipulation)\n  (:domain robosuite)\n"
    "  (:language place the object at the target position)\n    (:regions\n" + "\n".join(regions) + "\n    )\n\n"
    "  (:fixtures\n    main_table - table\n  )\n\n  (:objects\n" + "\n".join(objects) + "\n  )\n\n"
    "  (:obj_of_interest\n    %s_1\n  )\n\n  (:init\n" % v["key"] + "\n".join(init) + "\n  )\n\n"
    "  (:goal\n    (And (On %s_1 main_table_target_region))\n  )\n\n)\n" % v["key"])


def write(task, episode, catalog, folder):
  scene = episode_scene(task, episode, catalog)
  folder = Path(folder)
  folder.mkdir(parents=True, exist_ok=True)
  path = folder / ("coord_t%02d_e%03d.bddl" % (task["task_id"], episode))
  path.write_text(bddl(task, scene))
  meta = dict(task_id=task["task_id"], episode=episode, vessel=scene["vessel"]["key"], target=scene["target"],
    start=[round(float(v), 4) for v in scene["start"]],
    clutter=[dict(key=c["entry"]["key"], xy=[round(float(c["xy"][0]), 4), round(float(c["xy"][1]), 4)],
      yaw=round(c["yaw"], 4)) for c in scene["clutter"]])
  return path, scene, meta
