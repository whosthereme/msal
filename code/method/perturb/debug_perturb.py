import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perturb_lib as pl

TASKS = Path("/home/ubuntu/work/manuscript6/repo/sal_v2_identifiable/runs/"
             "20260829_g39_libero40_substrate_v1/libero40_tasks.json")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

t = int(sys.argv[1])
info = json.loads(TASKS.read_text())["tasks"][str(t)]
suite = benchmark.get_benchmark_dict()[info["suite"]]()
i = [j for j in range(suite.n_tasks) if suite.get_task(j).language.lower() == info["instruction"].lower()][0]
task = suite.get_task(i)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
env.seed(0)
env.reset()
env.set_init_state(suite.get_task_init_states(i)[0])
sim = env.env.sim
p0 = pl.object_poses(env)
s0 = sim.get_state()
for _ in range(10):
    env.step([0.0] * 6 + [-1.0])
p1 = pl.object_poses(env)
sim.set_state(s0)
sim.forward()
for _ in range(200):
    sim.step()
sim.forward()
p2 = pl.object_poses(env)
print("timestep", sim.model.opt.timestep, "control_freq", getattr(env.env, "control_freq", None),
      "nsub", getattr(env.env, "control_timestep", None))
for n in p0:
    print(n, "init", np.round(p0[n][:3], 4).tolist(), "after 10 env.step", np.round(p1[n][:3] - p0[n][:3], 4).tolist(),
          "after 200 sim.step", np.round(p2[n][:3] - p0[n][:3], 4).tolist())
for n, j in pl.movable(env):
    bid = sim.model.body_name2id(env.env.objects_dict[n].root_body)
    print(n, j, "body xpos", np.round(sim.data.body_xpos[bid], 4).tolist())
