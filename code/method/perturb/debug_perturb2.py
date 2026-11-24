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
pl.MAX_TRIES = 1
for r in (1e-4, 0.04):
    print(r, json.dumps(pl.perturb(env, t, 0, 98, r)))
