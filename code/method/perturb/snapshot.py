import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perturb_lib as pl

TASKS = Path("/home/ubuntu/work/manuscript6/repo/sal_v2_identifiable/runs/"
             "20260829_g39_libero40_substrate_v1/libero40_tasks.json")
ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, required=True)
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

info = json.loads(TASKS.read_text())["tasks"][str(args.task)]
suite = benchmark.get_benchmark_dict()[info["suite"]]()
i = [j for j in range(suite.n_tasks) if suite.get_task(j).language.lower() == info["instruction"].lower()][0]
task = suite.get_task(i)
init = suite.get_task_init_states(i)[args.episode]
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
table = json.load(open(HERE / "perturbations.json"))[str(args.task)]
imgs = []
for lev in (0, 1, 2):
    env.seed(args.episode)
    env.reset()
    env.set_init_state(init)
    if lev:
        pl.place(env, table[str(lev)][str(args.episode)]["offsets"])
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    imgs.append(np.asarray(obs["agentview_image"])[::-1])
Image.fromarray(np.concatenate(imgs, axis=1).astype(np.uint8)).save(args.out)
print("saved", args.out)
