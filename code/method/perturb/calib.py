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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--radii", default="0.03,0.06")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--dry", type=int, default=10, help="episodes per radius for the dry run")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    out = {}
    for t in [int(x) for x in args.tasks.split(",")]:
        info = json.loads(TASKS.read_text())["tasks"][str(t)]
        suite = benchmark.get_benchmark_dict()[info["suite"]]()
        i = [j for j in range(suite.n_tasks) if suite.get_task(j).language.lower() == info["instruction"].lower()][0]
        task = suite.get_task(i)
        init_states = suite.get_task_init_states(i)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        pos = []
        for ep in range(args.episodes):
            env.seed(ep)
            env.reset()
            env.set_init_state(init_states[ep % len(init_states)])
            pos.append({n: q[:2].tolist() for n, q in pl.object_poses(env).items()})
        names = list(pos[0])
        spread = {}
        for n in names:
            xy = np.array([p[n] for p in pos])
            spread[n] = {"std_xy_cm": (100 * xy.std(0)).round(2).tolist(),
                         "range_cm": (100 * (xy.max(0) - xy.min(0))).round(2).tolist()}
        dry = {}
        for r in [float(x) for x in args.radii.split(",")]:
            tries, acc = [], 0
            for ep in range(args.dry):
                env.seed(ep)
                env.reset()
                env.set_init_state(init_states[ep % len(init_states)])
                res = pl.perturb(env, t, ep, 99, r)
                tries.append(res["tries"])
                acc += res["accepted"]
            dry[str(r)] = {"accepted": acc, "of": args.dry, "mean_tries": float(np.mean(tries)), "max_tries": int(max(tries))}
        env.close()
        out[t] = {"suite": info["suite"], "instruction": info["instruction"], "movable": names, "spread": spread,
                  "dry_run": dry}
        print(t, json.dumps(out[t]), flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
