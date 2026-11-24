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
HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    table = {}
    for t in HELD:
        info = json.loads(TASKS.read_text())["tasks"][str(t)]
        suite = benchmark.get_benchmark_dict()[info["suite"]]()
        i = [j for j in range(suite.n_tasks) if suite.get_task(j).language.lower() == info["instruction"].lower()][0]
        task = suite.get_task(i)
        init_states = suite.get_task_init_states(i)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        table[str(t)] = {}
        for level, radius in pl.LEVELS.items():
            recs = {}
            for ep in range(args.episodes):
                env.seed(ep)
                env.reset()
                env.set_init_state(init_states[ep % len(init_states)])
                res = pl.perturb(env, t, ep, level, radius)
                if res["accepted"]:
                    want = pl.object_poses(env)
                    env.seed(ep)
                    env.reset()
                    env.set_init_state(init_states[ep % len(init_states)])
                    pl.place(env, res["offsets"])
                    got = pl.object_poses(env)
                    res["replay_max_abs"] = float(max(np.abs(want[n] - got[n]).max() for n in want))
                recs[str(ep)] = res
            table[str(t)][str(level)] = recs
            acc = sum(r["accepted"] for r in recs.values())
            print(t, level, radius, f"accepted {acc}/{args.episodes}",
                  "mean tries", round(float(np.mean([r["tries"] for r in recs.values()])), 2),
                  "max replay err", max((r.get("replay_max_abs", 0) for r in recs.values()), default=0), flush=True)
        env.close()
    args.out.write_text(json.dumps(table))


if __name__ == "__main__":
    main()
