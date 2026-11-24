import argparse
import atexit
import json
import runpy
import sys
from pathlib import Path

import numpy as np

SETTLE = 10


def main():
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--script", type=Path, required=True)
    args = ap.parse_args(sys.argv[1:cut])
    rest = sys.argv[cut + 1:]
    task = int(rest[rest.index("--task") + 1])
    from libero.libero.envs import OffScreenRenderEnv

    state = {"rec": None}

    def flush():
        r = state["rec"]
        if r is not None and r["steps"] > SETTLE:
            out = {"task": task, "policy": args.label, "episode": r["episode"], "targets": r["targets"],
                   "steps": r["steps"], "lift": {n: round(v, 4) for n, v in r["lift"].items()},
                   "disp": {n: round(float(np.linalg.norm(r["last"][n][:2] - r["ref"][n][:2])), 4) for n in r["ref"]}}
            with open(args.log, "a") as fh:
                fh.write(json.dumps(out) + "\n")
        state["rec"] = None

    def poses(env):
        inner = env.env
        out = {}
        for name, obj in inner.objects_dict.items():
            joints = [j for j in getattr(obj, "joints", []) if j]
            if joints:
                out[name] = np.array(inner.sim.data.get_joint_qpos(joints[0])[:3], dtype=float)
        return out

    orig_seed, orig_set, orig_step = OffScreenRenderEnv.seed, OffScreenRenderEnv.set_init_state, OffScreenRenderEnv.step

    def seed(self, s):
        self._log_episode = int(s)
        return orig_seed(self, s)

    def set_init_state(self, init_state):
        flush()
        obs = orig_set(self, init_state)
        state["rec"] = {"episode": getattr(self, "_log_episode", -1), "steps": 0, "ref": None, "lift": {},
                        "last": None, "targets": list(getattr(self.env, "obj_of_interest", []))}
        return obs

    def step(self, action):
        out = orig_step(self, action)
        r = state["rec"]
        if r is not None:
            r["steps"] += 1
            p = poses(self)
            if r["steps"] == SETTLE:
                r["ref"] = p
                r["lift"] = {n: 0.0 for n in p}
            elif r["steps"] > SETTLE:
                for n, q in p.items():
                    r["lift"][n] = max(r["lift"][n], float(q[2] - r["ref"][n][2]))
            r["last"] = p
        return out

    OffScreenRenderEnv.seed = seed
    OffScreenRenderEnv.set_init_state = set_init_state
    OffScreenRenderEnv.step = step
    atexit.register(flush)
    sys.argv = [str(args.script)] + rest
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
