from __future__ import annotations

import json
import math
import os

import numpy as np

SEED = 20261102
SETTLE_SIM = 200
TOL_XY, TOL_Z, TOL_TILT_DEG, MIN_SEP = 0.015, 0.02, 20.0, 0.05
MAX_TRIES = 300
KEEP_SEP = 0.10
LEVELS = {1: 0.04, 2: 0.08}


def movable(env):
    inner = env.env
    out = []
    for name, obj in inner.objects_dict.items():
        joints = [j for j in getattr(obj, "joints", []) if j]
        if joints:
            out.append((name, joints[0]))
    return out


def object_poses(env):
    sim = env.env.sim
    return {name: np.array(sim.data.get_joint_qpos(j), dtype=float) for name, j in movable(env)}


def tilt_deg(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    zz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def perturb(env, task: int, episode: int, level: int, radius: float, log_path: str | None = None):
    if radius <= 0:
        return {"task": task, "episode": episode, "level": level, "radius": 0.0, "tries": 0, "accepted": True}
    sim = env.env.sim
    base_state = sim.get_state()
    base = object_poses(env)
    names = list(base)
    for _ in range(SETTLE_SIM):
        sim.step()
    sim.forward()
    rest = object_poses(env)
    parent = {n: n for n in names}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n
    close = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.linalg.norm(base[a][:2] - base[b][:2]) < MIN_SEP:
                parent[root(a)] = root(b)
                close.add((a, b))
    groups = sorted({root(n) for n in names})
    rng = np.random.default_rng([SEED, task, episode, level])
    info = None
    for attempt in range(1, MAX_TRIES + 1):
        sim.set_state(base_state)
        sim.forward()
        shift = {}
        for g in groups:
            rho, phi = radius * math.sqrt(rng.random()), 2 * math.pi * rng.random()
            shift[g] = (rho * math.cos(phi), rho * math.sin(phi))
        offsets = {}
        for n, j in movable(env):
            dx, dy = shift[root(n)]
            q = base[n].copy()
            q[0] += dx
            q[1] += dy
            sim.data.set_joint_qpos(j, q)
            offsets[n] = [dx, dy]
        sim.forward()
        perturbed_state = sim.get_state()
        target = object_poses(env)
        sep_ok = all(np.linalg.norm(target[a][:2] - target[b][:2])
                     >= min(np.linalg.norm(base[a][:2] - base[b][:2]), KEEP_SEP)
                     for i, a in enumerate(names) for b in names[i + 1:] if (a, b) not in close)
        for _ in range(SETTLE_SIM):
            sim.step()
        sim.forward()
        after = object_poses(env)
        dev = {n: (float(np.linalg.norm((after[n][:2] - target[n][:2]) - (rest[n][:2] - base[n][:2]))),
                   float(after[n][2] - rest[n][2]),
                   abs(tilt_deg(after[n][3:7]) - tilt_deg(rest[n][3:7]))) for n in names}
        ok = sep_ok and all(d[0] <= TOL_XY and abs(d[1]) <= TOL_Z and d[2] <= TOL_TILT_DEG for d in dev.values())
        done = bool(env.env._check_success()) if ok else False
        last = {"sep_ok": sep_ok, "done": done, "dev": {n: [round(v, 4) for v in d] for n, d in dev.items()}}
        if ok and not done:
            sim.set_state(perturbed_state)
            sim.forward()
            info = {"task": task, "episode": episode, "level": level, "radius": radius, "tries": attempt,
                    "accepted": True, "offsets": offsets, "max_settle_xy": max(d[0] for d in dev.values())}
            break
    if info is None:
        sim.set_state(base_state)
        sim.forward()
        info = {"task": task, "episode": episode, "level": level, "radius": radius, "tries": MAX_TRIES,
                "accepted": False, "last": last}
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(json.dumps(info) + "\n")
    return info


def place(env, offsets: dict) -> None:
    sim = env.env.sim
    for n, j in movable(env):
        q = np.array(sim.data.get_joint_qpos(j), dtype=float)
        dx, dy = offsets[n]
        q[0] += dx
        q[1] += dy
        sim.data.set_joint_qpos(j, q)
    sim.forward()


def install(table_path: str, level: int, task: int):
    from libero.libero.envs import OffScreenRenderEnv

    table = json.load(open(table_path))[str(task)][str(level)]
    orig_seed, orig_set = OffScreenRenderEnv.seed, OffScreenRenderEnv.set_init_state

    def seed(self, s):
        self._pert_episode = int(s)
        return orig_seed(self, s)

    def set_init_state(self, init_state):
        orig_set(self, init_state)
        ep = getattr(self, "_pert_episode", None)
        rec = table.get(str(ep))
        if rec is None or not rec.get("accepted"):
            raise RuntimeError(f"task {task} episode {ep} level {level}: no valid perturbation (excluded)")
        place(self, rec["offsets"])
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    OffScreenRenderEnv.seed = seed
    OffScreenRenderEnv.set_init_state = set_init_state
