from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from msal_ext import hyb
from msal_ext.combo import scored_blocks
from msal_ext.hyb import (K, SHOTS, FEWSHOT_OFFSET, e2, held_episode_rows, invert_descriptor, make_blocks,
                            mean_mse, run_loto, run_nl, stored_film_nl, train_library)
from msal_exp.data_lang import LANG_NPZ, load_lang_v5
from lib.frozen_data import DeviceCache
from lib.pool_v5 import source_task_ids
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import ORACLE_DRAWS, weighted_ls_code

PROTOCOL = "revision_20261025_coverage_v1"
COV_SEED = 20261025
LEVELS = (8, 16, 24, 32, 40)
COMBO = Path("/home/ubuntu/work/manuscript6/runs/combo")
ACTIVE_GPU0 = Path("/home/ubuntu/work/manuscript6/runs/cov/active_gpu0")
GPU0_UUID = "GPU-b7aa8026-3c97-0115-c172-4bc0d901b3df"


def sources(held: int, seed: int, n: int) -> list[int]:
    others = [int(t) for t in source_task_ids() if t != held]
    order = np.random.default_rng([COV_SEED, seed, held]).permutation(len(others))
    chosen = {others[i] for i in order[:n]}
    return [t for t in others if t in chosen]


def stored_combo(seed: int) -> dict[int, dict]:
    path = COMBO / f"seed{seed}" / "folds.jsonl"
    if not path.exists():
        return {}
    return {int(r["held_task"]): r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu0_foreign() -> int:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout
    apps = sum(1 for line in out.splitlines() if line.strip() == GPU0_UUID)
    mine = sum(1 for p in ACTIVE_GPU0.glob("*") if p.name.isdigit() and _alive(int(p.name)))
    return apps - mine


def wait_gpu0(tag: str) -> None:
    if gpu0_foreign() <= 0:
        return
    print(f"[cov] {tag} waiting: another process is on GPU 0", flush=True)
    clear = 0
    while clear < 5:
        time.sleep(60)
        clear = clear + 1 if gpu0_foreign() <= 0 else 0
    print(f"[cov] {tag} GPU 0 clear for five minutes; continuing", flush=True)


def run_fold(*, held, seed, n, data, cache, episodes, device, stored_nl, stored_c):
    started = time.time()
    fit_ids = sources(held, seed, n)
    R, _ = run_loto.train_arm("film_nl", seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    R.eval()
    anchor = run_loto.score_forward(R, "film_nl", held=held, seed=seed, data=data, cache=cache, device=device)
    lib, stage2_loss = train_library(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    d_star = torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([held]), dtype=torch.float32, device=device)
    zero = torch.zeros(K, dtype=torch.float64, device=device)
    star = scored_blocks(R, lib, cache, data, held, d_star, seed)
    record = {"held_task": held, "seed": seed, "level": n, "fit_ids": fit_ids,
              "film_nl_float32": anchor["main_selection"], "stage2_final_loss": stage2_loss,
              "zero_shot": {s: mean_mse(b, zero, weights) for s, b in star.items()},
              "hybrid": {}, "inversion": {}, "combination": {}}
    for k in SHOTS:
        rows = held_episode_rows(episodes, data, held, k)
        demo = make_blocks(R, lib, cache, rows, d_star, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        code = weighted_ls_code(demo, weights)
        record["hybrid"][str(k)] = {s: mean_mse(b, code, weights) for s, b in star.items()}
        d_inv = invert_descriptor(R, demo, d_star, weights)
        demo_inv = make_blocks(R, lib, cache, rows, d_inv, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        z = weighted_ls_code(demo_inv, weights)
        inv = scored_blocks(R, lib, cache, data, held, d_inv, seed)
        record["inversion"][str(k)] = {s: mean_mse(b, zero, weights) for s, b in inv.items()}
        record["combination"][str(k)] = {s: mean_mse(b, z, weights) for s, b in inv.items()}
    if n == 40:
        c = stored_c.get(held)
        checks = {"film_nl": stored_nl.get(held) == record["film_nl_float32"]}
        if c is not None:
            checks["zero_shot"] = c["zero_shot"] == record["zero_shot"]
            for part in ("inversion", "combination"):
                checks[part] = all({s: c[part][k][s] for s in ("selection", "robustness")} ==
                                   {s: record[part][k][s] for s in ("selection", "robustness")} for k in c[part])
            checks["hybrid_16"] = c["hybrid_16"] == {s: record["hybrid"]["16"][s] for s in ("selection", "robustness")}
        record["anchor"] = checks
        record["anchor_bit_exact"] = c is not None and all(checks.values())
    record["seconds"] = time.time() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--held-tasks", default="")
    parser.add_argument("--gpu", type=int, default=-1, help="physical GPU of this runner (0 activates the GPU 0 gate)")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("TRAIN_STEPS is not the published 12,800; unset G12_TRAIN_STEPS")
    if args.smoke_steps:
        e2.TRAIN_STEPS = int(args.smoke_steps)
        args.output = args.output.with_name(args.output.name + "_SMOKE")
    args.output.mkdir(parents=True, exist_ok=True)
    folds_path = args.output / "folds.jsonl"
    done = set()
    if folds_path.exists():
        done = {int(json.loads(x)["held_task"]) for x in folds_path.read_text().splitlines() if x.strip()}
    gate = args.gpu == 0
    if gate:
        ACTIVE_GPU0.mkdir(parents=True, exist_ok=True)
        (ACTIVE_GPU0 / str(os.getpid())).touch()
    try:
        data = load_lang_v5("blind", verify_hash=not args.skip_hash)
        episodes = np.load(LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
        cache = DeviceCache(data, args.device)
        data.state_features = None
        stored_nl, stored_c = stored_film_nl(args.seed), stored_combo(args.seed)
        info = {"protocol": PROTOCOL, "seed": args.seed, "level": args.level, "cov_seed": COV_SEED,
                "data_sha256": data.sha256 or "not_verified_smoke", "environment": run_loto.environment(),
                "code_sha256": run_loto.code_hashes(), "shots": SHOTS, "smoke": bool(args.smoke_steps),
                "started_unix": time.time(), "rollouts_executed": 0, "sealed_rows_read": 0, "hyb_module": hyb.PROTOCOL}
        if not (args.output / "run_info.json").exists():
            (args.output / "run_info.json").write_text(json.dumps(info, indent=1))
        held_list = ([int(t) for t in args.held_tasks.split(",") if t.strip()] if args.held_tasks
                     else list(source_task_ids()))
        for held in held_list:
            if held in done:
                continue
            if gate:
                wait_gpu0(f"seed={args.seed} n={args.level} held={held}")
            kwargs = dict(held=held, seed=args.seed, n=args.level, data=data, cache=cache, episodes=episodes,
                          device=args.device, stored_nl=stored_nl, stored_c=stored_c)
            try:
                record = run_fold(**kwargs)
            except FloatingPointError:
                record = run_fold(**kwargs)
                record["retrained_after_nan"] = True
            with folds_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            c, i = record["combination"], record["inversion"]
            anchor = "" if args.level != 40 else f" anchor={'ok' if record['anchor_bit_exact'] else 'MISMATCH'}"
            print(f"[cov] seed={args.seed} n={args.level} held={held:2d}{anchor} zero={record['zero_shot']['selection']:.5f} "
                  f"inv16={i['16']['selection']:.5f} combo16={c['16']['selection']:.5f} "
                  f"lib16={record['hybrid']['16']['selection']:.5f} {record['seconds']:.0f}s", flush=True)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        if gate:
            (ACTIVE_GPU0 / str(os.getpid())).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
