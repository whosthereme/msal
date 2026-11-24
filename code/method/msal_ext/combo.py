from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from msal_ext import hyb
from msal_ext.hyb import (K, SHOTS, FEWSHOT_OFFSET, e2, held_episode_rows, invert_descriptor, make_blocks,
                            mean_mse, run_loto, run_nl, stored_film_nl, train_library)
from msal_exp.data_lang import LANG_NPZ, SPLIT_ROBUSTNESS, SPLIT_SELECTION, load_lang_v5
from lib.frozen_data import DeviceCache
from lib.pool_v5 import source_task_ids
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code

PROTOCOL = "revision_20261024_combo_v1"
SPLITS = (("selection", SPLIT_SELECTION, "SELECTION_OFFSET"), ("robustness", SPLIT_ROBUSTNESS, "ROBUSTNESS_OFFSET"))


def scored_blocks(R, lib, cache, data, held, descriptor, seed):
    return {name: make_blocks(R, lib, cache, data.rows_for_task(held, split), descriptor,
                              seed=seed + getattr(run_loto, offset), draws=NOISE_DRAWS)
            for name, split, offset in SPLITS}


def run_fold(*, held, seed, data, cache, episodes, device, stored):
    started = time.time()
    fit_ids = [t for t in source_task_ids() if t != held]
    R, _ = run_loto.train_arm("film_nl", seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    R.eval()
    anchor = run_loto.score_forward(R, "film_nl", held=held, seed=seed, data=data, cache=cache, device=device)
    lib, _ = train_library(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    d_star = torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([held]), dtype=torch.float32, device=device)
    zero = torch.zeros(K, dtype=torch.float64, device=device)
    star = scored_blocks(R, lib, cache, data, held, d_star, seed)
    record = {"held_task": held, "seed": seed, "anchor_float32": anchor["main_selection"],
              "stored_film_nl": stored.get(held),
              "zero_shot": {s: mean_mse(b, zero, weights) for s, b in star.items()},
              "inversion": {}, "combination": {}}
    record["anchor_bit_exact"] = record["stored_film_nl"] is not None and anchor["main_selection"] == record["stored_film_nl"]
    for k in SHOTS:
        rows = held_episode_rows(episodes, data, held, k)
        demo = make_blocks(R, lib, cache, rows, d_star, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        if k == max(SHOTS):
            code = weighted_ls_code(demo, weights)
            record["hybrid_16"] = {s: mean_mse(b, code, weights) for s, b in star.items()}
        d_inv = invert_descriptor(R, demo, d_star, weights)
        demo_inv = make_blocks(R, lib, cache, rows, d_inv, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        z = weighted_ls_code(demo_inv, weights)
        inv = scored_blocks(R, lib, cache, data, held, d_inv, seed)
        record["inversion"][str(k)] = {s: mean_mse(b, zero, weights) for s, b in inv.items()}
        record["combination"][str(k)] = {s: mean_mse(b, z, weights) for s, b in inv.items()}
        record["combination"][str(k)]["code_norm"] = float(z.norm())
    record["seconds"] = time.time() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--held-tasks", default="")
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
    data = load_lang_v5("blind", verify_hash=not args.skip_hash)
    episodes = np.load(LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
    cache = DeviceCache(data, args.device)
    data.state_features = None
    stored = stored_film_nl(args.seed)
    info = {"protocol": PROTOCOL, "seed": args.seed, "data_sha256": data.sha256 or "not_verified_smoke",
            "environment": run_loto.environment(), "code_sha256": run_loto.code_hashes(), "shots": SHOTS,
            "smoke": bool(args.smoke_steps), "started_unix": time.time(), "rollouts_executed": 0,
            "sealed_rows_read": 0, "hyb_module": hyb.PROTOCOL}
    if not (args.output / "run_info.json").exists():
        (args.output / "run_info.json").write_text(json.dumps(info, indent=1))
    held_list = [int(t) for t in args.held_tasks.split(",") if t.strip()] if args.held_tasks else list(source_task_ids())
    for held in held_list:
        if held in done:
            continue
        try:
            record = run_fold(held=held, seed=args.seed, data=data, cache=cache, episodes=episodes,
                              device=args.device, stored=stored)
        except FloatingPointError:
            record = run_fold(held=held, seed=args.seed, data=data, cache=cache, episodes=episodes,
                              device=args.device, stored=stored)
            record["retrained_after_nan"] = True
        with folds_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        c, i = record["combination"], record["inversion"]
        print(f"[combo] seed={args.seed} held={held:2d} anchor={'ok' if record['anchor_bit_exact'] else 'MISMATCH'} "
              f"zero={record['zero_shot']['selection']:.5f} inv16={i['16']['selection']:.5f} "
              f"combo16={c['16']['selection']:.5f} inv1={i['1']['selection']:.5f} combo1={c['1']['selection']:.5f} "
              f"{record['seconds']:.0f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
