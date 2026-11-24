from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from msal_ext import hyb
from msal_ext.hyb import (FEWSHOT_OFFSET, K, SHOTS, e2, held_episode_rows, invert_descriptor, make_blocks,
                            mean_mse, run_loto, run_nl, train_library)
from msal_ext.nl import FiLMNonlinearReadout, PlainNonlinearReadout
from lib import g39_data
from lib.frozen_data import DeviceCache
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code

PROTOCOL = "revision_20261024_libero40_confirmation_v1"
STOP = {"the", "a", "an", "and", "it", "of", "to"}
TASKS_JSON = g39_data.SUBSTRATE_DIR / "libero40_tasks.json"
SPLITS = (("selection", g39_data.SPLIT_VAL, "SELECTION_OFFSET"), ("robustness", g39_data.SPLIT_TEST, "ROBUSTNESS_OFFSET"))

e2.DEAD_ACTION_DIMS = tuple(g39_data.DEAD_ACTION_DIMS)
assert run_loto.SPLIT_TRAIN == g39_data.SPLIT_TRAIN and run_loto.SPLIT_SELECTION == g39_data.SPLIT_VAL
assert run_loto.SPLIT_ROBUSTNESS == g39_data.SPLIT_TEST

INSTRUCTIONS = {int(k): v["instruction"] for k, v in json.loads(TASKS_JSON.read_text())["tasks"].items()}
VOCAB = sorted({w for text in INSTRUCTIONS.values() for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP})


def word_descriptor(task_ids) -> np.ndarray:
    index = {w: i for i, w in enumerate(VOCAB)}
    rows = []
    for task_id in task_ids:
        row = np.zeros(len(VOCAB) + 1)
        row[-1] = 1.0
        for w in set(re.findall(r"[a-z]+", INSTRUCTIONS[int(task_id)].lower())):
            if w in index:
                row[index[w]] = 1.0
        rows.append(row)
    return np.stack(rows)


run_nl.arms.DESCRIPTORS["l40_words"] = word_descriptor
run_nl.arms.ARMS.update({
    "l40_film_nl": {"family": "film", "descriptor": "l40_words", "reg": False},
    "l40_plain_nl": {"family": "plain", "descriptor": None, "reg": False},
    "l40_film": {"family": "film", "descriptor": "l40_words", "reg": False},
})
run_nl.BUILDERS["l40_film_nl"] = lambda config, ids: FiLMNonlinearReadout(config, ids, word_descriptor)
run_nl.BUILDERS["l40_plain_nl"] = lambda config, ids: PlainNonlinearReadout(config, ids)
run_nl.CONDITIONED["l40_film_nl"] = ("film_gamma", "film_beta")
run_nl.CONDITIONED["l40_plain_nl"] = ()


def scored(R, lib, cache, data, held, descriptor, seed):
    return {name: make_blocks(R, lib, cache, data.rows_for_task(held, split), descriptor,
                              seed=seed + getattr(run_loto, offset), draws=NOISE_DRAWS)
            for name, split, offset in SPLITS}


def zero_shot(arm, *, held, seed, data, cache, device, fit_ids):
    model, _ = run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    model.eval()
    out = run_loto.score_forward(model, arm, held=held, seed=seed, data=data, cache=cache, device=device)
    return model, {"selection": out["main_selection"], "robustness": out["main_robustness"]}


def run_fold(*, held, seed, data, cache, device):
    started = time.time()
    fit_ids = [t for t in sorted(INSTRUCTIONS) if t != held]
    R, r_zero = zero_shot("l40_film_nl", held=held, seed=seed, data=data, cache=cache, device=device, fit_ids=fit_ids)
    plain, plain_zero = zero_shot("l40_plain_nl", held=held, seed=seed, data=data, cache=cache, device=device, fit_ids=fit_ids)
    del plain
    film, film_zero = zero_shot("l40_film", held=held, seed=seed, data=data, cache=cache, device=device, fit_ids=fit_ids)
    del film
    torch.cuda.empty_cache()
    lib, stage2_loss = train_library(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    d_star = torch.tensor(word_descriptor([held]), dtype=torch.float32, device=device)
    zero = torch.zeros(K, dtype=torch.float64, device=device)
    star = scored(R, lib, cache, data, held, d_star, seed)
    record = {"held_task": held, "seed": seed, "instruction": INSTRUCTIONS[held],
              "unseen_words": [w for w in set(re.findall(r"[a-z]+", INSTRUCTIONS[held].lower())) - STOP
                               if all(w not in re.findall(r"[a-z]+", INSTRUCTIONS[t].lower()) for t in fit_ids)],
              "zero_shot_float32": {"film_nl": r_zero, "plain_nl": plain_zero, "film": film_zero},
              "zero_shot": {s: mean_mse(b, zero, weights) for s, b in star.items()},
              "stage2_final_loss": stage2_loss, "hybrid": {}, "inversion": {}, "combination": {}}
    for k in SHOTS:
        rows = held_episode_rows(data.episode_indices, data, held, k)
        demo = make_blocks(R, lib, cache, rows, d_star, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        code = weighted_ls_code(demo, weights)
        record["hybrid"][str(k)] = {s: mean_mse(b, code, weights) for s, b in star.items()}
        d_inv = invert_descriptor(R, demo, d_star, weights)
        demo_inv = make_blocks(R, lib, cache, rows, d_inv, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        z = weighted_ls_code(demo_inv, weights)
        inv = scored(R, lib, cache, data, held, d_inv, seed)
        record["inversion"][str(k)] = {s: mean_mse(b, zero, weights) for s, b in inv.items()}
        record["combination"][str(k)] = {s: mean_mse(b, z, weights) for s, b in inv.items()}
        record["combination"][str(k)]["code_norm"] = float(z.norm())
        record["hybrid"][str(k)]["demo_rows"] = int(len(rows))
    record["seconds"] = time.time() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-tasks", required=True, help="range a-b of task ids, inclusive")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("TRAIN_STEPS is not the published 12,800; unset G12_TRAIN_STEPS")
    if args.smoke_steps:
        e2.TRAIN_STEPS = int(args.smoke_steps)
        args.output = args.output.with_name(args.output.name + "_SMOKE")
    first, last = (int(x) for x in args.held_tasks.split("-"))
    held_list = [t for t in sorted(INSTRUCTIONS) if first <= t <= last]
    args.output.mkdir(parents=True, exist_ok=True)
    folds_path = args.output / "folds.jsonl"
    done = set()
    if folds_path.exists():
        done = {int(json.loads(x)["held_task"]) for x in folds_path.read_text().splitlines() if x.strip()}
    data = g39_data.load_g39_data(verify_hash=not args.skip_hash)
    cache = DeviceCache(data, args.device)
    data.state_features = np.empty((0, g39_data.FROZEN_FEATURE_DIM), dtype=np.float16)
    info = {"protocol": PROTOCOL, "seed": args.seed, "held_tasks": held_list, "data_sha256": data.sha256 or "not_verified",
            "vocab": VOCAB, "environment": run_loto.environment(), "code_sha256": run_loto.code_hashes(),
            "dead_action_dims": list(e2.DEAD_ACTION_DIMS), "shots": SHOTS, "smoke": bool(args.smoke_steps),
            "started_unix": time.time(), "rollouts_executed": 0, "sealed_rows_read": 0, "hyb_module": hyb.PROTOCOL}
    if not (args.output / "run_info.json").exists():
        (args.output / "run_info.json").write_text(json.dumps(info, indent=1))
    for held in held_list:
        if held in done:
            continue
        try:
            record = run_fold(held=held, seed=args.seed, data=data, cache=cache, device=args.device)
        except FloatingPointError:
            record = run_fold(held=held, seed=args.seed, data=data, cache=cache, device=args.device)
            record["retrained_after_nan"] = True
        with folds_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        c, i = record["combination"], record["inversion"]
        print(f"[l40] seed={args.seed} held={held} zero={record['zero_shot']['selection']:.5f} "
              f"plain={record['zero_shot_float32']['plain_nl']['selection']:.5f} "
              f"film={record['zero_shot_float32']['film']['selection']:.5f} inv16={i['16']['selection']:.5f} "
              f"combo16={c['16']['selection']:.5f} {record['seconds']:.0f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
