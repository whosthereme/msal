from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "revision_20261111_ideal_v1"
ACTION_OFFSET = 110_000
EULER_STEPS = 10
ROW_BATCH = 4096


def action_error(model, rows, descriptor, cache, seed, weights, draws, stride) -> float:
    vals = []
    with torch.no_grad():
        for draw in range(draws):
            total, count = 0.0, 0.0
            for i in range(0, len(rows), ROW_BATCH):
                part = rows[i:i + ROW_BATCH]
                batch = cache.batch(part, seed=seed + ACTION_OFFSET + stride * draw + i)
                actions, masks, state = batch["actions"], batch["masks"], batch["state"]
                gen = torch.Generator(device=actions.device).manual_seed(int(seed + ACTION_OFFSET + stride * draw + i + 1))
                x = torch.randn(actions.shape, generator=gen, device=actions.device)
                n = actions.shape[0]
                d = descriptor.expand(n, -1)
                for s in range(EULER_STEPS):
                    t = torch.full((n,), 1.0 - s / EULER_STEPS, device=actions.device)
                    x = x - model.forward_descriptor(state, x, t, d) / EULER_STEPS
                error = (x - actions) ** 2 * weights[None, None, :]
                per_row = error.sum(dim=-1) / weights.sum().clamp_min(1e-12)
                total += float((per_row * masks).sum())
                count += float(masks.sum())
            vals.append(total / max(count, 1.0))
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("seen", "zero"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-tasks", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    from msal_ext import hyb
    from msal_ext.hyb import e2, run_loto, run_nl
    from msal_exp.data_lang import SPLIT_SELECTION, load_lang_v5
    from lib.frozen_data import DeviceCache
    from lib.pool_v5 import source_task_ids
    from lib.structured_field import channel_weights
    from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, nearest_source_task

    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("TRAIN_STEPS is not the published 12,800; unset G12_TRAIN_STEPS")
    if args.smoke_steps:
        e2.TRAIN_STEPS = int(args.smoke_steps)
        args.output = args.output.with_name(args.output.name + "_SMOKE")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.stage}_seed{args.seed}.jsonl"
    done = set()
    if path.exists():
        done = {int(json.loads(x)["task"]) for x in path.read_text().splitlines() if x.strip()}

    seed, device = args.seed, args.device
    data = load_lang_v5("blind", verify_hash=not args.skip_hash)
    cache = DeviceCache(data, device)
    data.state_features = None
    all_ids = [int(t) for t in source_task_ids()]
    tasks = [int(t) for t in args.held_tasks.split(",") if t.strip()] if args.held_tasks else all_ids
    stored = hyb.stored_film_nl(seed)
    weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    info = {"protocol": PROTOCOL, "stage": args.stage, "seed": seed, "smoke": bool(args.smoke_steps),
            "data_sha256": data.sha256 or "not_verified", "environment": run_loto.environment(),
            "action_range": [float(data.action_chunks.min()), float(data.action_chunks.max())],
            "started_unix": time.time()}
    info_path = args.output / f"{args.stage}_seed{seed}_info.json"
    if not info_path.exists():
        info_path.write_text(json.dumps(info, indent=1))

    def describe(task):
        return torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([task]), dtype=torch.float32, device=device)

    def act(model, task):
        rows = data.rows_for_task(task, SPLIT_SELECTION)
        return action_error(model, rows, describe(task), cache, seed, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE), len(rows)

    def write(record):
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    if args.stage == "seen":
        if all(t in done for t in tasks):
            print("ideal seen: nothing to do", flush=True)
        else:
            started = time.time()
            model, _ = run_loto.train_arm("film_nl", seed=seed, fit_ids=all_ids, data=data, cache=cache, device=device)
            model.eval()
            train_seconds = time.time() - started
            for task in tasks:
                if task in done:
                    continue
                flow = run_loto.score_forward(model, "film_nl", held=task, seed=seed, data=data, cache=cache, device=device)
                err, n = act(model, task)
                write({"stage": "seen", "seed": seed, "task": task, "flow_selection": flow["main_selection"],
                       "flow_robustness": flow["main_robustness"], "action_selection": err, "rows": n,
                       "stored_zero_flow_selection": stored.get(task), "train_seconds": train_seconds})
                print(f"[ideal seen] seed={seed} task={task} flow={flow['main_selection']:.5f} "
                      f"zero_stored={stored.get(task, float('nan')):.5f} act={err:.5f}", flush=True)
    else:
        for task in tasks:
            if task in done:
                continue
            started = time.time()
            fit_ids = [t for t in all_ids if t != task]
            model, _ = run_loto.train_arm("film_nl", seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
            model.eval()
            flow = run_loto.score_forward(model, "film_nl", held=task, seed=seed, data=data, cache=cache, device=device)
            err, n = act(model, task)
            neighbour, distance = nearest_source_task(task, fit_ids)
            write({"stage": "zero", "seed": seed, "task": task, "flow_selection": flow["main_selection"],
                   "flow_robustness": flow["main_robustness"], "action_selection": err, "rows": n,
                   "anchor_bit_exact": stored.get(task) == flow["main_selection"],
                   "nearest_source_distance_m": float(distance), "seconds": time.time() - started})
            print(f"[ideal zero] seed={seed} task={task} flow={flow['main_selection']:.5f} "
                  f"anchor={'ok' if stored.get(task) == flow['main_selection'] else 'MISMATCH'} act={err:.5f} "
                  f"{time.time() - started:.0f}s", flush=True)
            del model
            torch.cuda.empty_cache()
    print(f"ideal {args.stage} seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
