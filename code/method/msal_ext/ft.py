from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "revision_20261025_finetune_v1"
VARIANTS = ("readout", "all")
LRS = (1e-4, 1e-3)
STEPS = (25, 100, 400)
FT_BATCH = 512
FT_RNG_TAG = 7
FT_BATCH_OFFSET = 70_000_000
RUNS = Path("/home/ubuntu/work/manuscript6/runs")


def trainable(model, variant: str) -> list[torch.nn.Parameter]:
    if variant == "readout":
        return list(model.film_head.parameters()) + list(model.film_mlp.parameters())
    return list(model.parameters())


def stored_records(family: str, seed: int) -> dict[int, dict]:
    paths = ([RUNS / "combo" / f"seed{seed}" / "folds.jsonl"] if family == "dev"
             else sorted((RUNS / "l40").glob(f"seed{seed}_*/folds.jsonl")))
    out = {}
    for path in paths:
        for line in path.read_text().splitlines() if path.exists() else []:
            if line.strip():
                r = json.loads(line)
                out[int(r["held_task"])] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("dev", "l40"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--held-tasks", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=-1, help="physical GPU of this runner (0 activates the GPU 0 gate)")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    from msal_ext import hyb
    from msal_ext.cov import ACTIVE_GPU0, wait_gpu0
    from msal_ext.hyb import K, SHOTS, e2, held_episode_rows, mean_mse, new_library, run_loto, run_nl
    from lib.frozen_data import DeviceCache
    from lib.structured_field import channel_weights
    if args.family == "dev":
        from msal_ext.combo import scored_blocks as score
        from msal_exp.data_lang import LANG_NPZ, load_lang_v5
        from lib.pool_v5 import source_task_ids
        all_ids = [int(t) for t in source_task_ids()]
        descriptor = lambda held: run_nl.arms.DESCRIPTORS["saturated"]([held])
    else:
        from msal_ext import l40
        from msal_ext.l40 import scored as score
        from lib import g39_data
        all_ids = sorted(l40.INSTRUCTIONS)
        descriptor = lambda held: l40.word_descriptor([held])

    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("TRAIN_STEPS is not the published 12,800; unset G12_TRAIN_STEPS")
    steps_scored = STEPS
    if args.smoke_steps:
        e2.TRAIN_STEPS = int(args.smoke_steps)
        steps_scored = (1, 2)
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
        if args.family == "dev":
            data = load_lang_v5("blind", verify_hash=not args.skip_hash)
            episodes = np.load(LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
            cache = DeviceCache(data, args.device)
            data.state_features = None
            stored_nl = hyb.stored_film_nl(args.seed)
        else:
            data = g39_data.load_g39_data(verify_hash=not args.skip_hash)
            episodes = data.episode_indices
            cache = DeviceCache(data, args.device)
            data.state_features = np.empty((0, g39_data.FROZEN_FEATURE_DIM), dtype=np.float16)
            stored_nl = None
        stored = stored_records(args.family, args.seed)
        info = {"protocol": PROTOCOL, "family": args.family, "seed": args.seed, "variants": VARIANTS, "lrs": LRS,
                "steps": steps_scored, "batch": FT_BATCH, "shots": SHOTS, "data_sha256": data.sha256 or "not_verified",
                "environment": run_loto.environment(), "code_sha256": run_loto.code_hashes(),
                "smoke": bool(args.smoke_steps), "started_unix": time.time(), "rollouts_executed": 0,
                "sealed_rows_read": 0, "hyb_module": hyb.PROTOCOL}
        if not (args.output / "run_info.json").exists():
            (args.output / "run_info.json").write_text(json.dumps(info, indent=1))

        for held in [int(t) for t in args.held_tasks.split(",") if t.strip()]:
            if held in done:
                continue
            if gate:
                wait_gpu0(f"ft {args.family} seed={args.seed} held={held}")
            for attempt in (1, 2):
                try:
                    record = run_fold(args, held, all_ids, descriptor, data, cache, episodes, stored, stored_nl,
                                      score, steps_scored, hyb, e2, run_loto, held_episode_rows, mean_mse,
                                      new_library, channel_weights, K)
                    break
                except FloatingPointError:
                    if attempt == 2:
                        raise
            record["attempt"] = attempt
            with folds_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            best = min((v["selection"], key) for key, v in record["finetune"].items())
            print(f"[ft] {args.family} seed={args.seed} held={held} anchor={'ok' if record['anchor_bit_exact'] else 'MISMATCH'} "
                  f"zero={record['zero_shot']['selection']:.5f} best={best[0]:.5f} ({best[1]}) "
                  f"{record['seconds']:.0f}s", flush=True)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        if gate:
            (ACTIVE_GPU0 / str(os.getpid())).unlink(missing_ok=True)


def run_fold(args, held, all_ids, descriptor, data, cache, episodes, stored, stored_nl, score, steps_scored,
             hyb, e2, run_loto, held_episode_rows, mean_mse, new_library, channel_weights, K):
    started = time.time()
    seed, device = args.seed, args.device
    fit_ids = [t for t in all_ids if t != held]
    if args.family == "dev":
        R, _ = run_loto.train_arm("film_nl", seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
        R.eval()
        r32 = run_loto.score_forward(R, "film_nl", held=held, seed=seed, data=data, cache=cache, device=device)
        float32_ok = stored_nl.get(held) == r32["main_selection"]
    else:
        from msal_ext import l40
        R, r32 = l40.zero_shot("l40_film_nl", held=held, seed=seed, data=data, cache=cache, device=device,
                               fit_ids=fit_ids)
        s = stored.get(held, {}).get("zero_shot_float32", {}).get("film_nl")
        float32_ok = s is not None and s == r32
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    d_star = torch.tensor(descriptor(held), dtype=torch.float32, device=device)
    dummy = new_library(R, seed, len(fit_ids), device).eval()
    zero = torch.zeros(K, dtype=torch.float64, device=device)
    zero_shot = {s: mean_mse(b, zero, weights) for s, b in score(R, dummy, cache, data, held, d_star, seed).items()}
    ref = stored.get(held)
    record = {"held_task": held, "seed": seed, "family": args.family, "zero_shot": zero_shot,
              "anchor": {"float32": bool(float32_ok), "zero_shot": ref is not None and ref["zero_shot"] == zero_shot},
              "finetune": {}, "finetune_seconds": {}}
    record["anchor_bit_exact"] = all(record["anchor"].values())
    for k in (1, 2, 4, 8, 16):
        rows = held_episode_rows(episodes, data, held, k)
        for v_i, variant in enumerate(("readout", "all")):
            for lr_i, lr in enumerate((1e-4, 1e-3)):
                t0, t_score = time.time(), 0.0
                model = copy.deepcopy(R)
                for p in model.parameters():
                    p.requires_grad_(False)
                params = trainable(model, variant)
                for p in params:
                    p.requires_grad_(True)
                optimizer = torch.optim.Adam(params, lr=lr)
                rng = np.random.default_rng([seed, held, k, v_i, lr_i, FT_RNG_TAG])
                for step in range(1, max(steps_scored) + 1):
                    picked = rng.choice(rows, size=FT_BATCH, replace=True)
                    batch = cache.batch(picked, seed=seed * 7919 + step + FT_BATCH_OFFSET + 1_000 * k)
                    prediction = model.forward_descriptor(batch["state"], batch["x_t"], batch["time"],
                                                          d_star.expand(len(picked), -1))
                    loss = e2.masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"fine-tuning loss not finite (k={k}, {variant}, lr={lr})")
                    if step in steps_scored:
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        t1 = time.time()
                        key = f"k{k}_{variant}_lr{lr:g}_s{step}"
                        with torch.no_grad():
                            scores = {s: mean_mse(b, zero, weights)
                                      for s, b in score(model, dummy, cache, data, held, d_star, seed).items()}
                        record["finetune"][key] = {**scores, "train_loss": float(loss.detach()),
                                                   "training_seconds": t1 - t0 - t_score}
                        t_score += time.time() - t1
                record["finetune_seconds"][f"k{k}_{variant}_lr{lr:g}"] = {"training": time.time() - t0 - t_score,
                                                                          "scoring": t_score}
                del model, optimizer
    record["seconds"] = time.time() - started
    return record


if __name__ == "__main__":
    main()
