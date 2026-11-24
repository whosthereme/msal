from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

SEED, SHOTS = 148021, (1, 4, 16)
METHODS = ("setup", "code_only", "msal", "ft_all_100", "ft_all_400", "ft_readout_100", "ft_readout_400")


def status_kb(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise KeyError(field)


class RowCache:

    def __init__(self, state, actions, masks):
        self.state_features, self.action_chunks, self.action_valid_masks = state, actions, masks
        self.device = "cpu"

    def batch(self, rows, *, seed: int):
        index = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        state = self.state_features[index].float()
        actions = self.action_chunks[index]
        masks = self.action_valid_masks[index]
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(actions.shape, generator=generator)
        t = torch.rand(actions.shape[0], generator=generator) * 0.998 + 0.001
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        return {"state": state, "actions": actions, "masks": masks, "time": t, "x_t": x_t,
                "target": noise - actions}


def prep(out: Path) -> None:
    from msal_ext import run_nl
    from msal_ext.hyb import e2, held_episode_rows, run_loto, train_library
    from msal_exp.data_lang import LANG_NPZ, load_lang_v5
    from lib.frozen_data import DeviceCache
    from lib.pool_v5 import source_task_ids
    from lib.structured_field import channel_weights
    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("unset G12_TRAIN_STEPS")
    device = "cuda"
    data = load_lang_v5("blind", verify_hash=True)
    episodes = np.load(LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
    cache = DeviceCache(data, device)
    all_ids = [int(t) for t in source_task_ids()]
    held = all_ids[0]
    fit_ids = [t for t in all_ids if t != held]
    t0 = time.time()
    R, _ = run_loto.train_arm("film_nl", seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    R.eval()
    lib, _ = train_library(R, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    train_seconds = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    torch.save(R.cpu(), out / "R.pt")
    torch.save(lib.cpu(), out / "lib.pt")
    rows = held_episode_rows(episodes, data, held, max(SHOTS))
    per_k = {k: np.flatnonzero(np.isin(rows, held_episode_rows(episodes, data, held, k))).tolist() for k in SHOTS}
    index = torch.as_tensor(rows, dtype=torch.long, device=device)
    torch.save({"state": cache.state_features[index].cpu(), "actions": cache.action_chunks[index].cpu(),
                "masks": cache.action_valid_masks[index].cpu(), "rows_for_k": per_k,
                "d_star": torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([held]), dtype=torch.float32),
                "weights": channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device="cpu"),
                "held": held}, out / "rows.pt")
    n_all = sum(p.numel() for p in R.parameters())
    n_readout = sum(p.numel() for p in list(R.film_head.parameters()) + list(R.film_mlp.parameters()))
    info = {"held": held, "seed": SEED, "train_seconds_gpu": train_seconds, "head_parameters": n_all,
            "readout_parameters": n_readout, "library_parameters": sum(p.numel() for p in lib.atom_out.parameters()),
            "descriptor_dim": int(R.film_gamma.in_features), "rows_for_k": {k: len(v) for k, v in per_k.items()},
            "state_dim": int(cache.state_features.shape[1])}
    (out / "prep.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def run(d: Path, method: str, k: int, threads: int, reps: int = 3) -> dict:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from msal_ext import run_nl
    from msal_ext.hyb import e2, invert_descriptor, make_blocks
    from run_e35_1_zero_shot_code_loto import ORACLE_DRAWS, weighted_ls_code
    R = torch.load(d / "R.pt", weights_only=False).eval()
    lib = torch.load(d / "lib.pt", weights_only=False).eval()
    blob = torch.load(d / "rows.pt", weights_only=False)
    cache = RowCache(blob["state"], blob["actions"], blob["masks"])
    rows = np.asarray(blob["rows_for_k"][k])
    d_star, weights = blob["d_star"], blob["weights"]
    seed = SEED + 60_000

    def code_only():
        demo = make_blocks(R, lib, cache, rows, d_star, seed=seed, draws=ORACLE_DRAWS)
        return weighted_ls_code(demo, weights)

    def msal():
        demo = make_blocks(R, lib, cache, rows, d_star, seed=seed, draws=ORACLE_DRAWS)
        d_inv = invert_descriptor(R, demo, d_star, weights)
        demo_inv = make_blocks(R, lib, cache, rows, d_inv, seed=seed, draws=ORACLE_DRAWS)
        return weighted_ls_code(demo_inv, weights)

    def finetune(variant: str, steps: int):
        import copy
        model = copy.deepcopy(R)
        for p in model.parameters():
            p.requires_grad_(False)
        params = (list(model.film_head.parameters()) + list(model.film_mlp.parameters())
                  if variant == "readout" else list(model.parameters()))
        for p in params:
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(params, lr=1e-4)
        rng = np.random.default_rng([SEED, k, 7])
        for step in range(1, steps + 1):
            picked = rng.choice(rows, size=512, replace=True)
            batch = cache.batch(picked, seed=SEED * 7919 + step + 70_000_000 + 1_000 * k)
            prediction = model.forward_descriptor(batch["state"], batch["x_t"], batch["time"],
                                                  d_star.expand(len(picked), -1))
            loss = e2.masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return model

    jobs = {"setup": lambda: None, "code_only": code_only, "msal": msal,
            "ft_all_100": lambda: finetune("all", 100), "ft_all_400": lambda: finetune("all", 400),
            "ft_readout_100": lambda: finetune("readout", 100), "ft_readout_400": lambda: finetune("readout", 400)}
    rss0 = status_kb("VmRSS")
    Path("/proc/self/clear_refs").write_text("5")
    times = []
    for rep in range(reps):
        t0 = time.perf_counter()
        result = jobs[method]()
        times.append(time.perf_counter() - t0)
        if rep == 0:
            peak = status_kb("VmHWM")
        del result
    return {"method": method, "k": k, "threads": threads, "rows": int(len(rows)), "seconds": times,
            "median_seconds": statistics.median(times), "rss_after_setup_mb": rss0 / 1024,
            "peak_extra_mb": (peak - rss0) / 1024, "torch": torch.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prep", "run"))
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "prep":
        prep(args.dir)
    else:
        if torch.cuda.is_available():
            raise RuntimeError("run must be CPU-only: set CUDA_VISIBLE_DEVICES=")
        res = run(args.dir, args.method, args.k, args.threads, args.reps)
        with (args.dir / "bench.jsonl").open("a") as handle:
            handle.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
