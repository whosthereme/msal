from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from msal_ext import run_nl
from msal_exp import run_loto
from msal_exp.arms import _trunk_hidden
from msal_exp.data_lang import LANG_NPZ, SPLIT_ROBUSTNESS, SPLIT_SELECTION, SPLIT_TRAIN, load_lang_v5
from joint_sal_velocity import output_orthogonality_error
from lib.frozen_data import DeviceCache
from lib.nonlinear_atom_field import NonlinearAtomHead
from lib.pool_v5 import source_task_ids
from lib.structured_field import channel_weights, group_sparsity_penalty
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code, weighted_mse

e2 = run_loto.e2
PROTOCOL = "revision_20261024_hybrid_v1"
K, BOTTLENECK = 16, 92
SHOTS = (1, 2, 4, 8, 16)
STAGE2_RNG_OFFSET, STAGE2_BATCH_OFFSET, STAGE2_INIT_OFFSET = 1, 50_000_000, 2
FEWSHOT_OFFSET = 60_000
DRAW_STRIDE = 1_000_003
INVERSION_STEPS, INVERSION_LR = 200, 1e-2
STORED = Path("/home/ubuntu/work/manuscript6/runs/n1n2")


class ResidualLibrary(nn.Module):

    def __init__(self, hidden_dim: int, chunk: int, action: int, task_count: int) -> None:
        super().__init__()
        self.chunk, self.action = chunk, action
        self.atom_out = NonlinearAtomHead(hidden_dim, K, chunk * action, BOTTLENECK)
        self.codes = nn.Parameter(torch.randn(task_count, K) * 0.02)

    def fields(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.atom_out(hidden).reshape(hidden.shape[0], K, self.chunk, self.action)


def new_library(R, seed: int, task_count: int, device: str) -> ResidualLibrary:
    torch.manual_seed(seed + STAGE2_INIT_OFFSET)
    return ResidualLibrary(R.config.hidden_dim, R.config.chunk_length, R.config.action_dim, task_count).to(device)


def train_library(R, *, seed: int, fit_ids: list[int], data, cache, device: str):
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    lib = new_library(R, seed, len(fit_ids), device)
    lookup = {t: i for i, t in enumerate(fit_ids)}
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    rows_by_task = {t: data.rows_for_task(t, SPLIT_TRAIN) for t in fit_ids}
    optimizer = torch.optim.AdamW(lib.parameters(), lr=e2.LEARNING_RATE, weight_decay=e2.WEIGHT_DECAY)
    rng = np.random.default_rng(seed + STAGE2_RNG_OFFSET)
    per_task = max(e2.BATCH_SIZE // len(fit_ids), 1)
    loss = torch.tensor(float("nan"))
    for step in range(1, e2.TRAIN_STEPS + 1):
        picked = np.concatenate([rng.choice(rows_by_task[t], size=per_task, replace=True) for t in fit_ids])
        batch = cache.batch(picked, seed=seed * 7919 + step + STAGE2_BATCH_OFFSET)
        task_rows = torch.tensor([lookup[int(v)] for v in data.task_ids[picked]], device=device)
        with torch.no_grad():
            base = R.forward_with_rows(batch["state"], batch["x_t"], batch["time"], task_rows)
            hidden = _trunk_hidden(R, batch["state"], batch["x_t"], batch["time"])
        atoms = lib.fields(hidden)
        prediction = base + torch.einsum("bk,bktd->btd", lib.codes[task_rows], atoms)
        loss = (e2.masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
                + e2.ORTHOGONALITY_WEIGHT * output_orthogonality_error(atoms)
                + e2.GROUP_SPARSITY_WEIGHT * group_sparsity_penalty(lib.codes, atoms))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if not torch.isfinite(loss):
        raise FloatingPointError("stage-2 loss is not finite")
    lib.eval()
    return lib, float(loss.detach())


def make_blocks(R, lib, cache, rows, descriptor, *, seed: int, draws: int) -> list[dict]:
    blocks = []
    with torch.no_grad():
        for draw in range(draws):
            batch = cache.batch(rows, seed=seed + DRAW_STRIDE * draw)
            n = batch["state"].shape[0]
            base = R.forward_descriptor(batch["state"], batch["x_t"], batch["time"], descriptor.expand(n, -1))
            atoms = lib.fields(_trunk_hidden(R, batch["state"], batch["x_t"], batch["time"]))
            blocks.append({"base": base.double(), "atoms": atoms.double(), "target": batch["target"].double(),
                           "mask": batch["masks"].double(), "batch": batch})
    return blocks


def mean_mse(blocks, code, weights) -> float:
    return float(np.mean([weighted_mse(b, code, weights) for b in blocks]))


def invert_descriptor(R, blocks, d0, weights) -> torch.Tensor:
    hiddens = []
    with torch.no_grad():
        for b in blocks:
            hiddens.append(_trunk_hidden(R, b["batch"]["state"], b["batch"]["x_t"], b["batch"]["time"]))
    d = d0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([d], lr=INVERSION_LR)
    chunk, action = R.config.chunk_length, R.config.action_dim
    for _ in range(INVERSION_STEPS):
        loss = 0.0
        for b, hidden in zip(blocks, hiddens):
            n = hidden.shape[0]
            dd = d.expand(n, -1)
            modulated = hidden * (1.0 + R.film_gamma(dd)) + R.film_beta(dd)
            prediction = (R.film_head(modulated) + R.film_mlp(modulated)).reshape(n, chunk, action)
            loss = loss + e2.masked_channel_loss(prediction, b["batch"]["target"], b["batch"]["masks"], weights)
        optimizer.zero_grad(set_to_none=True)
        (loss / len(blocks)).backward()
        optimizer.step()
    return d.detach()


def held_episode_rows(episodes: np.ndarray, data, held: int, k: int) -> np.ndarray:
    rows = data.rows_for_task(held, SPLIT_TRAIN)
    chosen = np.sort(np.unique(episodes[rows]))[:k]
    return rows[np.isin(episodes[rows], chosen)]


def stored_film_nl(seed: int) -> dict[int, float]:
    path = STORED / f"blind_nl_film_nl_seed{seed}" / "folds.jsonl"
    return {int(r["held_task"]): float(r["main_selection"])
            for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def run_fold(*, held: int, seed: int, data, cache, episodes, device: str, stored: dict) -> dict:
    started = time.time()
    all_ids = list(source_task_ids())
    fit_ids = [t for t in all_ids if t != held]
    R, _ = run_loto.train_arm("film_nl", seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    R.eval()
    anchor = run_loto.score_forward(R, "film_nl", held=held, seed=seed, data=data, cache=cache, device=device)
    t_stage1 = time.time() - started
    lib, stage2_loss = train_library(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    t_stage2 = time.time() - started - t_stage1

    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    d_star = torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([held]), dtype=torch.float32, device=device)
    scored = {name: make_blocks(R, lib, cache, data.rows_for_task(held, split), d_star, seed=seed + offset,
                                draws=NOISE_DRAWS)
              for name, split, offset in (("selection", SPLIT_SELECTION, run_loto.SELECTION_OFFSET),
                                          ("robustness", SPLIT_ROBUSTNESS, run_loto.ROBUSTNESS_OFFSET))}
    zero = torch.zeros(K, dtype=torch.float64, device=device)
    record = {"held_task": held, "seed": seed, "anchor_float32": anchor["main_selection"],
              "stored_film_nl": stored.get(held), "stage2_final_loss": stage2_loss,
              "zero_shot": {s: mean_mse(b, zero, weights) for s, b in scored.items()},
              "hybrid": {}, "inversion": {}}
    record["anchor_bit_exact"] = record["stored_film_nl"] is not None and anchor["main_selection"] == record["stored_film_nl"]

    for k in SHOTS:
        rows = held_episode_rows(episodes, data, held, k)
        demo = make_blocks(R, lib, cache, rows, d_star, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        code = weighted_ls_code(demo, weights)
        record["hybrid"][str(k)] = {s: mean_mse(b, code, weights) for s, b in scored.items()}
        record["hybrid"][str(k)].update(code_norm=float(code.norm()), demo_rows=int(len(rows)))
        d_inv = invert_descriptor(R, demo, d_star, weights)
        inv_scored = {name: make_blocks(R, lib, cache, data.rows_for_task(held, split), d_inv, seed=seed + offset,
                                        draws=NOISE_DRAWS)
                      for name, split, offset in (("selection", SPLIT_SELECTION, run_loto.SELECTION_OFFSET),
                                                  ("robustness", SPLIT_ROBUSTNESS, run_loto.ROBUSTNESS_OFFSET))}
        record["inversion"][str(k)] = {s: mean_mse(b, zero, weights) for s, b in inv_scored.items()}
        record["inversion"][str(k)]["descriptor_shift"] = float((d_inv - d_star).norm())
        if k == max(SHOTS):
            random_lib = new_library(R, seed, len(fit_ids), device).eval()
            rdemo = make_blocks(R, random_lib, cache, rows, d_star, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
            rcode = weighted_ls_code(rdemo, weights)
            rscored = {name: make_blocks(R, random_lib, cache, data.rows_for_task(held, split), d_star,
                                         seed=seed + offset, draws=NOISE_DRAWS)
                       for name, split, offset in (("selection", SPLIT_SELECTION, run_loto.SELECTION_OFFSET),
                                                   ("robustness", SPLIT_ROBUSTNESS, run_loto.ROBUSTNESS_OFFSET))}
            record["random_atoms_16"] = {s: mean_mse(b, rcode, weights) for s, b in rscored.items()}
    record["seconds"] = {"stage1": t_stage1, "stage2": t_stage2, "total": time.time() - started}
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
            "inversion": {"steps": INVERSION_STEPS, "lr": INVERSION_LR}, "smoke": bool(args.smoke_steps),
            "started_unix": time.time(), "rollouts_executed": 0, "sealed_rows_read": 0}
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
        h = record["hybrid"]
        print(f"[hyb] seed={args.seed} held={held:2d} anchor={'ok' if record['anchor_bit_exact'] else 'MISMATCH'} "
              f"zero={record['zero_shot']['selection']:.5f} k1={h['1']['selection']:.5f} k4={h['4']['selection']:.5f} "
              f"k16={h['16']['selection']:.5f} inv16={record['inversion']['16']['selection']:.5f} "
              f"rand16={record['random_atoms_16']['selection']:.5f} {record['seconds']['total']:.0f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
