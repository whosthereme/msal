from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lib.frozen_data import DeviceCache
from lib.pool_v5 import build_pool, source_task_ids
from joint_sal_velocity import output_orthogonality_error
from lib.structured_field import (
    StructuredCodeFieldV5,
    channel_weights,
    group_sparsity_penalty,
)
from lib.v5_data import load_v5_data

import run_e2_structured_training as e2

PROTOCOL = "sal_v2_identifiable_g35_e35_1_zero_shot_code_loto_v1"

SPLIT_TRAIN, SPLIT_SELECTION = 0, 1

PASS_WIN_FLOOR = 27
FALSIFIER_WIN_CEILING = 14
TASK_COUNT = 41

NOISE_DRAWS = 8

ANCHOR_SELECTION_TOLERANCE = 0.0


def train_field(
    *,
    feature_set: str,
    seed: int,
    fit_task_ids: list[int],
    data: Any,
    cache: DeviceCache,
    device: str,
) -> tuple[StructuredCodeFieldV5, dict[int, int]]:
    lookup = {task_id: index for index, task_id in enumerate(fit_task_ids)}
    torch.manual_seed(seed)
    config = e2.JointSALConfig(
        state_dim=data.state_dim,
        chunk_length=int(data.action_chunks.shape[1]),
        action_dim=int(data.action_chunks.shape[2]),
    )
    model = StructuredCodeFieldV5(config, fit_task_ids, feature_set=feature_set).to(device)
    weights = channel_weights(config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)

    rows_by_task = {t: data.rows_for_task(t, SPLIT_TRAIN) for t in fit_task_ids}
    for task_id, rows in rows_by_task.items():
        if len(rows) == 0:
            raise RuntimeError(f"v5 source task {task_id} has no training rows")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=e2.LEARNING_RATE,
        weight_decay=e2.WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    per_task = max(e2.BATCH_SIZE // len(fit_task_ids), 1)

    for step in range(1, e2.TRAIN_STEPS + 1):
        picked = np.concatenate(
            [rng.choice(rows_by_task[t], size=per_task, replace=True) for t in fit_task_ids]
        )
        batch = cache.batch(picked, seed=seed * 7919 + step)
        task_rows = torch.tensor(
            [lookup[int(v)] for v in data.task_ids[picked]], device=device
        )
        codes = model.codes_for_rows(task_rows)
        base, atoms = model.components(batch["state"], batch["x_t"], batch["time"], None)
        prediction = base + torch.einsum("bk,bktd->btd", codes, atoms)
        fit = e2.masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
        loss = (
            fit
            + e2.ORTHOGONALITY_WEIGHT * output_orthogonality_error(atoms)
            + e2.GROUP_SPARSITY_WEIGHT * group_sparsity_penalty(model.structured_codes, atoms)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, lookup


def nearest_source_task(held: int, fit_task_ids: list[int]) -> tuple[int, float]:
    by_id = {task.task_id: task for task in build_pool()}
    hx, hy = by_id[held].target_xy_m
    best, best_distance = None, float("inf")
    for task_id in fit_task_ids:
        x, y = by_id[task_id].target_xy_m
        distance = float(np.hypot(x - hx, y - hy))
        if distance < best_distance:
            best, best_distance = task_id, distance
    return int(best), best_distance


ORACLE_ROWS_CAP = 1024
ORACLE_DRAWS = 4


def collect_blocks(
    model: StructuredCodeFieldV5,
    data: Any,
    cache: DeviceCache,
    held: int,
    *,
    split: int,
    seed: int,
    draws: int,
    rows_cap: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    rows = data.rows_for_task(held, split)
    if len(rows) == 0:
        raise RuntimeError(f"held task {held} has no rows in split {split}")
    if rows_cap is not None:
        rows = np.random.default_rng(seed).permutation(rows)[:rows_cap]
    blocks = []
    with torch.no_grad():
        for draw in range(draws):
            batch = cache.batch(rows, seed=seed + 1_000_003 * draw)
            base, atoms = model.components(
                batch["state"], batch["x_t"], batch["time"], None
            )
            blocks.append(
                {
                    "base": base.double(),
                    "atoms": atoms.double(),
                    "target": batch["target"].double(),
                    "mask": batch["masks"].double(),
                }
            )
    return blocks


def weighted_mse(block: dict[str, torch.Tensor], code: torch.Tensor, weights: torch.Tensor) -> float:
    prediction = block["base"] + torch.einsum(
        "k,bktd->btd", code.double(), block["atoms"]
    )
    error = (prediction - block["target"]) ** 2 * weights.double()[None, None, :]
    per_row = error.sum(dim=-1) / weights.double().sum().clamp_min(1e-12)
    return float(
        (per_row * block["mask"]).sum() / block["mask"].sum().clamp_min(1.0)
    )


def weighted_ls_code(
    blocks: list[dict[str, torch.Tensor]], weights: torch.Tensor, *, ridge: float = 0.0
) -> torch.Tensor:
    channel = weights.double().sqrt()[None, None, :]
    atom_count = int(blocks[0]["atoms"].shape[1])
    normal = torch.zeros(
        atom_count, atom_count, dtype=torch.float64, device=blocks[0]["atoms"].device
    )
    rhs = torch.zeros(atom_count, dtype=torch.float64, device=normal.device)
    for block in blocks:
        block_scale = 1.0 / float(block["mask"].sum().clamp_min(1.0))
        scale = channel * block["mask"].sqrt()[:, :, None]
        design = (block["atoms"] * scale[:, None]).permute(0, 2, 3, 1).reshape(-1, atom_count)
        residual = ((block["target"] - block["base"]) * scale).reshape(-1)
        normal += block_scale * (design.T @ design)
        rhs += block_scale * (design.T @ residual)
    if ridge > 0:
        normal = normal + ridge * torch.eye(atom_count, dtype=normal.dtype, device=normal.device)
    return torch.linalg.lstsq(normal, rhs[:, None]).solution[:, 0]


def run_fold(
    *,
    held: int,
    feature_set: str,
    seed: int,
    data: Any,
    cache: DeviceCache,
    device: str,
) -> dict[str, Any]:
    all_ids = list(source_task_ids())
    fit_ids = [t for t in all_ids if t != held]
    if len(fit_ids) != len(all_ids) - 1:
        raise RuntimeError(f"held task {held} is not a v5 source task")

    started = time.time()
    model, _ = train_field(
        feature_set=feature_set,
        seed=seed,
        fit_task_ids=fit_ids,
        data=data,
        cache=cache,
        device=device,
    )
    model.eval()

    with torch.no_grad():
        phi = model.codes_for_task_ids([held])[0].detach()
        source = model.codes_for_task_ids(fit_ids).detach()
    mean_code = source.mean(dim=0)
    neighbour, neighbour_distance = nearest_source_task(held, fit_ids)
    nn_code = source[fit_ids.index(neighbour)]

    weights = channel_weights(
        int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device
    )
    scored = collect_blocks(
        model, data, cache, held, split=SPLIT_SELECTION, seed=seed + 70_000, draws=NOISE_DRAWS
    )
    train_blocks = collect_blocks(
        model,
        data,
        cache,
        held,
        split=SPLIT_TRAIN,
        seed=seed + 60_000,
        draws=ORACLE_DRAWS,
        rows_cap=ORACLE_ROWS_CAP,
    )
    oracle = weighted_ls_code(train_blocks, weights)
    del train_blocks
    oracle_absolute = weighted_ls_code(scored, weights)

    codes = {
        "phi": phi,
        "nn": nn_code,
        "mean": mean_code,
        "oracle": oracle,
        "oracle_absolute": oracle_absolute,
    }
    scores = {
        name: float(np.mean([weighted_mse(block, code, weights) for block in scored]))
        for name, code in codes.items()
    }
    del scored

    achievable = scores["mean"] - scores["oracle"]
    return {
        "held_task": held,
        "feature_set": feature_set,
        "seed": seed,
        "fit_task_count": len(fit_ids),
        "nearest_source_task": neighbour,
        "nearest_source_distance_m": neighbour_distance,
        "mse": scores,
        "beats_nn": bool(scores["phi"] < scores["nn"]),
        "beats_mean": bool(scores["phi"] < scores["mean"]),
        "beats_both": bool(scores["phi"] < scores["nn"] and scores["phi"] < scores["mean"]),
        "achievable_gain_m": float(achievable),
        "captured_fraction": (
            float((scores["mean"] - scores["phi"]) / achievable)
            if achievable > 1e-12
            else None
        ),
        "oracle_beats_mean": bool(scores["oracle"] < scores["mean"]),
        "phi_beats_oracle": bool(scores["phi"] < scores["oracle"]),
        "phi_beats_oracle_absolute": bool(scores["phi"] < scores["oracle_absolute"]),
        "code_norms": {
            "phi": float(phi.norm()),
            "nn": float(nn_code.norm()),
            "mean": float(mean_code.norm()),
            "oracle": float(oracle.norm()),
            "oracle_absolute": float(oracle_absolute.norm()),
        },
        "wall_clock_seconds": time.time() - started,
    }


def verdict(folds: list[dict[str, Any]]) -> dict[str, Any]:
    expected = set(source_task_ids())
    present = {int(fold["held_task"]) for fold in folds}
    missing = sorted(expected - present)
    wins_both = sum(1 for fold in folds if fold["beats_both"])
    wins_nn = sum(1 for fold in folds if fold["beats_nn"])
    wins_mean = sum(1 for fold in folds if fold["beats_mean"])

    if len(expected) != TASK_COUNT or missing:
        return {
            "outcome": "void",
            "void_reason": (
                f"census short: {len(present)} of {TASK_COUNT} folds, missing {missing}"
            ),
            "fold_count": len(folds),
            "missing_tasks": missing,
            "wins_beats_both": wins_both,
        }

    broken = sorted(f["held_task"] for f in folds if f["phi_beats_oracle_absolute"])
    if broken:
        return {
            "outcome": "void",
            "void_reason": (
                f"phi beat the absolute ceiling on tasks {broken}; the ceiling is fit on "
                f"the scored rows, so this is a harness fault, not a result"
            ),
            "fold_count": len(folds),
            "missing_tasks": [],
            "wins_beats_both": wins_both,
        }

    if wins_both >= PASS_WIN_FLOOR:
        outcome = "pass"
    elif wins_both <= FALSIFIER_WIN_CEILING:
        outcome = "falsifier_fired"
    else:
        outcome = "intermediate_no_claim"

    captured = [f["captured_fraction"] for f in folds if f["captured_fraction"] is not None]
    return {
        "outcome": outcome,
        "fold_count": len(folds),
        "missing_tasks": [],
        "wins_beats_both": wins_both,
        "wins_beats_nn": wins_nn,
        "wins_beats_mean": wins_mean,
        "pass_win_floor": PASS_WIN_FLOOR,
        "falsifier_win_ceiling": FALSIFIER_WIN_CEILING,
        "sign_test_p_one_sided": sign_test_p(len(folds), wins_both),
        "captured_fraction_median": (
            float(np.median(captured)) if captured else None
        ),
        "captured_fraction_defined_on": len(captured),
        "oracle_beats_mean_count": sum(1 for f in folds if f["oracle_beats_mean"]),
        "phi_beats_train_oracle_count": sum(1 for f in folds if f["phi_beats_oracle"]),
    }


def sign_test_p(trials: int, wins: int) -> float:
    from math import comb

    return sum(comb(trials, i) for i in range(wins, trials + 1)) / 2**trials


E33_1_RUN_TEMPLATE = (
    "runs/20260813_g33_e33_1_v5_structured_{feature_set}_seed{seed}_v1/gate_report.json"
)


def run_anchor(
    *, feature_set: str, seed: int, data: Any, cache: DeviceCache, device: str
) -> dict[str, Any]:
    path = ROOT / E33_1_RUN_TEMPLATE.format(feature_set=feature_set, seed=seed)
    if not path.exists():
        raise FileNotFoundError(f"E33.1 anchor report missing: {path}")
    published = float(json.loads(path.read_text())["selection_loss"])

    all_ids = list(source_task_ids())
    model, lookup = train_field(
        feature_set=feature_set,
        seed=seed,
        fit_task_ids=all_ids,
        data=data,
        cache=cache,
        device=device,
    )
    weights = channel_weights(
        int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device
    )
    reproduced = e2.validation_loss(
        model,
        data,
        cache,
        data.rows_for_split(SPLIT_SELECTION),
        lookup,
        weights,
        seed + 20_000,
        device,
    )
    gap = abs(reproduced - published)
    return {
        "feature_set": feature_set,
        "seed": seed,
        "published_selection_loss": published,
        "reproduced_selection_loss": reproduced,
        "absolute_gap": gap,
        "relative_gap": gap / max(abs(published), 1e-30),
        "tolerance": ANCHOR_SELECTION_TOLERANCE,
        "holds": bool(gap <= ANCHOR_SELECTION_TOLERANCE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", default="saturated_v5")
    parser.add_argument("--seed", type=int, default=148021)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--anchor-only",
        action="store_true",
        help="run only the all-41 code-path anchor and report the measured gap",
    )
    parser.add_argument(
        "--held-tasks",
        default="",
        help="comma-separated subset of source ids for a smoke run; the verdict voids on a "
        "short census, which is the intended behaviour",
    )
    args = parser.parse_args()

    data = load_v5_data()
    cache = DeviceCache(data, args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    anchor = run_anchor(
        feature_set=args.feature_set,
        seed=args.seed,
        data=data,
        cache=cache,
        device=args.device,
    )
    print(
        f"anchor: published {anchor['published_selection_loss']:.10f} "
        f"reproduced {anchor['reproduced_selection_loss']:.10f} "
        f"gap {anchor['absolute_gap']:.3e} holds={anchor['holds']}",
        flush=True,
    )
    if args.anchor_only:
        (args.output / "anchor_report.json").write_text(json.dumps(anchor, indent=2))
        return

    held_tasks = (
        [int(v) for v in args.held_tasks.split(",") if v.strip()]
        if args.held_tasks
        else list(source_task_ids())
    )
    folds: list[dict[str, Any]] = []
    for index, held in enumerate(held_tasks, start=1):
        fold = run_fold(
            held=held,
            feature_set=args.feature_set,
            seed=args.seed,
            data=data,
            cache=cache,
            device=args.device,
        )
        folds.append(fold)
        print(
            f"[{index:2d}/{len(held_tasks)}] task {held:2d}  "
            f"phi {fold['mse']['phi']:.6f}  nn {fold['mse']['nn']:.6f}  "
            f"mean {fold['mse']['mean']:.6f}  oracle {fold['mse']['oracle']:.6f}  "
            f"both={fold['beats_both']}  {fold['wall_clock_seconds']:.0f}s",
            flush=True,
        )
        (args.output / "folds.json").write_text(json.dumps(folds, indent=2))

    report = {
        "protocol": PROTOCOL,
        "audit_only": True,
        "can_claim_sal_result": False,
        "sal_claim_generated": False,
        "rollouts_executed": 0,
        "feature_set": args.feature_set,
        "seed": args.seed,
        "noise_draws": NOISE_DRAWS,
        "source_dataset_sha256": data.sha256,
        "anchor": anchor,
        "verdict": verdict(folds),
        "folds": folds,
        "wall_clock_seconds": time.time() - started,
    }
    (args.output / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
