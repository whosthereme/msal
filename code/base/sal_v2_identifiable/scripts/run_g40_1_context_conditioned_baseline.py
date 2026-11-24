from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lib.context_cond import ContextConditionedField, build_context_features
from lib.frozen_data import DeviceCache, SPLIT_SELECTION, SPLIT_TRAIN
from lib.pool_v5 import build_pool, source_task_ids
from lib.structured_field import channel_weights
from lib.v5_data import load_v5_data

import run_e2_structured_training as e2
from run_e35_1_zero_shot_code_loto import (
    NOISE_DRAWS,
    collect_blocks,
    weighted_mse,
)

PROTOCOL = "sal_v2_identifiable_g40_e40_1_context_conditioned_baseline_v1"

E35_1_GLOB = "runs/*_g35_e35_1_saturated_v5_seed{seed}_v1/gate_report.json"


def resolve_reference(seed: int) -> Path:
    matches = sorted(ROOT.glob(E35_1_GLOB.format(seed=seed)))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one E35.1 reference for seed {seed}, found {len(matches)}: "
            f"{[str(m.relative_to(ROOT)) for m in matches]}"
        )
    return matches[0]

PASS_WIN_FLOOR = 27
FALSIFIER_WIN_CEILING = 14

PHI_MATCH_RATIO = 1.01

ARMS = ("context_raw", "context_phi", "context_phi_wide")


def train_context_arm(
    *,
    arm: str,
    seed: int,
    fit_ids: list[int],
    data,
    cache: DeviceCache,
    device: str,
) -> tuple[ContextConditionedField, dict[int, int]]:
    lookup = {task_id: index for index, task_id in enumerate(fit_ids)}
    torch.manual_seed(seed)
    config = e2.JointSALConfig(
        state_dim=data.state_dim,
        chunk_length=int(data.action_chunks.shape[1]),
        action_dim=int(data.action_chunks.shape[2]),
    )
    features = build_context_features(
        fit_ids,
        arm=arm,
        feature_set="saturated_v5",
        pool="v5",
        radius_count=4,
    )
    model = ContextConditionedField(
        config, len(fit_ids), arm=arm, task_features=features
    ).to(device)
    weights = channel_weights(config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)

    rows_by_task = {t: data.rows_for_task(t, SPLIT_TRAIN) for t in fit_ids}
    for task_id, rows in rows_by_task.items():
        if len(rows) == 0:
            raise RuntimeError(f"v5 source task {task_id} has no training rows")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=e2.LEARNING_RATE,
        weight_decay=e2.WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    per_task = max(e2.BATCH_SIZE // len(fit_ids), 1)

    for step in range(1, e2.TRAIN_STEPS + 1):
        picked = np.concatenate(
            [rng.choice(rows_by_task[t], size=per_task, replace=True) for t in fit_ids]
        )
        batch = cache.batch(picked, seed=seed * 7919 + step)
        task_rows = torch.tensor([lookup[int(v)] for v in data.task_ids[picked]], device=device)
        prediction = model.forward_with_rows(
            batch["state"], batch["x_t"], batch["time"], task_rows
        )
        loss = e2.masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, lookup


def score_held(
    *,
    model: ContextConditionedField,
    held: int,
    seed: int,
    data,
    cache: DeviceCache,
    device: str,
) -> float:
    weights = channel_weights(
        int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device
    )
    rows = data.rows_for_task(held, SPLIT_SELECTION)
    if len(rows) == 0:
        raise RuntimeError(f"held task {held} has no selection rows")

    held_features = build_context_features(
        [held],
        arm=model.arm,
        feature_set="saturated_v5",
        pool="v5",
        radius_count=4,
    )
    with torch.no_grad():
        model.task_features = torch.cat(
            [model.task_features, torch.tensor(held_features, dtype=torch.float32, device=device)],
            dim=0,
        )
        held_row = torch.tensor(
            [model.task_features.shape[0] - 1], device=device, dtype=torch.long
        )
        total, count = 0.0, 0
        with torch.no_grad():
            for draw in range(NOISE_DRAWS):
                batch = cache.batch(rows, seed=seed + 70_000 + 1_000_003 * draw)
                prediction = model.forward_with_rows(
                    batch["state"],
                    batch["x_t"],
                    batch["time"],
                    held_row.expand(batch["state"].shape[0]),
                )
                error = (prediction - batch["target"]) ** 2 * weights[None, None, :]
                per_row = error.sum(dim=-1) / weights.sum().clamp_min(1e-12)
                mask = batch["masks"]
                total += float((per_row * mask).sum() / mask.sum().clamp_min(1.0))
                count += 1
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--held-tasks",
        default="",
        help="comma-separated subset for a smoke run; the verdict voids on a short census",
    )
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()

    reference_path = resolve_reference(args.seed)
    reference = json.loads(reference_path.read_text())
    reference_mse = {int(f["held_task"]): float(f["mse"]["phi"]) for f in reference["folds"]}

    data = load_v5_data()
    cache = DeviceCache(data, args.device)
    all_ids = list(source_task_ids())
    if args.held_tasks:
        held_list = [int(t) for t in args.held_tasks.split(",") if t.strip()]
    else:
        held_list = list(all_ids)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")

    started = time.time()
    folds: list[dict] = []
    for held in held_list:
        fit_ids = [t for t in all_ids if t != held]
        if len(fit_ids) != len(all_ids) - 1:
            raise RuntimeError(f"held task {held} is not a v5 source task")
        for arm in arms:
            model, _ = train_context_arm(
                arm=arm, seed=args.seed, fit_ids=fit_ids, data=data, cache=cache, device=args.device
            )
            model.eval()
            mse = score_held(
                model=model, held=held, seed=args.seed, data=data, cache=cache, device=args.device
            )
            split_stats = model.parameter_split()
            del model
            torch.cuda.empty_cache()
            phi_mse = reference_mse[held]
            folds.append(
                {
                    "held_task": int(held),
                    "seed": int(args.seed),
                    "arm": arm,
                    "mse": float(mse),
                    "phi_mse": phi_mse,
                    "relative_to_phi": float(mse / phi_mse) if phi_mse > 0 else float("nan"),
                    "matches_phi": bool(mse <= PHI_MATCH_RATIO * phi_mse),
                    "beats_phi": bool(mse < phi_mse),
                    **split_stats,
                }
            )
            print(
                f"[{arm}] held={held} mse={mse:.6f} phi={phi_mse:.6f} "
                f"rel={mse / phi_mse:.3f}",
                flush=True,
            )

    verdicts = {}
    for arm in arms:
        members = [f for f in folds if f["arm"] == arm]
        matches = sum(1 for f in members if f["matches_phi"])
        wins = sum(1 for f in members if f["beats_phi"])
        n = len(members)
        if n != len(held_list):
            outcome = "v_void_incomplete_census"
        elif matches >= PASS_WIN_FLOOR:
            outcome = "context_matches_phi"
        elif matches <= FALSIFIER_WIN_CEILING:
            outcome = "factorization_adds_measured_value"
        else:
            outcome = "v_inconclusive"
        verdicts[arm] = {
            "fold_count": n,
            "expected_folds": len(held_list),
            "matches_phi": matches,
            "beats_phi": wins,
            "match_ratio_bar": PHI_MATCH_RATIO,
            "pass_win_floor": PASS_WIN_FLOOR,
            "falsifier_win_ceiling": FALSIFIER_WIN_CEILING,
            "outcome": outcome,
        }

    payload = {
        "protocol": PROTOCOL,
        "audit_only": False,
        "can_claim_sal_result": True,
        "sal_claim_generated": True,
        "rollouts_executed": 0,
        "seed": int(args.seed),
        "reference_run": str(reference_path.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "collapsed_check": True,
        "folds": folds,
        "verdict": verdicts,
        "wall_clock_seconds": float(time.time() - started),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gate_report.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
