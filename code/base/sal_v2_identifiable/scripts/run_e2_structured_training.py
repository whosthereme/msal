from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from joint_sal_velocity import (
    JointSALConfig,
    JointSALVelocityField,
    output_orthogonality_error,
)

from lib.frozen_data import (
    SPLIT_ROBUSTNESS,
    SPLIT_SELECTION,
    SPLIT_TRAIN,
    DeviceCache,
    load_source_data,
)
from lib.output_gauge import atom_gram, capacity_report, code_covariance
from lib.structured_field import (
    StructuredCodeField,
    channel_weights,
    group_sparsity_penalty,
    verify_dead_channels,
)

TRAIN_STEPS = int(__import__("os").environ.get("G12_TRAIN_STEPS", 12_800))
BATCH_SIZE = 512
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
CHECK_EVERY = 800
ORTHOGONALITY_WEIGHT = 0.02

GROUP_SPARSITY_WEIGHT = 1e-3
DEAD_ACTION_DIMS = (3, 4, 5)


def masked_channel_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target) ** 2 * weights[None, None, :]
    per_row = error.sum(dim=-1) / weights.sum().clamp_min(1e-12)
    return (per_row * mask).sum() / mask.sum().clamp_min(1.0)


def validation_loss(
    model: Any,
    data: Any,
    cache: DeviceCache,
    rows: np.ndarray,
    task_row_lookup: dict[int, int],
    weights: torch.Tensor,
    seed: int,
    device: str,
    chunk: int = 2048,
) -> float:
    total, count = 0.0, 0
    for start in range(0, len(rows), chunk):
        subset = rows[start : start + chunk]
        batch = cache.batch(subset, seed=seed + start)
        task_rows = torch.tensor(
            [task_row_lookup[int(value)] for value in data.task_ids[subset]], device=device
        )
        with torch.no_grad():
            codes = (
                model.codes_for_rows(task_rows)
                if isinstance(model, StructuredCodeField)
                else model.source_codes(task_rows)
            )
            prediction = model.forward_with_code(
                batch["state"], batch["x_t"], batch["time"], codes
            )
            loss = masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
        total += float(loss) * len(subset)
        count += len(subset)
    return total / max(count, 1)


def train_one(
    *,
    parameterization: str,
    feature_set: str | None,
    seed: int,
    data: Any,
    cache: DeviceCache,
    device: str,
) -> dict[str, Any]:
    task_ids = data.unique_task_ids()
    task_row_lookup = {task_id: index for index, task_id in enumerate(task_ids)}
    torch.manual_seed(seed)
    config = JointSALConfig(
        state_dim=data.state_dim,
        chunk_length=int(data.action_chunks.shape[1]),
        action_dim=int(data.action_chunks.shape[2]),
    )
    if parameterization == "structured":
        model = StructuredCodeField(config, task_ids, feature_set=feature_set)
    elif parameterization == "free_embedding":
        model = JointSALVelocityField(config, source_task_count=len(task_ids))
    else:
        raise ValueError(f"unknown parameterization {parameterization!r}")
    model = model.to(device)

    weights = channel_weights(config.action_dim, dead_dims=DEAD_ACTION_DIMS, device=device)
    train_rows = data.rows_for_split(SPLIT_TRAIN)
    selection_rows = data.rows_for_split(SPLIT_SELECTION)
    robustness_rows = data.rows_for_split(SPLIT_ROBUSTNESS)

    rows_by_task = {task_id: data.rows_for_task(task_id, SPLIT_TRAIN) for task_id in task_ids}
    for task_id, rows in rows_by_task.items():
        if len(rows) == 0:
            raise RuntimeError(f"source task {task_id} has no training rows")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    per_task = max(BATCH_SIZE // len(task_ids), 1)
    history: list[dict[str, float]] = []
    started = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        picked = np.concatenate(
            [rng.choice(rows_by_task[task_id], size=per_task, replace=True) for task_id in task_ids]
        )
        batch = cache.batch(picked, seed=seed * 7919 + step)
        task_rows = torch.tensor(
            [task_row_lookup[int(value)] for value in data.task_ids[picked]], device=device
        )
        codes = (
            model.codes_for_rows(task_rows)
            if isinstance(model, StructuredCodeField)
            else model.source_codes(task_rows)
        )
        base, atoms = model.components(batch["state"], batch["x_t"], batch["time"], None)
        prediction = base + torch.einsum("bk,bktd->btd", codes, atoms)
        fit = masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
        orthogonality = output_orthogonality_error(atoms)
        all_codes = (
            model.structured_codes if isinstance(model, StructuredCodeField)
            else model.source_codes.weight
        )
        sparsity = group_sparsity_penalty(all_codes, atoms)
        loss = fit + ORTHOGONALITY_WEIGHT * orthogonality + GROUP_SPARSITY_WEIGHT * sparsity

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % CHECK_EVERY == 0 or step == 1:
            history.append(
                {
                    "step": step,
                    "fit_loss": float(fit.detach()),
                    "orthogonality": float(orthogonality.detach()),
                    "group_sparsity": float(sparsity.detach()),
                    "total_loss": float(loss.detach()),
                }
            )

    selection_loss = validation_loss(
        model, data, cache, selection_rows, task_row_lookup, weights, seed + 20_000, device
    )
    robustness_loss = validation_loss(
        model, data, cache, robustness_rows, task_row_lookup, weights, seed + 30_000, device
    )

    probe_rows = np.random.default_rng(seed + 40_000).permutation(robustness_rows)[:3072]
    probe = cache.batch(probe_rows, seed=seed + 50_000)
    with torch.no_grad():
        _, probe_atoms = model.components(probe["state"], probe["x_t"], probe["time"], None)
    final_codes = (
        model.structured_codes.detach() if isinstance(model, StructuredCodeField)
        else model.source_codes.weight.detach()
    )
    capacity = capacity_report(atom_gram(probe_atoms), code_covariance(final_codes))

    return {
        "parameterization": parameterization,
        "feature_set": feature_set,
        "seed": seed,
        "train_rows": int(len(train_rows)),
        "history": history,
        "selection_loss": selection_loss,
        "robustness_loss": robustness_loss,
        "capacity": capacity,
        "codes": final_codes.cpu().double().tolist(),
        "wall_clock_seconds": float(time.time() - started),
        "model_state_dict": {
            key: value.cpu() for key, value in model.state_dict().items()
        },
        "config": config.__dict__,
        "source_task_ids": task_ids,
        "task_id_to_row": task_row_lookup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameterization", choices=("structured", "free_embedding"), required=True)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.parameterization == "structured" and not args.feature_set:
        raise ValueError("--feature-set is required for structured parameterization")

    data = load_source_data()
    dead = verify_dead_channels(data.action_chunks, DEAD_ACTION_DIMS)
    for name, stats in dead.items():
        if stats["max_abs"] != 0.0 or stats["nonzero_count"] != 0:
            raise RuntimeError(
                f"declared dead channel {name} is not identically zero: {stats}. "
                "The registered channel-weighting precondition fails; aborting."
            )

    cache = DeviceCache(data, device=args.device)
    result = train_one(
        parameterization=args.parameterization,
        feature_set=args.feature_set,
        seed=args.seed,
        data=data,
        cache=cache,
        device=args.device,
    )

    output = args.output.resolve()
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    state = result.pop("model_state_dict")
    torch.save(
        {
            "model_state_dict": state,
            "config": result["config"],
            "source_task_ids": result["source_task_ids"],
            "task_id_to_row": result["task_id_to_row"],
            "parameterization": result["parameterization"],
            "feature_set": result["feature_set"],
            "seed": result["seed"],
            "source_dataset_sha256": data.sha256,
        },
        output / "data" / "structured_sal.pt",
    )
    report = {
        "protocol": "sal_v2_identifiable_g12_e2_structured_training_v1",
        "audit_only": True,
        "can_claim_sal_result": False,
        "sal_claim_generated": False,
        "dead_channel_precondition": dead,
        "group_sparsity_weight": GROUP_SPARSITY_WEIGHT,
        "orthogonality_weight": ORTHOGONALITY_WEIGHT,
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "source_dataset_sha256": data.sha256,
        **result,
    }
    (output / "gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="ascii"
    )
    print(
        json.dumps(
            {
                "run": str(output),
                "parameterization": result["parameterization"],
                "feature_set": result["feature_set"],
                "seed": result["seed"],
                "selection_loss": result["selection_loss"],
                "robustness_loss": result["robustness_loss"],
                "K_eff_099": result["capacity"]["effective_capacity"]["0.99"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
