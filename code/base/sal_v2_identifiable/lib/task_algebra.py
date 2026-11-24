from __future__ import annotations

from typing import Sequence

import numpy as np

DIRECTION_COUNT = 8
RADIUS_COUNT = 4

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "physical_xy": ("const", "cos1", "sin1", "r_cos1", "r_sin1"),
    "physical_xy_radial": ("const", "cos1", "sin1", "r", "r_cos1", "r_sin1"),
    "harmonic2_radial2": (
        "const", "cos1", "sin1", "cos2", "sin2", "r", "r2", "r_cos1", "r_sin1",
    ),
    "saturated": (
        "const", "cos1", "sin1", "cos2", "sin2", "cos3", "sin3", "cos4",
        "r", "r2", "r3", "r_cos1", "r_sin1",
    ),
}


def _radius_scale(radius: np.ndarray) -> np.ndarray:
    span = max(RADIUS_COUNT - 1, 1)
    return 2.0 * radius.astype(np.float64) / span - 1.0


def feature_columns(
    directions: Sequence[int] | np.ndarray,
    radii: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    direction = np.asarray(directions, dtype=np.int64)
    radius = np.asarray(radii, dtype=np.int64)
    if direction.shape != radius.shape:
        raise ValueError("directions and radii must have the same shape")
    if direction.min() < 0 or direction.max() >= DIRECTION_COUNT:
        raise ValueError(f"direction out of range [0, {DIRECTION_COUNT})")
    if radius.min() < 0 or radius.max() >= RADIUS_COUNT:
        raise ValueError(f"radius out of range [0, {RADIUS_COUNT})")
    theta = 2.0 * np.pi * direction.astype(np.float64) / DIRECTION_COUNT
    scaled = _radius_scale(radius)
    columns: dict[str, np.ndarray] = {"const": np.ones_like(theta)}
    for harmonic in range(1, DIRECTION_COUNT // 2 + 1):
        columns[f"cos{harmonic}"] = np.cos(harmonic * theta)
        if harmonic < DIRECTION_COUNT // 2:
            columns[f"sin{harmonic}"] = np.sin(harmonic * theta)
    for power in range(1, RADIUS_COUNT):
        columns["r" if power == 1 else f"r{power}"] = scaled**power
    columns["r_cos1"] = scaled * np.cos(theta)
    columns["r_sin1"] = scaled * np.sin(theta)
    return columns


def design_matrix(
    directions: Sequence[int] | np.ndarray,
    radii: Sequence[int] | np.ndarray,
    feature_set: str = "physical_xy_radial",
) -> tuple[np.ndarray, tuple[str, ...]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    names = FEATURE_SETS[feature_set]
    columns = feature_columns(directions, radii)
    matrix = np.stack([columns[name] for name in names], axis=1)
    return matrix, names


def task_indices(task_ids: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(task_ids, dtype=np.int64)
    if ids.max() >= 32:
        raise ValueError(
            "task_indices is v4-only (hardcoded to task_id = direction_8 * 4 + radius). "
            f"v5 task ids 32–54 decode incorrectly; max id in input: {ids.max()}. "
            "Use pool_v5.PoolTask.direction_16 and .radius_index instead."
        )
    return ids // RADIUS_COUNT, ids % RADIUS_COUNT


def split_residue(task_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    ids = np.asarray(task_ids, dtype=np.int64)
    if ids.max() >= 32:
        raise ValueError(
            "split_residue is v4-only. At n=55 it names 5 source tasks sealed and misses 5 actual "
            f"sealed tasks; max id in input: {ids.max()}. Use pool_v5.sealed_task_ids() instead."
        )
    direction, radius = task_indices(task_ids)
    return (direction + radius) % 4


def explained_variance(
    codes: np.ndarray,
    directions: Sequence[int] | np.ndarray,
    radii: Sequence[int] | np.ndarray,
    feature_set: str = "physical_xy_radial",
) -> dict[str, float]:
    matrix, names = design_matrix(directions, radii, feature_set)
    centered = codes.astype(np.float64) - codes.astype(np.float64).mean(axis=0, keepdims=True)
    coefficients, *_ = np.linalg.lstsq(matrix, centered, rcond=None)
    residual = centered - matrix @ coefficients
    total = float((centered**2).sum())
    return {
        "feature_set": feature_set,
        "num_features": len(names),
        "design_rank": int(np.linalg.matrix_rank(matrix)),
        "r_squared": float(1.0 - (residual**2).sum() / max(total, 1e-30)),
        "residual_fraction": float((residual**2).sum() / max(total, 1e-30)),
    }
