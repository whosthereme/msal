from __future__ import annotations

from typing import Sequence

import numpy as np

from .pool_v5 import DIRECTION_COUNT_16, RADII_M, build_pool

RADIUS_COUNT = len(RADII_M)

MAX_HARMONIC = DIRECTION_COUNT_16 // 2


def _angular_names() -> tuple[str, ...]:
    names = ["const"]
    for harmonic in range(1, MAX_HARMONIC + 1):
        names.append(f"cos{harmonic}")
        if harmonic < MAX_HARMONIC:
            names.append(f"sin{harmonic}")
    return tuple(names)


_RADIAL_NAMES = ("r", "r2", "r3", "r_cos1", "r_sin1")

FEATURE_SETS_V5: dict[str, tuple[str, ...]] = {
    "physical_xy_v5": ("const", "cos1", "sin1", "r_cos1", "r_sin1"),
    "physical_xy_radial_v5": ("const", "cos1", "sin1", "r", "r_cos1", "r_sin1"),
    "harmonic2_radial2_v5": (
        "const", "cos1", "sin1", "cos2", "sin2", "r", "r2", "r_cos1", "r_sin1",
    ),
    "saturated_v5": _angular_names() + _RADIAL_NAMES,
}

EXPECTED_RANKS: dict[str, int] = {
    "physical_xy_v5": 5,
    "physical_xy_radial_v5": 6,
    "harmonic2_radial2_v5": 9,
    "saturated_v5": 21,
}


def _radius_scale(radius: np.ndarray) -> np.ndarray:
    span = max(RADIUS_COUNT - 1, 1)
    return 2.0 * radius.astype(np.float64) / span - 1.0


def feature_columns_v5(
    directions_16: Sequence[int] | np.ndarray,
    radii: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    direction = np.asarray(directions_16, dtype=np.int64)
    radius = np.asarray(radii, dtype=np.int64)
    if direction.shape != radius.shape:
        raise ValueError("directions_16 and radii must have the same shape")
    if direction.size == 0:
        raise ValueError("no tasks given")
    if direction.min() < 0 or direction.max() >= DIRECTION_COUNT_16:
        raise ValueError(
            f"direction_16 out of range [0, {DIRECTION_COUNT_16}): "
            f"[{direction.min()}, {direction.max()}]"
        )
    if radius.min() < 0 or radius.max() >= RADIUS_COUNT:
        raise ValueError(f"radius out of range [0, {RADIUS_COUNT})")

    theta = 2.0 * np.pi * direction.astype(np.float64) / DIRECTION_COUNT_16
    scaled = _radius_scale(radius)
    columns: dict[str, np.ndarray] = {"const": np.ones_like(theta)}
    for harmonic in range(1, MAX_HARMONIC + 1):
        columns[f"cos{harmonic}"] = np.cos(harmonic * theta)
        if harmonic < MAX_HARMONIC:
            columns[f"sin{harmonic}"] = np.sin(harmonic * theta)
    for power in range(1, RADIUS_COUNT):
        columns["r" if power == 1 else f"r{power}"] = scaled**power
    columns["r_cos1"] = scaled * np.cos(theta)
    columns["r_sin1"] = scaled * np.sin(theta)
    return columns


def design_matrix_v5(
    directions_16: Sequence[int] | np.ndarray,
    radii: Sequence[int] | np.ndarray,
    feature_set: str = "physical_xy_radial_v5",
) -> tuple[np.ndarray, tuple[str, ...]]:
    if feature_set not in FEATURE_SETS_V5:
        raise ValueError(
            f"unknown v5 feature set {feature_set!r}; choose from {sorted(FEATURE_SETS_V5)}"
        )
    names = FEATURE_SETS_V5[feature_set]
    columns = feature_columns_v5(directions_16, radii)
    return np.stack([columns[name] for name in names], axis=1), names


def design_matrix_for_tasks(
    task_ids: Sequence[int] | np.ndarray,
    feature_set: str = "physical_xy_radial_v5",
) -> tuple[np.ndarray, tuple[str, ...]]:
    pool = {task.task_id: task for task in build_pool()}
    ids = [int(t) for t in np.asarray(task_ids, dtype=np.int64).ravel()]
    unknown = sorted({t for t in ids if t not in pool})
    if unknown:
        raise ValueError(f"task ids {unknown} are not in the v5 pool")
    directions = [pool[t].direction_16 for t in ids]
    radii = [pool[t].radius_index for t in ids]
    return design_matrix_v5(directions, radii, feature_set)


def explained_variance_v5(
    codes: np.ndarray,
    directions_16: Sequence[int] | np.ndarray,
    radii: Sequence[int] | np.ndarray,
    feature_set: str = "physical_xy_radial_v5",
) -> dict[str, float]:
    matrix, names = design_matrix_v5(directions_16, radii, feature_set)
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
