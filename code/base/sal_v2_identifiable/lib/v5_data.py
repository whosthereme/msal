from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from .frozen_data import sha256_file
from .pool_v5 import build_pool, sealed_task_ids, source_task_ids

V5_NPZ = (
    PACKAGE
    / "runs"
    / "20260811_g33_smolvla_v5_spatial_substrate_v1"
    / "data"
    / "v5_spatial_state_action_fields.npz"
)

EXPECTED_V5_SHA = "236ef36aa40c2dcea381b16696d29b588102f88d8cdc68088b65aa802ca023cf"

TASK_SPLIT_SOURCE = 0
TASK_SPLIT_SEALED = 1

EXPECTED_SOURCE_TASKS = 41
EXPECTED_SEALED_TASKS = 14
EXPECTED_TOTAL_TASKS = 55

DIRECTION_COUNT_16 = 16
RADIUS_COUNT = 4


@dataclass
class V5Data:

    state_features: np.ndarray
    action_chunks: np.ndarray
    action_valid_masks: np.ndarray
    task_ids: np.ndarray
    direction_16: np.ndarray
    radius_indices: np.ndarray
    episode_indices: np.ndarray
    step_indices: np.ndarray
    split_codes: np.ndarray
    task_split_codes: np.ndarray
    sha256: str

    @property
    def state_dim(self) -> int:
        return int(self.state_features.shape[1])

    def rows_for_split(self, split: int) -> np.ndarray:
        return np.flatnonzero(
            (self.split_codes == split) & (self.task_split_codes == TASK_SPLIT_SOURCE)
        )

    def sealed_rows_for_split(self, split: int) -> np.ndarray:
        return np.flatnonzero(
            (self.split_codes == split) & (self.task_split_codes == TASK_SPLIT_SEALED)
        )

    def rows_for_task(self, task_id: int, split: int | None = None) -> np.ndarray:
        mask = self.task_ids == task_id
        if split is not None:
            mask &= self.split_codes == split
        return np.flatnonzero(mask)

    def unique_task_ids(self) -> list[int]:
        return [
            int(value)
            for value in np.unique(self.task_ids[self.task_split_codes == TASK_SPLIT_SOURCE])
        ]

    def sealed_task_ids(self) -> list[int]:
        return [
            int(value)
            for value in np.unique(self.task_ids[self.task_split_codes == TASK_SPLIT_SEALED])
        ]

    def direction_8_for_v4_rows(self) -> np.ndarray:
        halved = self.direction_16 // 2
        return np.where(self.direction_16 % 2 == 0, halved, -1)


def load_v5_data(verify_hash: bool = True) -> V5Data:
    if not V5_NPZ.exists():
        raise FileNotFoundError(
            f"v5 substrate missing: {V5_NPZ}. Build it with build_v5_spatial_substrate.py "
            "(PROTOCOL_G33 §11d); it takes ~32 min on one RTX 4090 D."
        )
    digest = sha256_file(V5_NPZ) if verify_hash else ""
    if verify_hash and digest != EXPECTED_V5_SHA:
        raise RuntimeError(
            f"v5 substrate digest {digest} != registered {EXPECTED_V5_SHA}. This substrate carries "
            "the 14-task sealed split; a changed digest means the holdout is not the one downstream "
            "results were measured against, so the run is not comparable. Refusing to load."
        )
    payload = np.load(V5_NPZ)
    task_ids = payload["task_ids"].astype(np.int64)
    task_split_codes = payload["task_split_codes"].astype(np.int64)
    direction_16 = payload["direction_indices"].astype(np.int64)
    radius_indices = payload["radius_indices"].astype(np.int64)

    sealed_truth = set(sealed_task_ids())
    source_truth = set(source_task_ids())
    sealed_present = {int(v) for v in np.unique(task_ids[task_split_codes == TASK_SPLIT_SEALED])}
    source_present = {int(v) for v in np.unique(task_ids[task_split_codes == TASK_SPLIT_SOURCE])}

    if sealed_present != sealed_truth:
        raise RuntimeError(
            "sealed half does not match pool_v5.sealed_task_ids(): "
            f"extra {sorted(sealed_present - sealed_truth)}, "
            f"missing {sorted(sealed_truth - sealed_present)}"
        )
    if source_present != source_truth:
        raise RuntimeError(
            "source half does not match pool_v5.source_task_ids(): "
            f"extra {sorted(source_present - source_truth)}, "
            f"missing {sorted(source_truth - source_present)}"
        )
    if source_present & sealed_truth:
        raise RuntimeError(
            f"source half contains sealed tasks: {sorted(source_present & sealed_truth)}"
        )
    if len(source_present) != EXPECTED_SOURCE_TASKS or len(sealed_present) != EXPECTED_SEALED_TASKS:
        raise RuntimeError(
            f"expected {EXPECTED_SOURCE_TASKS} source and {EXPECTED_SEALED_TASKS} sealed tasks, "
            f"got {len(source_present)} and {len(sealed_present)}"
        )

    pool = {t.task_id: t for t in build_pool()}
    if len(pool) != EXPECTED_TOTAL_TASKS:
        raise RuntimeError(f"pool_v5 built {len(pool)} tasks, expected {EXPECTED_TOTAL_TASKS}")
    expected_direction = np.array([pool[int(t)].direction_16 for t in task_ids], dtype=np.int64)
    expected_radius = np.array([pool[int(t)].radius_index for t in task_ids], dtype=np.int64)
    if not np.array_equal(direction_16, expected_direction):
        bad = int(np.count_nonzero(direction_16 != expected_direction))
        raise RuntimeError(
            f"direction_indices column disagrees with pool_v5.direction_16 on {bad} rows. If this "
            "column holds 0-7, the npz is a v4 substrate and belongs to lib/parallel_data.py."
        )
    if not np.array_equal(radius_indices, expected_radius):
        bad = int(np.count_nonzero(radius_indices != expected_radius))
        raise RuntimeError(
            f"radius_indices column disagrees with pool_v5.radius_index on {bad} rows"
        )
    if direction_16.min() < 0 or direction_16.max() >= DIRECTION_COUNT_16:
        raise RuntimeError(
            f"direction_16 out of range [0, {DIRECTION_COUNT_16}): "
            f"[{direction_16.min()}, {direction_16.max()}]"
        )

    return V5Data(
        state_features=payload["state_features"],
        action_chunks=payload["action_chunks"],
        action_valid_masks=payload["action_valid_masks"],
        task_ids=task_ids,
        direction_16=direction_16,
        radius_indices=radius_indices,
        episode_indices=payload["episode_indices"].astype(np.int64),
        step_indices=payload["step_indices"].astype(np.int64),
        split_codes=payload["split_codes"].astype(np.int64),
        task_split_codes=task_split_codes,
        sha256=digest,
    )
