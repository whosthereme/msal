from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from .frozen_data import sha256_file
from .task_algebra import split_residue

PARALLEL_NPZ = (
    PACKAGE
    / "runs"
    / "20260728_g15_parallel_spatial_substrate_v1"
    / "data"
    / "parallel_spatial_state_action_fields.npz"
)

EXPECTED_PARALLEL_SHA = "73699bb39a679e410185f48ebb30b9102baa3162249fcb59791a2feb8f4e0ca5"

TASK_SPLIT_SOURCE = 0
TASK_SPLIT_SEALED = 1


@dataclass
class ParallelData:

    state_features: np.ndarray
    action_chunks: np.ndarray
    action_valid_masks: np.ndarray
    task_ids: np.ndarray
    direction_indices: np.ndarray
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


def load_parallel_data(verify_hash: bool = True) -> ParallelData:
    if not PARALLEL_NPZ.exists():
        raise FileNotFoundError(f"parallel substrate missing: {PARALLEL_NPZ}")
    digest = sha256_file(PARALLEL_NPZ) if verify_hash else ""
    if verify_hash and digest != EXPECTED_PARALLEL_SHA:
        raise RuntimeError(
            f"parallel substrate digest {digest} != registered {EXPECTED_PARALLEL_SHA}. This "
            f"substrate carries the sealed split; a changed digest means the holdout E15.4 spent "
            f"is not the one it was spent on, so the run is not comparable. Refusing to load."
        )
    payload = np.load(PARALLEL_NPZ)
    task_ids = payload["task_ids"].astype(np.int64)
    task_split_codes = payload["task_split_codes"].astype(np.int64)

    sealed_ids = np.unique(task_ids[task_split_codes == TASK_SPLIT_SEALED])
    source_ids = np.unique(task_ids[task_split_codes == TASK_SPLIT_SOURCE])
    if not (split_residue(sealed_ids) == 3).all():
        raise RuntimeError(f"sealed half contains non-sealed tasks: {sealed_ids.tolist()}")
    if (split_residue(source_ids) == 3).any():
        raise RuntimeError(f"source half contains sealed tasks: {source_ids.tolist()}")
    if len(sealed_ids) != 8 or len(source_ids) != 24:
        raise RuntimeError(
            f"expected 24 source and 8 sealed tasks, got {len(source_ids)} and {len(sealed_ids)}"
        )
    return ParallelData(
        state_features=payload["state_features"],
        action_chunks=payload["action_chunks"],
        action_valid_masks=payload["action_valid_masks"],
        task_ids=task_ids,
        direction_indices=payload["direction_indices"].astype(np.int64),
        radius_indices=payload["radius_indices"].astype(np.int64),
        episode_indices=payload["episode_indices"].astype(np.int64),
        step_indices=payload["step_indices"].astype(np.int64),
        split_codes=payload["split_codes"].astype(np.int64),
        task_split_codes=task_split_codes,
        sha256=digest,
    )
