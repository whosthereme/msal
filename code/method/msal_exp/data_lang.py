from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
LANG_RUN = REPO / "runs" / "20260813_g33_v5_language_substrate_v1"
LANG_NPZ = LANG_RUN / "data" / "v5_language_conditioned_state_action_fields.npz"
EXPECTED_SHA = "99efd134456987b75fc94df2088cb90f60b0057d337cb46b93d642cebe539ff0"

FEATURE_KEYS = {"blind": "frozen_extractor_features", "lang": "state_features"}

SPLIT_TRAIN, SPLIT_SELECTION, SPLIT_ROBUSTNESS = 0, 1, 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class LangV5Data:

    state_features: np.ndarray | None
    action_chunks: np.ndarray
    action_valid_masks: np.ndarray
    task_ids: np.ndarray
    direction_16: np.ndarray
    radius_indices: np.ndarray
    split_codes: np.ndarray
    feature: str
    feature_dim: int
    sha256: str

    @property
    def state_dim(self) -> int:
        return int(self.feature_dim)

    def rows_for_split(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split_codes == split)

    def rows_for_task(self, task_id: int, split: int | None = None) -> np.ndarray:
        mask = self.task_ids == task_id
        if split is not None:
            mask &= self.split_codes == split
        return np.flatnonzero(mask)


def load_lang_v5(feature: str, verify_hash: bool = True) -> LangV5Data:
    from lib.pool_v5 import build_pool, source_task_ids

    if feature not in FEATURE_KEYS:
        raise ValueError(f"feature must be one of {sorted(FEATURE_KEYS)}, got {feature!r}")
    if not LANG_NPZ.exists():
        raise FileNotFoundError(LANG_NPZ)
    digest = sha256_file(LANG_NPZ) if verify_hash else ""
    if verify_hash and digest != EXPECTED_SHA:
        raise RuntimeError(f"language substrate digest {digest} != registered {EXPECTED_SHA}")

    payload = np.load(LANG_NPZ)
    available = payload["feature_available"].astype(bool)
    if not available.all():
        raise RuntimeError(f"{int((~available).sum())} rows unavailable; this runner assumes none")
    task_split = payload["task_split_codes"].astype(np.int64)
    if (task_split != 0).any():
        raise RuntimeError("language substrate contains non-source rows; refusing")
    task_ids = payload["task_ids"].astype(np.int64)
    present = {int(t) for t in np.unique(task_ids)}
    if present != set(source_task_ids()):
        raise RuntimeError(f"task census mismatch: {sorted(present ^ set(source_task_ids()))}")

    direction = payload["direction_indices"].astype(np.int64)
    radius = payload["radius_indices"].astype(np.int64)
    pool = {t.task_id: t for t in build_pool()}
    if not np.array_equal(direction, np.array([pool[int(t)].direction_16 for t in task_ids])):
        raise RuntimeError("direction column disagrees with pool_v5")
    if not np.array_equal(radius, np.array([pool[int(t)].radius_index for t in task_ids])):
        raise RuntimeError("radius column disagrees with pool_v5")

    block = payload[FEATURE_KEYS[feature]]
    return LangV5Data(
        state_features=block,
        action_chunks=payload["action_chunks"],
        action_valid_masks=payload["action_valid_masks"],
        task_ids=task_ids,
        direction_16=direction,
        radius_indices=radius,
        split_codes=payload["split_codes"].astype(np.int64),
        feature=feature,
        feature_dim=int(block.shape[1]),
        sha256=digest,
    )
