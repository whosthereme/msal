from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib import pool_libero40 as pool
from lib.frozen_data import sha256_file

SUBSTRATE_DIR = (
    Path(__file__).resolve().parents[1] / "runs" / "20260829_g39_libero40_substrate_v1"
)
SHARD_DIR = SUBSTRATE_DIR / "shards"

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2

DEAD_ACTION_DIMS: tuple[int, ...] = ()

FROZEN_FEATURE_DIM = 31_680


LANGUAGE_DIR = SUBSTRATE_DIR / "language"


@dataclass
class G39Data:

    state_features: np.ndarray
    action_chunks: np.ndarray
    action_valid_masks: np.ndarray
    task_ids: np.ndarray
    episode_indices: np.ndarray
    step_indices: np.ndarray
    split_codes: np.ndarray
    sha256: str

    @property
    def state_dim(self) -> int:
        return int(self.state_features.shape[1])

    def rows_for_split(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split_codes == split)

    def rows_for_task(self, task_id: int, split: int | None = None) -> np.ndarray:
        mask = self.task_ids == task_id
        if split is not None:
            mask &= self.split_codes == split
        return np.flatnonzero(mask)

    def unique_task_ids(self) -> list[int]:
        return [int(value) for value in np.unique(self.task_ids)]


def _census(shard_dir: Path) -> list[Path]:
    expected = [pool.FIRST_TASK_ID + i for i in range(pool.POOL_SIZE)]
    paths, missing = [], []
    for task_id in expected:
        path = shard_dir / f"task{task_id}.npz"
        (paths if path.exists() else missing).append(path if path.exists() else task_id)
    if missing:
        raise RuntimeError(
            f"substrate incomplete: {len(paths)} of {pool.POOL_SIZE} shards present, missing task ids "
            f"{missing}. An incomplete substrate must not be consumed -- rebuild rather than proceeding "
            f"on a short pool."
        )
    return paths


def load_g39_data(shard_dir: Path = SHARD_DIR, verify_hash: bool = True) -> G39Data:
    paths = _census(shard_dir)

    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "state_features",
            "action_chunks",
            "action_valid_masks",
            "task_ids",
            "episode_indices",
            "step_indices",
            "split_codes",
        )
    }
    digests = []
    for path in paths:
        payload = np.load(path)
        for key in parts:
            parts[key].append(payload[key])
        if verify_hash:
            digests.append(sha256_file(path))

    merged = {key: np.concatenate(values, axis=0) for key, values in parts.items()}

    features = merged["state_features"]
    if features.shape[1] != FROZEN_FEATURE_DIM:
        raise RuntimeError(
            f"frozen feature width is {features.shape[1]}, expected {FROZEN_FEATURE_DIM}"
        )
    present = {int(v) for v in np.unique(merged["task_ids"])}
    expected = {pool.FIRST_TASK_ID + i for i in range(pool.POOL_SIZE)}
    if present != expected:
        raise RuntimeError(
            f"task id set mismatch: {len(present)} present, {len(expected)} expected; "
            f"missing {sorted(expected - present)}, unexpected {sorted(present - expected)}"
        )
    pool.assert_pool_ids(sorted(present), sealed=())

    for task_id in sorted(present):
        for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            count = int(((merged["task_ids"] == task_id) & (merged["split_codes"] == split)).sum())
            if count == 0:
                raise RuntimeError(f"task {task_id} has no rows at split {split}")

    return G39Data(
        state_features=features,
        action_chunks=merged["action_chunks"],
        action_valid_masks=merged["action_valid_masks"],
        task_ids=merged["task_ids"].astype(np.int64),
        episode_indices=merged["episode_indices"].astype(np.int64),
        step_indices=merged["step_indices"].astype(np.int64),
        split_codes=merged["split_codes"].astype(np.int64),
        sha256="+".join(digests) if digests else "",
    )


@dataclass
class G39Language:

    state_features: np.ndarray
    available: np.ndarray
    sha256: str
    prompt_swap_relative_error: float

    @property
    def state_dim(self) -> int:
        return int(self.state_features.shape[1])

    def available_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        return rows[self.available[rows]]


def load_g39_language(data: G39Data, verify_hash: bool = True) -> G39Language:
    import json

    paths = []
    missing = []
    for i in range(pool.POOL_SIZE):
        task_id = pool.FIRST_TASK_ID + i
        path = LANGUAGE_DIR / f"task{task_id}.npz"
        (paths if path.exists() else missing).append(path if path.exists() else task_id)
    if missing:
        raise RuntimeError(
            f"language surface incomplete: {len(paths)} of {pool.POOL_SIZE} shards, missing {missing}"
        )

    blocks, task_ids, episodes, steps, digests = [], [], [], [], []
    for path in paths:
        payload = np.load(path)
        blocks.append(payload["language_features"])
        task_ids.append(payload["task_ids"])
        episodes.append(payload["episode_indices"])
        steps.append(payload["step_indices"])
        if verify_hash:
            digests.append(sha256_file(path))

    features = np.concatenate(blocks, axis=0)
    for name, got, want in (
        ("task_ids", np.concatenate(task_ids), data.task_ids),
        ("episode_indices", np.concatenate(episodes), data.episode_indices),
        ("step_indices", np.concatenate(steps), data.step_indices),
    ):
        if got.shape != want.shape or not np.array_equal(got.astype(np.int64), want):
            raise RuntimeError(
                f"language surface is not row-aligned to the frozen surface: {name} differs. A "
                f"row-shifted array has the same length and the same multiset, so this is checked "
                f"element-wise on purpose."
            )
    if features.shape[1] != FROZEN_FEATURE_DIM:
        raise RuntimeError(f"joint prefix width is {features.shape[1]}, expected {FROZEN_FEATURE_DIM}")

    report_path = SUBSTRATE_DIR / "language_build_report.json"
    report = json.loads(report_path.read_text(encoding="ascii"))
    prompt_swap = float(report["prompt_swap_relative_error"])
    if prompt_swap <= 1e-4:
        raise RuntimeError(
            f"the language surface's own build report records prompt_swap={prompt_swap:.3e}, so it is "
            f"language-blind and is the frozen surface wearing this one's name"
        )
    if report.get("partial"):
        raise RuntimeError("language surface was built with --limit-tasks and must not be consumed")

    return G39Language(
        state_features=features,
        available=np.ones(features.shape[0], dtype=bool),
        sha256="+".join(digests) if digests else "",
        prompt_swap_relative_error=prompt_swap,
    )
