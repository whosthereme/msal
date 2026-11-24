from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frozen_data import PACKAGE, SPLIT_TRAIN, SourceData, sha256_file

G10_RUN = (
    PACKAGE
    / "runs"
    / "20260724_g10_morse_spatial_rollout_distribution_shift_audit_v2_singleton_aligned"
)
TRACE_DIR = G10_RUN / "data" / "rollout_traces"

REGISTERED_METHOD = "baseline_full_rank"

REGISTERED_ROW_COUNT = 1755
REGISTERED_FIELD_MSE_ALL7 = 0.4719
REGISTERED_LOOKUP_MSE_ALL7 = 0.0472

DEAD_ACTION_DIMS = (3, 4, 5)
LIVE_ACTION_DIMS = tuple(dim for dim in range(7) if dim not in DEAD_ACTION_DIMS)

_TASK_IN_NAME = re.compile(r"_task_(\d+)\.npz$")


@dataclass
class OffManifoldSurface:

    state_features: np.ndarray
    expert_actions: np.ndarray
    field_actions: np.ndarray
    task_ids: np.ndarray
    phase_indices: np.ndarray
    phase_names: tuple[str, ...]
    environment_steps: np.ndarray
    novelty_ratios: np.ndarray
    global_distances: np.ndarray
    trace_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.expert_actions)

    def rows_for_task(self, task_id: int) -> np.ndarray:
        return np.flatnonzero(self.task_ids == int(task_id))


def _digest_traces(paths: list[Path]) -> str:
    import hashlib

    combined = hashlib.sha256()
    for path in paths:
        combined.update(sha256_file(path).encode("ascii"))
    return combined.hexdigest()


def load_off_manifold_surface(
    phase_names: tuple[str, ...],
    *,
    method: str = REGISTERED_METHOD,
    verify_row_count: bool = True,
) -> OffManifoldSurface:
    paths = sorted(TRACE_DIR.glob(f"{method}_task_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {method!r} traces under {TRACE_DIR}")

    name_to_index = {name: index for index, name in enumerate(phase_names)}
    features: list[np.ndarray] = []
    expert: list[np.ndarray] = []
    field: list[np.ndarray] = []
    tasks: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    novelty: list[np.ndarray] = []
    distance: list[np.ndarray] = []

    for path in paths:
        match = _TASK_IN_NAME.search(path.name)
        if match is None:
            raise ValueError(f"cannot read a task id out of {path.name!r}")
        payload = np.load(path, allow_pickle=True)
        rows = len(payload["corrective_actions"])
        raw_names = [str(name) for name in payload["phase_names"]]
        unknown = sorted(set(raw_names) - set(name_to_index))
        if unknown:
            raise ValueError(
                f"{path.name} carries phase names {unknown} that are not in the frozen "
                f"label set {phase_names}; refusing to guess an index for them"
            )
        features.append(np.asarray(payload["singleton_state_features"], dtype=np.float32))
        expert.append(np.asarray(payload["corrective_actions"], dtype=np.float64))
        field.append(np.asarray(payload["predicted_actions"], dtype=np.float64))
        tasks.append(np.full(rows, int(match.group(1)), dtype=np.int64))
        phases.append(np.array([name_to_index[name] for name in raw_names], dtype=np.int64))
        steps.append(np.asarray(payload["environment_steps"], dtype=np.int64))
        novelty.append(np.asarray(payload["task_phase_novelty_ratios"], dtype=np.float64))
        distance.append(np.asarray(payload["global_distances"], dtype=np.float64))

    surface = OffManifoldSurface(
        state_features=np.concatenate(features),
        expert_actions=np.concatenate(expert),
        field_actions=np.concatenate(field),
        task_ids=np.concatenate(tasks),
        phase_indices=np.concatenate(phases),
        phase_names=tuple(phase_names),
        environment_steps=np.concatenate(steps),
        novelty_ratios=np.concatenate(novelty),
        global_distances=np.concatenate(distance),
        trace_sha256=_digest_traces(paths),
    )
    if verify_row_count and surface.row_count != REGISTERED_ROW_COUNT:
        raise ValueError(
            f"off-manifold surface has {surface.row_count} rows, but PROTOCOL_G14 "
            f"registered {REGISTERED_ROW_COUNT}; the surface has changed and every g14 "
            "number would be measured against a different set of states"
        )
    return surface


def action_mse(
    predicted: np.ndarray, target: np.ndarray, *, mask: np.ndarray | None = None
) -> dict[str, float]:
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    error = (np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2
    if mask is not None:
        weight = np.asarray(mask, dtype=np.float64)
        if weight.shape != error.shape:
            raise ValueError(f"mask shape {weight.shape} does not match {error.shape}")
        total = float((error * weight).sum())
        count = float(weight.sum())
        live = float((error[:, LIVE_ACTION_DIMS] * weight[:, LIVE_ACTION_DIMS]).sum())
        live_count = float(weight[:, LIVE_ACTION_DIMS].sum())
        return {
            "all7": total / max(count, 1.0),
            "live4": live / max(live_count, 1.0),
            "entries": count,
        }
    return {
        "all7": float(error.mean()),
        "live4": float(error[:, LIVE_ACTION_DIMS].mean()),
        "entries": float(error.size),
    }


def lookup_table_predictions(
    data: SourceData,
    phase_indices: np.ndarray,
    surface: OffManifoldSurface,
    *,
    split: int = SPLIT_TRAIN,
    allowed_tasks: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    rows = data.rows_for_split(split)
    if allowed_tasks is not None:
        keep = np.isin(data.task_ids[rows], np.asarray(allowed_tasks))
        rows = rows[keep]
    emitted = data.action_chunks[:, 0, :].astype(np.float64)

    cell: dict[tuple[int, int], np.ndarray] = {}
    per_task: dict[int, np.ndarray] = {}
    for task in np.unique(data.task_ids[rows]):
        task_rows = rows[data.task_ids[rows] == task]
        per_task[int(task)] = emitted[task_rows].mean(axis=0)
        for phase in np.unique(phase_indices[task_rows]):
            members = task_rows[phase_indices[task_rows] == phase]
            cell[(int(task), int(phase))] = emitted[members].mean(axis=0)
    global_mean = emitted[rows].mean(axis=0)

    predictions = np.empty((surface.row_count, emitted.shape[1]), dtype=np.float64)
    counts = {"cell": 0, "task": 0, "global": 0}
    for index in range(surface.row_count):
        key = (int(surface.task_ids[index]), int(surface.phase_indices[index]))
        if key in cell:
            predictions[index] = cell[key]
            counts["cell"] += 1
        elif key[0] in per_task:
            predictions[index] = per_task[key[0]]
            counts["task"] += 1
        else:
            predictions[index] = global_mean
            counts["global"] += 1
    return predictions, counts
