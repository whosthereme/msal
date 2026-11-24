from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frozen_data import sha256_file

PACKAGE = Path(__file__).resolve().parents[2]
PARALLEL_LABELS_NPZ = (
    PACKAGE
    / "runs"
    / "20260728_g15_parallel_phase_labels_v1"
    / "data"
    / "parallel_phase_labels.npz"
)


@dataclass
class ParallelPhaseLabels:

    phase_indices: np.ndarray
    phase_names: tuple[str, ...]
    sha256: str

    @property
    def phase_count(self) -> int:
        return len(self.phase_names)

    def index_of(self, name: str) -> int:
        return self.phase_names.index(name)


def load_parallel_phase_labels(data, verify_alignment: bool = True) -> ParallelPhaseLabels:
    if not PARALLEL_LABELS_NPZ.exists():
        raise FileNotFoundError(f"parallel phase labels missing: {PARALLEL_LABELS_NPZ}")
    digest = sha256_file(PARALLEL_LABELS_NPZ)
    payload = np.load(PARALLEL_LABELS_NPZ)
    indices = payload["phase_indices"].astype(np.int64)
    if verify_alignment:
        if len(indices) != len(data.task_ids):
            raise RuntimeError(
                f"phase labels have {len(indices)} rows, substrate has {len(data.task_ids)}"
            )
        for column in ("task_ids", "episode_indices", "step_indices", "task_split_codes"):
            if not np.array_equal(payload[column].astype(np.int64), getattr(data, column)):
                raise RuntimeError(f"phase labels disagree with the substrate on {column}")
    return ParallelPhaseLabels(
        phase_indices=indices,
        phase_names=tuple(str(name) for name in payload["phase_names"]),
        sha256=digest,
    )
