from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from joint_sal_velocity import JointSALConfig, JointSALVelocityField

from .task_algebra import split_residue

SOURCE_NPZ = (
    PACKAGE
    / "runs"
    / "20260722_g4_morse_spatial_source_data_spatial_v2"
    / "data"
    / "source_spatial_state_action_fields.npz"
)
BASELINE_CHECKPOINT = (
    PACKAGE
    / "runs"
    / "20260722_g4_morse_spatial_joint_global_sal_source_spatial_v2"
    / "data"
    / "joint_global_sal.pt"
)
BASELINE_SHA256 = "2d232cd6ef682024b1065dce155d051f99aea185723bc18b70a46c4c59ed7763"
PHASE_NPZ = (
    PACKAGE
    / "runs"
    / "20260724_g6_morse_spatial_phase_label_audit_v2"
    / "data"
    / "source_phase_labels.npz"
)

SPLIT_TRAIN, SPLIT_SELECTION, SPLIT_ROBUSTNESS = 0, 1, 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SourceData:

    state_features: np.ndarray
    action_chunks: np.ndarray
    action_valid_masks: np.ndarray
    task_ids: np.ndarray
    direction_indices: np.ndarray
    radius_indices: np.ndarray
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


def load_source_data(verify_hash: bool = True) -> SourceData:
    if not SOURCE_NPZ.exists():
        raise FileNotFoundError(f"frozen source dataset missing: {SOURCE_NPZ}")
    digest = sha256_file(SOURCE_NPZ) if verify_hash else ""
    payload = np.load(SOURCE_NPZ)
    task_ids = payload["task_ids"].astype(np.int64)
    residues = split_residue(np.unique(task_ids))
    if (residues == 3).any():
        raise RuntimeError(
            "frozen source dataset unexpectedly contains sealed development tasks"
        )
    return SourceData(
        state_features=payload["state_features"],
        action_chunks=payload["action_chunks"],
        action_valid_masks=payload["action_valid_masks"],
        task_ids=task_ids,
        direction_indices=payload["direction_indices"].astype(np.int64),
        radius_indices=payload["radius_indices"].astype(np.int64),
        episode_indices=payload["episode_indices"].astype(np.int64),
        step_indices=payload["step_indices"].astype(np.int64),
        split_codes=payload["split_codes"].astype(np.int64),
        sha256=digest,
    )


@dataclass
class PhaseLabels:

    phase_indices: np.ndarray
    phase_names: tuple[str, ...]
    sha256: str

    @property
    def phase_count(self) -> int:
        return len(self.phase_names)

    def index_of(self, name: str) -> int:
        return self.phase_names.index(name)


def load_phase_labels(data: SourceData, verify_alignment: bool = True) -> PhaseLabels:
    if not PHASE_NPZ.exists():
        raise FileNotFoundError(f"frozen phase labels missing: {PHASE_NPZ}")
    payload = np.load(PHASE_NPZ)
    indices = payload["phase_indices"].astype(np.int64)
    if verify_alignment:
        if indices.shape[0] != data.task_ids.shape[0]:
            raise RuntimeError(
                f"phase labels have {indices.shape[0]} rows, source has "
                f"{data.task_ids.shape[0]}"
            )
        for key, expected in (
            ("task_ids", data.task_ids),
            ("episode_indices", data.episode_indices),
            ("step_indices", data.step_indices),
            ("split_codes", data.split_codes),
        ):
            if not np.array_equal(payload[key].astype(np.int64), expected):
                raise RuntimeError(f"phase labels are not row-aligned to source: {key}")
    return PhaseLabels(
        phase_indices=indices,
        phase_names=tuple(str(name) for name in payload["phase_names"]),
        sha256=sha256_file(PHASE_NPZ),
    )


def load_baseline_field(
    device: str = "cuda", verify_hash: bool = True
) -> tuple[JointSALVelocityField, dict[str, Any]]:
    if verify_hash:
        digest = sha256_file(BASELINE_CHECKPOINT)
        if digest != BASELINE_SHA256:
            raise RuntimeError(
                f"baseline checkpoint hash mismatch: expected {BASELINE_SHA256}, got {digest}"
            )
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = JointSALConfig(**checkpoint["config"])
    model = JointSALVelocityField(
        config, source_task_count=len(checkpoint["source_task_ids"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


class DeviceCache:

    def __init__(self, data: SourceData, device: str = "cuda") -> None:
        self.device = device
        self.state_features = torch.from_numpy(data.state_features).to(device)
        self.action_chunks = torch.from_numpy(data.action_chunks).to(device)
        self.action_valid_masks = torch.from_numpy(data.action_valid_masks).to(device)

    def batch(self, rows: np.ndarray, *, seed: int) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(np.asarray(rows), device=self.device, dtype=torch.long)
        state = self.state_features[index].float()
        actions = self.action_chunks[index]
        masks = self.action_valid_masks[index]
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noise = torch.randn(actions.shape, generator=generator, device=self.device)
        time = (
            torch.rand(actions.shape[0], generator=generator, device=self.device) * 0.998
            + 0.001
        )
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        return {
            "state": state,
            "actions": actions,
            "masks": masks,
            "time": time,
            "x_t": x_t,
            "target": noise - actions,
        }


def flow_batch(
    data: SourceData,
    rows: np.ndarray,
    *,
    seed: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    state = torch.from_numpy(data.state_features[rows].astype(np.float32)).to(device)
    actions = torch.from_numpy(data.action_chunks[rows]).to(device)
    masks = torch.from_numpy(data.action_valid_masks[rows]).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(actions.shape, generator=generator, device=device)
    time = torch.rand(actions.shape[0], generator=generator, device=device) * 0.998 + 0.001
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    return {
        "state": state,
        "actions": actions,
        "masks": masks,
        "time": time,
        "x_t": x_t,
        "target": noise - actions,
    }


def phase_lookup_table(
    data: SourceData, phase_labels: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    table: dict[tuple[int, int], list[np.ndarray]] = {}
    for row in rows:
        key = (int(data.task_ids[row]), int(phase_labels[row]))
        table.setdefault(key, []).append(data.action_chunks[row])
    return np.array(
        [np.mean(np.stack(values), axis=0) for values in table.values()], dtype=np.float32
    )
