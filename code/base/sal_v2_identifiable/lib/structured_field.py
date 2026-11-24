from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_PARENT = Path(__file__).resolve().parents[2]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from joint_sal_velocity import JointSALConfig, JointSALVelocityField

from .task_algebra import design_matrix, task_indices
from .task_algebra_v5 import FEATURE_SETS_V5, design_matrix_for_tasks


class StructuredCodeField(JointSALVelocityField):

    def __init__(
        self,
        config: JointSALConfig,
        source_task_ids: list[int],
        feature_set: str = "physical_xy_radial",
    ) -> None:
        super().__init__(config, source_task_count=len(source_task_ids))
        directions, radii = task_indices(source_task_ids)
        matrix, names = design_matrix(directions, radii, feature_set)
        self.feature_set = feature_set
        self.feature_names = tuple(names)
        self.source_task_ids = list(int(task_id) for task_id in source_task_ids)
        self.register_buffer(
            "task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True
        )
        self.code_map = nn.Linear(len(names), config.k_max, bias=False)
        nn.init.normal_(self.code_map.weight, std=0.02)
        with torch.no_grad():
            self.source_codes.weight.zero_()
        self.source_codes.weight.requires_grad_(False)

    @property
    def structured_codes(self) -> torch.Tensor:
        return self.code_map(self.task_features)

    def codes_for_rows(self, task_rows: torch.Tensor) -> torch.Tensor:
        return self.code_map(self.task_features[task_rows])

    def codes_for_task_ids(self, task_ids: list[int]) -> torch.Tensor:
        directions, radii = task_indices(task_ids)
        matrix, names = design_matrix(directions, radii, self.feature_set)
        if tuple(names) != self.feature_names:
            raise ValueError("feature set mismatch between model and query")
        features = torch.tensor(
            matrix, dtype=self.task_features.dtype, device=self.task_features.device
        )
        return self.code_map(features)


class StructuredCodeFieldV5(StructuredCodeField):

    def __init__(
        self,
        config: JointSALConfig,
        source_task_ids: list[int],
        feature_set: str = "physical_xy_radial_v5",
    ) -> None:
        if feature_set not in FEATURE_SETS_V5:
            raise ValueError(
                f"unknown v5 feature set {feature_set!r}; choose from {sorted(FEATURE_SETS_V5)}. "
                f"v4 names are not accepted here: the grids differ, so a v4 name would build a "
                f"45-degree design on a 22.5-degree pool"
            )
        JointSALVelocityField.__init__(self, config, source_task_count=len(source_task_ids))
        ids = [int(task_id) for task_id in source_task_ids]
        matrix, names = design_matrix_for_tasks(ids, feature_set)
        self.feature_set = feature_set
        self.feature_names = tuple(names)
        self.source_task_ids = ids
        self.register_buffer(
            "task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True
        )
        self.code_map = nn.Linear(len(names), config.k_max, bias=False)
        nn.init.normal_(self.code_map.weight, std=0.02)
        with torch.no_grad():
            self.source_codes.weight.zero_()
        self.source_codes.weight.requires_grad_(False)

    def codes_for_task_ids(self, task_ids: list[int]) -> torch.Tensor:
        matrix, names = design_matrix_for_tasks(
            [int(task_id) for task_id in task_ids], self.feature_set
        )
        if tuple(names) != self.feature_names:
            raise ValueError("feature set mismatch between model and query")
        features = torch.tensor(
            matrix, dtype=self.task_features.dtype, device=self.task_features.device
        )
        return self.code_map(features)


def group_sparsity_penalty(codes: torch.Tensor, atom_fields: torch.Tensor) -> torch.Tensor:
    if atom_fields.ndim != 4:
        raise ValueError(f"expected (B, K, T, D) atom fields, got {tuple(atom_fields.shape)}")
    gain = atom_fields.flatten(2).pow(2).sum(dim=-1).mean(dim=0).clamp_min(1e-12).sqrt()
    spread = codes.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
    return (gain.detach() * spread).sum()


def channel_weights(
    action_dim: int,
    *,
    dead_dims: tuple[int, ...] = (3, 4, 5),
    gripper_dim: int = 6,
    gripper_weight: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    weights = torch.ones(action_dim, dtype=torch.float32, device=device)
    for dim in dead_dims:
        if 0 <= dim < action_dim:
            weights[dim] = 0.0
    if 0 <= gripper_dim < action_dim:
        weights[gripper_dim] = gripper_weight
    return weights


def verify_dead_channels(action_chunks: np.ndarray, dims: tuple[int, ...] = (3, 4, 5)) -> dict:
    report = {}
    for dim in dims:
        column = action_chunks[..., dim]
        report[f"dim_{dim}"] = {
            "max_abs": float(np.abs(column).max()),
            "nonzero_count": int((column != 0).sum()),
        }
    return report
