from __future__ import annotations

from typing import Sequence

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_PARENT = Path(__file__).resolve().parents[2]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from joint_sal_velocity import JointSALVelocityField

from .structured_field import StructuredCodeField


CONTEXT_ARMS = ("context_raw", "context_phi", "context_phi_wide")

WIDE_BOTTLENECK = 64


class ContextConditionedField(StructuredCodeField):

    def __init__(
        self,
        config,
        source_task_count: int,
        *,
        arm: str = "context_phi",
        task_features: np.ndarray,
    ) -> None:
        if arm not in CONTEXT_ARMS:
            raise ValueError(f"unknown context arm {arm!r}; choose from {CONTEXT_ARMS}")
        if task_features.ndim != 2 or task_features.shape[0] != source_task_count:
            raise ValueError(
                f"task_features must be [source_task_count, F]; got {task_features.shape} "
                f"for {source_task_count} tasks"
            )
        JointSALVelocityField.__init__(self, config, source_task_count=source_task_count)
        self.arm = arm
        self.register_buffer(
            "task_features", torch.tensor(task_features, dtype=torch.float32), persistent=True
        )
        descriptor_dim = int(task_features.shape[1])
        output_dim = config.chunk_length * config.action_dim
        del self.atom_out
        del self.source_codes
        del self.base_out
        if arm == "context_phi_wide":
            self.context_bottleneck = nn.Sequential(
                nn.Linear(config.hidden_dim + descriptor_dim, WIDE_BOTTLENECK),
                nn.SiLU(),
            )
            self.context_head = nn.Linear(WIDE_BOTTLENECK, output_dim)
        else:
            self.context_bottleneck = None
            self.context_head = nn.Linear(config.hidden_dim + descriptor_dim, output_dim)

    @property
    def descriptor_dim(self) -> int:
        return int(self.task_features.shape[1])

    def head_output(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        task_rows: torch.Tensor,
    ) -> torch.Tensor:
        from joint_sal_velocity import sinusoidal_time_embedding

        batch_size = state_features.shape[0]
        state = (state_features.float() - self.state_mean) / self.state_std
        state_hidden = self.state_encoder(state)
        action_hidden = self.action_encoder(noisy_actions.reshape(batch_size, -1).float())
        time_hidden = sinusoidal_time_embedding(time.float(), self.config.time_dim)
        hidden = self.trunk(torch.cat([state_hidden, action_hidden, time_hidden], dim=-1))
        descriptor = self.task_features[task_rows.long()]
        joint = torch.cat([hidden, descriptor.to(hidden.dtype)], dim=-1)
        if self.context_bottleneck is not None:
            joint = self.context_bottleneck(joint)
        return self.context_head(joint).reshape(
            batch_size, self.config.chunk_length, self.config.action_dim
        )

    def forward_with_rows(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        task_rows: torch.Tensor,
    ) -> torch.Tensor:
        return self.head_output(state_features, noisy_actions, time, task_rows)

    def parameter_split(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "trainable_parameters": int(total),
            "task_conditioned_parameters": 0,
            "shared_trunk_parameters": int(total),
        }


def build_context_features(
    task_ids: Sequence[int],
    *,
    arm: str,
    feature_set: str,
    pool: str,
    radius_count: int,
) -> np.ndarray:
    if arm not in CONTEXT_ARMS:
        raise ValueError(f"unknown context arm {arm!r}")
    if pool == "v4":
        from .task_algebra import design_matrix, task_indices

        directions, radii = task_indices(task_ids)
        matrix, _ = design_matrix(directions, radii, feature_set)
    elif pool == "v5":
        from .task_algebra_v5 import design_matrix_for_tasks

        matrix, _ = design_matrix_for_tasks(task_ids, feature_set)
        from .pool_v5 import build_pool

        by_id = {task.task_id: task for task in build_pool()}
        directions = np.array([by_id[int(t)].direction_16 for t in task_ids])
        radii = np.array([by_id[int(t)].radius_index for t in task_ids])
    else:
        raise ValueError(f"unknown pool {pool!r}; choose 'v4' or 'v5'")

    if arm in ("context_phi", "context_phi_wide"):
        return np.asarray(matrix, dtype=np.float64)
    direction_count = 8 if pool == "v4" else 16
    theta = 2.0 * np.pi * directions.astype(np.float64) / direction_count
    span = max(radius_count - 1, 1)
    scaled = 2.0 * radii.astype(np.float64) / span - 1.0
    return np.stack([np.cos(theta), np.sin(theta), scaled], axis=1)
