from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from joint_sal_velocity import JointSALConfig, JointSALVelocityField

CLOCK_MODES = ("no_clock", "oracle_clock", "predicted_clock")

CLOCK_DIM = 32


def clock_embedding(clock: torch.Tensor, dimension: int = CLOCK_DIM) -> torch.Tensor:
    if dimension % 2:
        raise ValueError("clock embedding dimension must be even")
    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(1_000.0), dimension // 2, device=clock.device)
    )
    angles = clock.float()[:, None] * frequencies[None, :] * 2.0 * math.pi
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class ClockedSALField(nn.Module):

    def __init__(
        self,
        config: JointSALConfig,
        source_task_count: int,
        *,
        clock_mode: str,
        clock_dim: int = CLOCK_DIM,
    ) -> None:
        super().__init__()
        if clock_mode not in CLOCK_MODES:
            raise ValueError(f"unknown clock_mode {clock_mode!r}; choose from {CLOCK_MODES}")
        if config.phase_count:
            raise ValueError(
                "phase conditioning consumes an oracle phase index and is not deployable; "
                "the clock arms here replace it rather than stacking on it"
            )
        self.config = config
        self.clock_mode = clock_mode
        self.uses_clock = clock_mode != "no_clock"
        self.clock_dim = clock_dim if self.uses_clock else 0

        self.field = JointSALVelocityField(config, source_task_count=source_task_count)
        if self.uses_clock:
            trunk_input = (
                config.state_hidden_dim + config.action_hidden_dim + config.time_dim
            )
            self.clock_proj = nn.Sequential(
                nn.Linear(clock_dim, clock_dim),
                nn.SiLU(),
                nn.Linear(clock_dim, trunk_input),
            )
            nn.init.zeros_(self.clock_proj[-1].weight)
            nn.init.zeros_(self.clock_proj[-1].bias)
        else:
            self.clock_proj = None

    @property
    def output_dim(self) -> int:
        return self.field.output_dim

    def set_state_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.field.set_state_normalizer(mean, std)

    def source_codes(self, task_rows: torch.Tensor) -> torch.Tensor:
        return self.field.source_codes(task_rows)

    def components(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        clock: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state_features.shape[0]
        if self.uses_clock:
            if clock is None or clock.shape != (batch_size,):
                raise ValueError("clock must have shape [batch] for a clocked model")
        elif clock is not None:
            raise ValueError("clock was provided to a model without a clock channel")

        inner = self.field
        state = (state_features.float() - inner.state_mean) / inner.state_std
        state_hidden = inner.state_encoder(state)
        action_hidden = inner.action_encoder(noisy_actions.reshape(batch_size, -1).float())
        from joint_sal_velocity import sinusoidal_time_embedding

        time_hidden = sinusoidal_time_embedding(time.float(), inner.config.time_dim)
        trunk_input = torch.cat([state_hidden, action_hidden, time_hidden], dim=-1)
        if self.uses_clock:
            trunk_input = trunk_input + self.clock_proj(clock_embedding(clock, self.clock_dim))
        hidden = inner.trunk(trunk_input)
        base = inner.base_out(hidden).reshape(
            batch_size, inner.config.chunk_length, inner.config.action_dim
        )
        atoms = inner.atom_out(hidden).reshape(
            batch_size, inner.config.k_max, inner.config.chunk_length, inner.config.action_dim
        )
        return base, atoms

    def forward_with_code(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        code: torch.Tensor,
        clock: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base, atoms = self.components(state_features, noisy_actions, time, clock)
        return base + torch.einsum("bk,bktd->btd", code, atoms)


def episode_lengths(
    task_ids: np.ndarray, episode_indices: np.ndarray, step_indices: np.ndarray
) -> dict[tuple[int, int], int]:
    lengths: dict[tuple[int, int], int] = {}
    for task, episode, step in zip(task_ids, episode_indices, step_indices):
        key = (int(task), int(episode))
        lengths[key] = max(lengths.get(key, 0), int(step))
    return lengths


def oracle_clock_values(
    task_ids: np.ndarray,
    episode_indices: np.ndarray,
    step_indices: np.ndarray,
) -> np.ndarray:
    lengths = episode_lengths(task_ids, episode_indices, step_indices)
    denominators = np.array(
        [max(lengths[(int(t), int(e))], 1) for t, e in zip(task_ids, episode_indices)],
        dtype=np.float64,
    )
    return np.clip(step_indices.astype(np.float64) / denominators, 0.0, 1.0)


def fit_length_predictor(
    descriptors: np.ndarray, lengths: np.ndarray, *, ridge: float = 1.0
) -> np.ndarray:
    design = np.asarray(descriptors, dtype=np.float64)
    target = np.asarray(lengths, dtype=np.float64)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def predicted_clock_values(
    descriptors: np.ndarray, coefficients: np.ndarray, step_indices: np.ndarray
) -> np.ndarray:
    predicted = np.asarray(descriptors, dtype=np.float64) @ np.asarray(
        coefficients, dtype=np.float64
    )
    return np.clip(step_indices.astype(np.float64) / np.maximum(predicted, 1.0), 0.0, 1.0)
