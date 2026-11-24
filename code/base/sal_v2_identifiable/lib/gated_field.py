from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PARENT_ROOT = Path(__file__).resolve().parents[2]
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

from joint_sal_velocity import JointSALConfig, JointSALVelocityField

GATE_MODES = ("global", "phase_gated", "descriptor_gated")


class CellGatedField(nn.Module):

    def __init__(
        self,
        config: JointSALConfig,
        source_task_count: int,
        *,
        gate_mode: str,
        cell_count: int = 1,
        descriptor_dim: int = 6,
    ) -> None:
        super().__init__()
        if gate_mode not in GATE_MODES:
            raise ValueError(f"unknown gate_mode {gate_mode!r}; choose from {GATE_MODES}")
        if gate_mode == "global" and cell_count != 1:
            raise ValueError("the global arm has exactly one cell")
        if cell_count < 1:
            raise ValueError("cell_count must be >= 1")
        if config.k_max % cell_count:
            raise ValueError(
                f"k_max={config.k_max} must divide evenly into {cell_count} cells so that "
                "total code capacity is matched across arms"
            )
        self.config = config
        self.gate_mode = gate_mode
        self.cell_count = cell_count
        self.per_cell_rank = config.k_max // cell_count

        self.field = JointSALVelocityField(config, source_task_count=source_task_count)

        self.cell_codes = nn.Parameter(
            torch.zeros(source_task_count, cell_count, self.per_cell_rank)
        )
        nn.init.normal_(self.cell_codes, std=0.01)

        self.cell_offsets = nn.Parameter(
            torch.zeros(cell_count, config.chunk_length, config.action_dim)
        )

        if gate_mode == "descriptor_gated":
            self.descriptor_gate = nn.Sequential(
                nn.Linear(descriptor_dim, 64),
                nn.SiLU(),
                nn.Linear(64, cell_count),
            )
        else:
            self.descriptor_gate = None


    def gate_weights(
        self,
        batch_size: int,
        *,
        phase_indices: torch.Tensor | None,
        descriptors: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if self.gate_mode == "global":
            return torch.ones(batch_size, 1, device=device)
        if self.gate_mode == "phase_gated":
            if phase_indices is None:
                raise ValueError("phase_gated requires phase_indices")
            if phase_indices.shape != (batch_size,):
                raise ValueError(
                    f"phase_indices must have shape [{batch_size}], got {tuple(phase_indices.shape)}"
                )
            return torch.nn.functional.one_hot(
                phase_indices.long().clamp(0, self.cell_count - 1),
                num_classes=self.cell_count,
            ).to(torch.float32)
        if descriptors is None:
            raise ValueError("descriptor_gated requires descriptors")
        if descriptors.shape[0] != batch_size:
            raise ValueError("descriptors must have one row per batch element")
        return torch.softmax(self.descriptor_gate(descriptors.float()), dim=-1)


    def forward_with_rows(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        task_rows: torch.Tensor,
        *,
        phase_indices: torch.Tensor | None = None,
        descriptors: torch.Tensor | None = None,
        code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = state_features.shape[0]
        base, atoms = self.field.components(state_features, noisy_actions, time, None)
        alpha = self.gate_weights(
            batch_size,
            phase_indices=phase_indices,
            descriptors=descriptors,
            device=state_features.device,
        )
        codes = self.cell_codes[task_rows] if code_override is None else code_override
        if codes.shape[1:] != (self.cell_count, self.per_cell_rank):
            raise ValueError(
                f"codes must have shape [batch, {self.cell_count}, {self.per_cell_rank}], "
                f"got {tuple(codes.shape)}"
            )
        cell_atoms = atoms.reshape(
            batch_size, self.cell_count, self.per_cell_rank,
            self.config.chunk_length, self.config.action_dim,
        )
        per_cell = self.cell_offsets[None] + torch.einsum(
            "bck,bcktd->bctd", codes, cell_atoms
        )
        return base + torch.einsum("bc,bctd->btd", alpha, per_cell)


    def code_energy_by_cell(self) -> torch.Tensor:
        return (self.cell_codes.detach() ** 2).sum(dim=(0, 2))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
