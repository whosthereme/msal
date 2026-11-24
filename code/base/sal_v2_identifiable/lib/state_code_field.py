from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_PARENT = Path(__file__).resolve().parents[2]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from joint_sal_velocity import JointSALConfig, JointSALVelocityField

CODE_SOURCES = ("per_task", "from_features")


class StateCodedSALField(nn.Module):

    def __init__(
        self,
        config: JointSALConfig,
        source_task_count: int,
        *,
        code_source: str,
        code_input_dim: int | None = None,
        code_hidden_dim: int = 128,
        code_depth: int = 1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if code_depth < 1:
            raise ValueError(f"code_depth must be at least 1, got {code_depth}")
        if code_source not in CODE_SOURCES:
            raise ValueError(f"unknown code_source {code_source!r}; choose from {CODE_SOURCES}")
        if code_source == "from_features" and not code_input_dim:
            raise ValueError("from_features needs a positive code_input_dim")
        if code_source == "per_task" and freeze_encoder:
            raise ValueError("per_task has no encoder to freeze")
        self.config = config
        self.code_source = code_source
        self.source_task_count = int(source_task_count)
        self.code_input_dim = int(code_input_dim or 0)
        self.code_hidden_dim = int(code_hidden_dim)
        self.code_depth = int(code_depth)
        self.frozen_encoder = bool(freeze_encoder)

        self.field = JointSALVelocityField(config, source_task_count=source_task_count)

        if code_source == "per_task":
            self.encoder = None
            self.register_buffer("code_mean", torch.zeros(0))
            self.register_buffer("code_std", torch.ones(0))
        else:
            del self.field.source_codes
            self.register_buffer("code_mean", torch.zeros(self.code_input_dim))
            self.register_buffer("code_std", torch.ones(self.code_input_dim))
            layers: list[nn.Module] = []
            width = self.code_input_dim
            for _ in range(self.code_depth):
                layers += [nn.Linear(width, code_hidden_dim), nn.SiLU(), nn.LayerNorm(code_hidden_dim)]
                width = code_hidden_dim
            layers.append(nn.Linear(width, config.k_max))
            self.encoder = nn.Sequential(*layers)
            nn.init.normal_(self.encoder[-1].weight, std=0.01)
            nn.init.zeros_(self.encoder[-1].bias)
            if freeze_encoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad_(False)


    def set_code_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if self.encoder is None:
            raise ValueError("per_task has no code input to normalize")
        if mean.shape != (self.code_input_dim,) or std.shape != (self.code_input_dim,):
            raise ValueError(
                f"normalizer must have shape ({self.code_input_dim},), got "
                f"{tuple(mean.shape)} and {tuple(std.shape)}"
            )
        self.code_mean.copy_(mean.detach().float())
        self.code_std.copy_(std.detach().float().clamp_min(1e-5))


    def predict_codes(
        self, task_rows: torch.Tensor | None, code_features: torch.Tensor | None
    ) -> torch.Tensor:
        if self.code_source == "per_task":
            if task_rows is None:
                raise ValueError("per_task needs task_rows")
            return self.field.source_codes(task_rows)
        if code_features is None:
            raise ValueError("from_features needs code_features")
        if code_features.shape[1] != self.code_input_dim:
            raise ValueError(
                f"code_features must be {self.code_input_dim}-d, got {code_features.shape[1]}"
            )
        scaled = (code_features.float() - self.code_mean) / self.code_std
        return self.encoder(scaled)


    def forward_with_rows(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        task_rows: torch.Tensor | None = None,
        *,
        code_features: torch.Tensor | None = None,
        code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base, atoms = self.field.components(state_features, noisy_actions, time, None)
        codes = (
            code_override
            if code_override is not None
            else self.predict_codes(task_rows, code_features)
        )
        if codes.shape != (state_features.shape[0], self.config.k_max):
            raise ValueError(
                f"codes must have shape [{state_features.shape[0]}, {self.config.k_max}], "
                f"got {tuple(codes.shape)}"
            )
        return base + torch.einsum("bk,bktd->btd", codes, atoms)

    def atoms_for(
        self, state_features: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        _, atoms = self.field.components(state_features, noisy_actions, time, None)
        return atoms


    def parameter_split(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        atom = sum(p.numel() for p in self.field.atom_out.parameters())
        if self.code_source == "per_task":
            code = int(self.field.source_codes.weight.numel())
        else:
            code = int(sum(p.numel() for p in self.encoder.parameters()))
        code_trainable = 0 if self.frozen_encoder else code
        return {
            "trainable_parameters": int(trainable),
            "task_conditioned_parameters": int(atom + code),
            "task_conditioned_parameters_trainable": int(atom + code_trainable),
            "shared_trunk_parameters": int(trainable - atom - code_trainable),
            "code_source_parameters": code,
            "code_source_parameters_trainable": int(code_trainable),
            "atom_head_parameters": int(atom),
        }
