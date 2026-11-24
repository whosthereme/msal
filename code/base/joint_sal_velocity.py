from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class JointSALConfig:
    state_dim: int = 4800
    chunk_length: int = 16
    action_dim: int = 7
    k_max: int = 16
    hidden_dim: int = 512
    state_hidden_dim: int = 512
    action_hidden_dim: int = 256
    time_dim: int = 64
    phase_count: int = 0
    phase_dim: int = 0


def sinusoidal_time_embedding(time: torch.Tensor, dimension: int = 64) -> torch.Tensor:
    if time.ndim != 1 or dimension % 2:
        raise ValueError("time must be 1D and dimension must be even")
    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), dimension // 2, device=time.device)
    )
    angles = time[:, None] * frequencies[None, :] * 2.0 * math.pi
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class JointSALVelocityField(nn.Module):
    def __init__(self, config: JointSALConfig, source_task_count: int) -> None:
        super().__init__()
        if (config.phase_count == 0) != (config.phase_dim == 0):
            raise ValueError("phase_count and phase_dim must both be zero or both be positive")
        self.config = config
        output_dim = config.chunk_length * config.action_dim
        self.register_buffer("state_mean", torch.zeros(config.state_dim))
        self.register_buffer("state_std", torch.ones(config.state_dim))
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.state_hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(output_dim, config.action_hidden_dim),
            nn.SiLU(),
        )
        self.phase_embedding = (
            nn.Embedding(config.phase_count, config.phase_dim) if config.phase_count else None
        )
        trunk_input = (
            config.state_hidden_dim + config.action_hidden_dim + config.time_dim + config.phase_dim
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.base_out = nn.Linear(config.hidden_dim, output_dim)
        self.atom_out = nn.Linear(config.hidden_dim, config.k_max * output_dim)
        self.source_codes = nn.Embedding(source_task_count, config.k_max)
        nn.init.normal_(self.source_codes.weight, std=0.05)
        nn.init.zeros_(self.base_out.weight)
        nn.init.zeros_(self.base_out.bias)
        nn.init.normal_(self.atom_out.weight, std=0.002)
        nn.init.zeros_(self.atom_out.bias)

    @property
    def output_dim(self) -> int:
        return self.config.chunk_length * self.config.action_dim

    def set_state_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.state_mean.shape or std.shape != self.state_std.shape:
            raise ValueError("State normalizer shape mismatch")
        self.state_mean.copy_(mean)
        self.state_std.copy_(std.clamp_min(1e-5))

    def components(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        phase_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state_features.shape[0]
        state = (state_features.float() - self.state_mean) / self.state_std
        state_hidden = self.state_encoder(state)
        action_hidden = self.action_encoder(noisy_actions.reshape(batch_size, -1).float())
        time_hidden = sinusoidal_time_embedding(time.float(), self.config.time_dim)
        inputs = [state_hidden, action_hidden, time_hidden]
        if self.phase_embedding is not None:
            if phase_indices is None or phase_indices.shape != (batch_size,):
                raise ValueError("phase_indices must have shape [batch] for a phase-conditioned model")
            inputs.append(self.phase_embedding(phase_indices.long()))
        elif phase_indices is not None:
            raise ValueError("phase_indices were provided to a model without phase conditioning")
        hidden = self.trunk(torch.cat(inputs, dim=-1))
        base = self.base_out(hidden).reshape(batch_size, self.config.chunk_length, self.config.action_dim)
        atoms = self.atom_out(hidden).reshape(
            batch_size, self.config.k_max, self.config.chunk_length, self.config.action_dim
        )
        return base, atoms

    def forward_with_code(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        code: torch.Tensor,
        active_rank: int | None = None,
        phase_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base, atoms = self.components(state_features, noisy_actions, time, phase_indices)
        if active_rank is not None:
            if not 1 <= active_rank <= self.config.k_max:
                raise ValueError("active_rank out of range")
            code = code.clone()
            code[:, active_rank:] = 0.0
        return base + torch.einsum("bk,bktd->btd", code, atoms)

    def forward(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        source_task_rows: torch.Tensor,
        active_rank: int | None = None,
        phase_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_code(
            state_features,
            noisy_actions,
            time,
            self.source_codes(source_task_rows),
            active_rank=active_rank,
            phase_indices=phase_indices,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        state_features: torch.Tensor,
        code: torch.Tensor,
        noise: torch.Tensor,
        active_rank: int,
        denoise_steps: int = 10,
        phase_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions = noise.float()
        dt = -1.0 / denoise_steps
        for step in range(denoise_steps):
            time = torch.full(
                (state_features.shape[0],), 1.0 - step / denoise_steps, device=state_features.device
            )
            velocity = self.forward_with_code(
                state_features,
                actions,
                time,
                code,
                active_rank,
                phase_indices,
            )
            actions = actions + dt * velocity
        return actions.clamp(-1.0, 1.0)


def output_orthogonality_error(atoms: torch.Tensor) -> torch.Tensor:
    fields = atoms.permute(1, 0, 2, 3).reshape(atoms.shape[1], -1)
    fields = fields / fields.norm(dim=1, keepdim=True).clamp_min(1e-8)
    gram = fields @ fields.T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return ((gram - identity) ** 2).sum() / (gram.shape[0] * (gram.shape[0] - 1))


@torch.no_grad()
def canonicalize_codes_and_atoms(model: JointSALVelocityField) -> dict[str, Any]:
    probe_state = torch.randn(4, model.config.state_dim, device=model.state_mean.device)
    probe_actions = torch.randn(
        4, model.config.chunk_length, model.config.action_dim, device=model.state_mean.device
    )
    probe_time = torch.linspace(0.1, 0.9, 4, device=model.state_mean.device)
    probe_codes = model.source_codes.weight[:4].clone()
    probe_phases = (
        torch.zeros(4, dtype=torch.long, device=model.state_mean.device)
        if model.config.phase_count
        else None
    )
    before = model.forward_with_code(
        probe_state, probe_actions, probe_time, probe_codes, phase_indices=probe_phases
    )

    codes = model.source_codes.weight.detach().double()
    mean = codes.mean(dim=0)
    centered = codes - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    rotation = vh.T
    atom_weight = model.atom_out.weight.detach().double().reshape(
        model.config.k_max, model.output_dim, model.config.hidden_dim
    )
    atom_bias = model.atom_out.bias.detach().double().reshape(model.config.k_max, model.output_dim)
    base_weight = model.base_out.weight.detach().double()
    base_bias = model.base_out.bias.detach().double()
    base_weight = base_weight + torch.einsum("k,kah->ah", mean, atom_weight)
    base_bias = base_bias + torch.einsum("k,ka->a", mean, atom_bias)
    rotated_weight = torch.einsum("kj,kah->jah", rotation, atom_weight)
    rotated_bias = torch.einsum("kj,ka->ja", rotation, atom_bias)
    rotated_codes = centered @ rotation
    model.base_out.weight.copy_(base_weight.to(model.base_out.weight))
    model.base_out.bias.copy_(base_bias.to(model.base_out.bias))
    model.atom_out.weight.copy_(rotated_weight.reshape_as(model.atom_out.weight).to(model.atom_out.weight))
    model.atom_out.bias.copy_(rotated_bias.reshape_as(model.atom_out.bias).to(model.atom_out.bias))
    model.source_codes.weight.copy_(rotated_codes.to(model.source_codes.weight))

    after = model.forward_with_code(
        probe_state,
        probe_actions,
        probe_time,
        model.source_codes.weight[:4],
        phase_indices=probe_phases,
    )
    relative_error = float((before - after).norm().div(before.norm().clamp_min(1e-12)).item())
    return {
        "relative_output_error": relative_error,
        "singular_values": singular_values.cpu().tolist(),
        "code_mean_norm_before": float(mean.norm().item()),
        "code_mean_norm_after": float(model.source_codes.weight.mean(dim=0).norm().item()),
    }
