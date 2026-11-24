from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_PACKAGE = Path(__file__).resolve().parents[2]
if str(_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE))

from joint_sal_velocity import JointSALConfig, JointSALVelocityField


def parity_bottleneck(hidden_dim: int, output_dim: int) -> int:
    linear = hidden_dim * output_dim + output_dim
    return min(
        range(1, 4 * output_dim),
        key=lambda h: abs((hidden_dim * h + h) + (h * output_dim + output_dim) - linear),
    )


class NonlinearAtomHead(nn.Module):

    def __init__(self, hidden_dim: int, k_max: int, output_dim: int, bottleneck: int | None = None) -> None:
        super().__init__()
        if min(hidden_dim, k_max, output_dim) < 1:
            raise ValueError("hidden_dim, k_max and output_dim must all be positive")
        self.bottleneck = parity_bottleneck(hidden_dim, output_dim) if bottleneck is None else int(bottleneck)
        if self.bottleneck < 1:
            raise ValueError("bottleneck must be positive")
        self.hidden_dim, self.k_max, self.output_dim = hidden_dim, k_max, output_dim
        self.w1 = nn.Parameter(torch.empty(k_max, self.bottleneck, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(k_max, self.bottleneck))
        self.w2 = nn.Parameter(torch.empty(k_max, output_dim, self.bottleneck))
        self.b2 = nn.Parameter(torch.zeros(k_max, output_dim))
        nn.init.normal_(self.w1, std=hidden_dim ** -0.5)
        nn.init.normal_(self.w2, std=0.1 * self.bottleneck ** -0.5)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected [batch, {self.hidden_dim}], got {tuple(hidden.shape)}")
        inner = F.silu(torch.einsum("kch,bh->bkc", self.w1, hidden) + self.b1[None])
        return (torch.einsum("koc,bkc->bko", self.w2, inner) + self.b2[None]).reshape(
            hidden.shape[0], self.k_max * self.output_dim
        )


class NonlinearAtomField(JointSALVelocityField):

    def __init__(
        self,
        config: JointSALConfig,
        source_task_count: int,
        bottleneck: int | None = None,
    ) -> None:
        super().__init__(config, source_task_count=source_task_count)
        self.atom_out = NonlinearAtomHead(
            config.hidden_dim, config.k_max, self.output_dim, bottleneck
        )

    @property
    def bottleneck(self) -> int:
        return self.atom_out.bottleneck


def _trunk_activation(
    field: JointSALVelocityField,
    state_features: torch.Tensor,
    noisy_actions: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    handle = field.trunk.register_forward_hook(lambda _m, _i, out: captured.append(out.detach()))
    try:
        with torch.no_grad():
            field.components(state_features, noisy_actions, time)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected exactly one trunk activation, captured {len(captured)}")
    return captured[0]


def linear_map_residual(
    field: JointSALVelocityField,
    state_features: torch.Tensor,
    noisy_actions: torch.Tensor,
    time: torch.Tensor,
    code: torch.Tensor,
) -> float:
    if code.ndim != 2 or code.shape[0] != 1:
        raise ValueError("code must be [1, k_max]; a per-row code does not isolate the head")
    batch = state_features.shape[0]
    if batch <= field.config.hidden_dim:
        raise ValueError(
            f"need more rows than hidden_dim ({field.config.hidden_dim}) or the least-squares "
            f"solve is underdetermined and the residual is 0 by construction; got {batch}"
        )
    with torch.no_grad():
        outputs = field.forward_with_code(
            state_features, noisy_actions, time, code.expand(batch, -1)
        ).reshape(batch, -1)
        hidden = _trunk_activation(field, state_features, noisy_actions, time)
        design = torch.cat([hidden, torch.ones(batch, 1, device=hidden.device)], dim=-1)
        residual = outputs - design @ torch.linalg.lstsq(design, outputs).solution
    return float(residual.norm() / outputs.norm().clamp_min(1e-12))


class LinearBottleneckAtomHead(NonlinearAtomHead):

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected [batch, {self.hidden_dim}], got {tuple(hidden.shape)}")
        inner = torch.einsum("kch,bh->bkc", self.w1, hidden) + self.b1[None]
        return (torch.einsum("koc,bkc->bko", self.w2, inner) + self.b2[None]).reshape(
            hidden.shape[0], self.k_max * self.output_dim
        )


class LinearBottleneckAtomField(JointSALVelocityField):

    def __init__(
        self, config: JointSALConfig, source_task_count: int, bottleneck: int | None = None
    ) -> None:
        super().__init__(config, source_task_count=source_task_count)
        self.atom_out = LinearBottleneckAtomHead(
            config.hidden_dim, config.k_max, self.output_dim, bottleneck
        )

    @property
    def bottleneck(self) -> int:
        return self.atom_out.bottleneck
