from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from msal_exp.arms import DescriptorFn, FiLMField, PlainField, _trunk_hidden
from joint_sal_velocity import JointSALConfig, JointSALVelocityField
from lib.nonlinear_atom_field import NonlinearAtomHead
from lib.structured_field import StructuredCodeFieldV5

BOTTLENECK = 92
READOUT_WIDTH = 16 * BOTTLENECK


class NonlinearSAL(StructuredCodeFieldV5):

    def __init__(self, config: JointSALConfig, source_task_ids: list[int]) -> None:
        super().__init__(config, source_task_ids, feature_set="saturated_v5")
        self.atom_out = NonlinearAtomHead(config.hidden_dim, config.k_max, self.output_dim, BOTTLENECK)


class ReadoutMLP(nn.Module):

    def __init__(self, hidden_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        self.inner = nn.Linear(hidden_dim, width)
        self.outer = nn.Linear(width, output_dim)
        nn.init.normal_(self.inner.weight, std=hidden_dim ** -0.5)
        nn.init.zeros_(self.inner.bias)
        nn.init.normal_(self.outer.weight, std=0.1 * width ** -0.5)
        nn.init.zeros_(self.outer.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.outer(F.silu(self.inner(hidden)))


class FiLMNonlinearReadout(FiLMField):
    def __init__(self, config: JointSALConfig, source_task_ids: list[int], descriptor_fn: DescriptorFn) -> None:
        super().__init__(config, source_task_ids, descriptor_fn)
        self.film_mlp = ReadoutMLP(config.hidden_dim, READOUT_WIDTH, config.chunk_length * config.action_dim)

    def forward_descriptor(self, state, noisy_actions, time, descriptor: torch.Tensor) -> torch.Tensor:
        hidden = _trunk_hidden(self, state, noisy_actions, time)
        hidden = hidden * (1.0 + self.film_gamma(descriptor)) + self.film_beta(descriptor)
        out = self.film_head(hidden) + self.film_mlp(hidden)
        return out.reshape(state.shape[0], self.config.chunk_length, self.config.action_dim)


class PlainNonlinearReadout(PlainField):
    def __init__(self, config: JointSALConfig, source_task_ids: list[int]) -> None:
        super().__init__(config, source_task_ids)
        self.plain_mlp = ReadoutMLP(config.hidden_dim, READOUT_WIDTH, config.chunk_length * config.action_dim)

    def forward_descriptor(self, state, noisy_actions, time, descriptor=None) -> torch.Tensor:
        hidden = _trunk_hidden(self, state, noisy_actions, time)
        out = self.plain_head(hidden) + self.plain_mlp(hidden)
        return out.reshape(state.shape[0], self.config.chunk_length, self.config.action_dim)


class DescriptorMoE(JointSALVelocityField):

    def __init__(self, config: JointSALConfig, source_task_ids: list[int], descriptor_fn: DescriptorFn) -> None:
        JointSALVelocityField.__init__(self, config, source_task_count=len(source_task_ids))
        del self.source_codes
        matrix = descriptor_fn(source_task_ids)
        self.descriptor_fn = descriptor_fn
        self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
        self.gate = nn.Linear(int(matrix.shape[1]), config.k_max, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)
        self.atom_out = NonlinearAtomHead(config.hidden_dim, config.k_max, self.output_dim, BOTTLENECK)

    def forward_descriptor(self, state, noisy_actions, time, descriptor: torch.Tensor) -> torch.Tensor:
        hidden = _trunk_hidden(self, state, noisy_actions, time)
        batch, chunk, action = hidden.shape[0], self.config.chunk_length, self.config.action_dim
        base = self.base_out(hidden).reshape(batch, chunk, action)
        experts = self.atom_out(hidden).reshape(batch, self.config.k_max, chunk, action)
        weights = torch.softmax(self.gate(descriptor), dim=-1)
        return base + torch.einsum("bk,bktd->btd", weights, experts)

    def forward_with_rows(self, state, noisy_actions, time, task_rows) -> torch.Tensor:
        return self.forward_descriptor(state, noisy_actions, time, self.task_features[task_rows.long()])
