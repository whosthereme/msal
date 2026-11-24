from __future__ import annotations

import torch
from torch import nn

from msal_exp.arms import DescriptorFn, _trunk_hidden
from joint_sal_velocity import JointSALConfig, JointSALVelocityField


class TiedAtomSAL(JointSALVelocityField):
    def __init__(self, config: JointSALConfig, source_task_ids: list[int], descriptor_fn: DescriptorFn) -> None:
        JointSALVelocityField.__init__(self, config, source_task_count=len(source_task_ids))
        del self.atom_out
        matrix = descriptor_fn(source_task_ids)
        self.descriptor_fn = descriptor_fn
        self.source_task_ids = [int(t) for t in source_task_ids]
        self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
        self.code_map = nn.Linear(int(matrix.shape[1]), config.k_max, bias=False)
        nn.init.normal_(self.code_map.weight, std=0.02)
        self.atom_scale = nn.Parameter(torch.randn(config.k_max, config.hidden_dim) * 0.02)
        self.atom_shift = nn.Parameter(torch.randn(config.k_max, config.hidden_dim) * 0.02)
        with torch.no_grad():
            self.source_codes.weight.zero_()
        self.source_codes.weight.requires_grad_(False)

    @property
    def structured_codes(self) -> torch.Tensor:
        return self.code_map(self.task_features)

    def codes_for_rows(self, task_rows: torch.Tensor) -> torch.Tensor:
        return self.code_map(self.task_features[task_rows])

    def codes_for_task_ids(self, task_ids: list[int]) -> torch.Tensor:
        matrix = self.descriptor_fn(task_ids)
        features = torch.tensor(matrix, dtype=self.task_features.dtype, device=self.task_features.device)
        return self.code_map(features)

    def components(self, state_features, noisy_actions, time, phase_indices=None):
        if phase_indices is not None:
            raise ValueError("TiedAtomSAL has no phase conditioning")
        hidden = _trunk_hidden(self, state_features, noisy_actions, time)
        batch = hidden.shape[0]
        chunk, action = self.config.chunk_length, self.config.action_dim
        base = self.base_out(hidden).reshape(batch, chunk, action)
        modulated = hidden[:, None, :] * self.atom_scale[None] + self.atom_shift[None]
        atoms = torch.einsum("bkh,oh->bko", modulated, self.base_out.weight)
        return base, atoms.reshape(batch, self.config.k_max, chunk, action)
