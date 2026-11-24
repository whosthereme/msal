from __future__ import annotations

import torch
from torch import nn

from msal_ext.nl import ReadoutMLP
from joint_sal_velocity import JointSALConfig
from lib.nonlinear_atom_field import NonlinearAtomHead
from lib.structured_field import StructuredCodeFieldV5

ATOM_BOTTLENECK = 46
BASE_WIDTH = 736


class LinearPlusMLP(nn.Module):

    def __init__(self, linear: nn.Linear, width: int) -> None:
        super().__init__()
        self.linear = linear
        self.mlp = ReadoutMLP(linear.in_features, width, linear.out_features)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden) + self.mlp(hidden)


class NonlinearBaseSAL(StructuredCodeFieldV5):
    def __init__(self, config: JointSALConfig, source_task_ids: list[int]) -> None:
        super().__init__(config, source_task_ids, feature_set="saturated_v5")
        self.base_out = LinearPlusMLP(self.base_out, BASE_WIDTH)
        self.atom_out = NonlinearAtomHead(config.hidden_dim, config.k_max, self.output_dim, ATOM_BOTTLENECK)
