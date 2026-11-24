from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from joint_sal_velocity import JointSALConfig, JointSALVelocityField, sinusoidal_time_embedding
from lib.context_cond import ContextConditionedField, build_context_features
from lib.pool_v5 import build_pool
from lib.structured_field import StructuredCodeFieldV5
from lib.task_algebra_v5 import design_matrix_for_tasks

DescriptorFn = Callable[[Sequence[int]], np.ndarray]

ARMS: dict[str, dict] = {
    "ctx_add": {"family": "context", "descriptor": "saturated", "reg": False},
    "film": {"family": "film", "descriptor": "saturated", "reg": False},
    "bilinear": {"family": "sal_identity", "descriptor": "saturated", "reg": False},
    "sal": {"family": "sal_v5", "descriptor": "saturated", "reg": True},
    "sal_noreg": {"family": "sal_v5", "descriptor": "saturated", "reg": False},
    "plain": {"family": "plain", "descriptor": None, "reg": False},
    "sal_word": {"family": "sal_desc", "descriptor": "word", "reg": True},
}

CODE_AFFINE_FAMILIES = {"sal_v5", "sal_identity", "sal_desc"}


def saturated_descriptor(task_ids: Sequence[int]) -> np.ndarray:
    matrix, _ = design_matrix_for_tasks([int(t) for t in task_ids], "saturated_v5")
    return np.asarray(matrix, dtype=np.float64)


def word_descriptor(task_ids: Sequence[int]) -> np.ndarray:
    pool = {t.task_id: t for t in build_pool()}
    rows = []
    for task_id in task_ids:
        task = pool[int(task_id)]
        row = np.zeros(16 + 4, dtype=np.float64)
        row[task.direction_16] = 1.0
        row[16 + task.radius_index] = 1.0
        rows.append(row)
    return np.stack(rows)


DESCRIPTORS: dict[str, DescriptorFn] = {"saturated": saturated_descriptor, "word": word_descriptor}


def _trunk_hidden(model: JointSALVelocityField, state, noisy_actions, time) -> torch.Tensor:
    batch_size = state.shape[0]
    normalized = (state.float() - model.state_mean) / model.state_std
    state_hidden = model.state_encoder(normalized)
    action_hidden = model.action_encoder(noisy_actions.reshape(batch_size, -1).float())
    time_hidden = sinusoidal_time_embedding(time.float(), model.config.time_dim)
    return model.trunk(torch.cat([state_hidden, action_hidden, time_hidden], dim=-1))


class DescriptorSAL(JointSALVelocityField):

    def __init__(
        self,
        config: JointSALConfig,
        source_task_ids: list[int],
        descriptor_fn: DescriptorFn,
        *,
        fixed_identity: bool = False,
    ) -> None:
        JointSALVelocityField.__init__(self, config, source_task_count=len(source_task_ids))
        matrix = descriptor_fn(source_task_ids)
        self.descriptor_fn = descriptor_fn
        self.source_task_ids = [int(t) for t in source_task_ids]
        self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
        width = int(matrix.shape[1])
        self.code_map = nn.Linear(width, config.k_max, bias=False)
        nn.init.normal_(self.code_map.weight, std=0.02)
        if fixed_identity:
            if width != config.k_max:
                raise ValueError(f"identity code map needs K == F, got K={config.k_max}, F={width}")
            with torch.no_grad():
                self.code_map.weight.copy_(torch.eye(width))
            self.code_map.weight.requires_grad_(False)
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


class FiLMField(JointSALVelocityField):

    def __init__(self, config: JointSALConfig, source_task_ids: list[int], descriptor_fn: DescriptorFn):
        JointSALVelocityField.__init__(self, config, source_task_count=len(source_task_ids))
        del self.atom_out
        del self.source_codes
        del self.base_out
        matrix = descriptor_fn(source_task_ids)
        self.descriptor_fn = descriptor_fn
        self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
        width = int(matrix.shape[1])
        self.film_gamma = nn.Linear(width, config.hidden_dim, bias=False)
        self.film_beta = nn.Linear(width, config.hidden_dim, bias=False)
        nn.init.normal_(self.film_gamma.weight, std=0.02)
        nn.init.normal_(self.film_beta.weight, std=0.02)
        self.film_head = nn.Linear(config.hidden_dim, config.chunk_length * config.action_dim)

    def forward_descriptor(self, state, noisy_actions, time, descriptor: torch.Tensor) -> torch.Tensor:
        hidden = _trunk_hidden(self, state, noisy_actions, time)
        hidden = hidden * (1.0 + self.film_gamma(descriptor)) + self.film_beta(descriptor)
        return self.film_head(hidden).reshape(state.shape[0], self.config.chunk_length, self.config.action_dim)

    def forward_with_rows(self, state, noisy_actions, time, task_rows) -> torch.Tensor:
        return self.forward_descriptor(state, noisy_actions, time, self.task_features[task_rows.long()])


class PlainField(JointSALVelocityField):

    def __init__(self, config: JointSALConfig, source_task_ids: list[int]):
        JointSALVelocityField.__init__(self, config, source_task_count=len(source_task_ids))
        del self.atom_out
        del self.source_codes
        del self.base_out
        self.plain_head = nn.Linear(config.hidden_dim, config.chunk_length * config.action_dim)

    def forward_descriptor(self, state, noisy_actions, time, descriptor=None) -> torch.Tensor:
        hidden = _trunk_hidden(self, state, noisy_actions, time)
        return self.plain_head(hidden).reshape(state.shape[0], self.config.chunk_length, self.config.action_dim)

    def forward_with_rows(self, state, noisy_actions, time, task_rows) -> torch.Tensor:
        return self.forward_descriptor(state, noisy_actions, time)


def build_model(arm: str, config: JointSALConfig, fit_ids: list[int]) -> nn.Module:
    spec = ARMS[arm]
    family = spec["family"]
    if family == "sal_v5":
        return StructuredCodeFieldV5(config, fit_ids, feature_set="saturated_v5")
    if family == "context":
        features = build_context_features(
            fit_ids, arm="context_phi", feature_set="saturated_v5", pool="v5", radius_count=4
        )
        return ContextConditionedField(config, len(fit_ids), arm="context_phi", task_features=features)
    if family == "sal_identity":
        return DescriptorSAL(config, fit_ids, DESCRIPTORS[spec["descriptor"]], fixed_identity=True)
    if family == "sal_desc":
        return DescriptorSAL(config, fit_ids, DESCRIPTORS[spec["descriptor"]])
    if family == "film":
        return FiLMField(config, fit_ids, DESCRIPTORS[spec["descriptor"]])
    if family == "plain":
        return PlainField(config, fit_ids)
    raise ValueError(f"unknown family {family!r}")


def config_for(arm: str, data) -> JointSALConfig:
    k_max = 21 if ARMS[arm]["family"] == "sal_identity" else 16
    return JointSALConfig(
        state_dim=data.state_dim,
        chunk_length=int(data.action_chunks.shape[1]),
        action_dim=int(data.action_chunks.shape[2]),
        k_max=k_max,
    )


def parameter_report(model: nn.Module, arm: str) -> dict[str, int]:
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    total = int(sum(p.numel() for p in trainable.values()))
    family = ARMS[arm]["family"]
    if family in CODE_AFFINE_FAMILIES:
        conditioned = sum(p.numel() for n, p in trainable.items() if n.startswith(("atom_out", "code_map")))
    elif family == "film":
        conditioned = sum(p.numel() for n, p in trainable.items() if n.startswith(("film_gamma", "film_beta")))
    elif family == "context":
        descriptor_width = int(model.task_features.shape[1])
        conditioned = descriptor_width * int(model.context_head.weight.shape[0])
    else:
        conditioned = 0
    return {"trainable_parameters": total, "task_conditioned_parameters": int(conditioned)}
