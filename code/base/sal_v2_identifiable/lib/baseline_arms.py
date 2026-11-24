from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PARENT_ROOT = Path(__file__).resolve().parents[2]
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

from joint_sal_velocity import JointSALConfig, JointSALVelocityField

BASELINE_ARMS = ("moe_soft_gate", "moe_hard_gate", "full_per_task", "single_shared")

GATED_ARMS = ("moe_soft_gate", "moe_hard_gate")

TASK_GATE_EMBED_DIM = 16


class BaselineHeadField(nn.Module):

    def __init__(
        self,
        config: JointSALConfig,
        source_task_count: int,
        *,
        arm: str,
        expert_count: int = 1,
        task_gate: bool = False,
    ) -> None:
        super().__init__()
        if arm not in BASELINE_ARMS:
            raise ValueError(f"unknown arm {arm!r}; choose from {BASELINE_ARMS}")
        if arm in GATED_ARMS and expert_count < 2:
            raise ValueError(f"{arm} needs expert_count >= 2, got {expert_count}")
        if source_task_count < 1:
            raise ValueError("source_task_count must be >= 1")
        self.config = config
        self.arm = arm
        self.source_task_count = int(source_task_count)
        self.task_gate = task_gate

        self.field = JointSALVelocityField(config, source_task_count=source_task_count)
        del self.field.atom_out
        del self.field.source_codes
        del self.field.base_out

        output_dim = config.chunk_length * config.action_dim
        if arm == "single_shared":
            self.head_count = 1
        elif arm == "full_per_task":
            self.head_count = self.source_task_count
        else:
            self.head_count = int(expert_count)

        self.head_weight = nn.Parameter(torch.zeros(self.head_count, config.hidden_dim, output_dim))
        self.head_bias = nn.Parameter(torch.zeros(self.head_count, output_dim))
        for index in range(self.head_count):
            nn.init.normal_(self.head_weight[index], std=0.002)

        if arm in GATED_ARMS:
            router_in = config.hidden_dim
            if task_gate:
                self.task_embed = nn.Embedding(self.source_task_count, TASK_GATE_EMBED_DIM)
                nn.init.normal_(self.task_embed.weight, std=0.02)
                router_in += TASK_GATE_EMBED_DIM
            else:
                self.task_embed = None
            self.router = nn.Sequential(
                nn.Linear(router_in, 64),
                nn.SiLU(),
                nn.Linear(64, self.head_count),
            )
        else:
            self.router = None
            self.task_embed = None


    def _trunk_hidden(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        from joint_sal_velocity import sinusoidal_time_embedding

        field = self.field
        batch_size = state_features.shape[0]
        state = (state_features.float() - field.state_mean) / field.state_std
        state_hidden = field.state_encoder(state)
        action_hidden = field.action_encoder(noisy_actions.reshape(batch_size, -1).float())
        time_hidden = sinusoidal_time_embedding(time.float(), self.config.time_dim)
        return field.trunk(torch.cat([state_hidden, action_hidden, time_hidden], dim=-1))

    def routing_weights(
        self, hidden: torch.Tensor, task_rows: torch.Tensor
    ) -> torch.Tensor:
        if self.arm == "single_shared":
            return torch.ones(hidden.shape[0], 1, device=hidden.device)
        if self.arm == "full_per_task":
            return torch.nn.functional.one_hot(
                task_rows.long().clamp(0, self.head_count - 1), num_classes=self.head_count
            ).to(hidden.dtype)
        router_input = hidden
        if self.task_embed is not None:
            embedded = self.task_embed(
                task_rows.long().clamp(0, self.source_task_count - 1)
            ).to(hidden.dtype)
            router_input = torch.cat([hidden, embedded], dim=-1)
        logits = self.router(router_input)
        soft = torch.softmax(logits, dim=-1)
        if self.arm == "moe_soft_gate":
            return soft
        top = soft.argmax(dim=-1)
        hard = torch.nn.functional.one_hot(top, num_classes=self.head_count).to(soft.dtype)
        return hard + soft - soft.detach()


    def forward_with_rows(
        self,
        state_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        task_rows: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state_features.shape[0]
        hidden = self._trunk_hidden(state_features, noisy_actions, time)
        weights = self.routing_weights(hidden, task_rows)
        per_head = torch.einsum("bd,hdo->bho", hidden, self.head_weight) + self.head_bias[None]
        mixed = torch.einsum("bh,bho->bo", weights, per_head)
        return mixed.reshape(batch_size, self.config.chunk_length, self.config.action_dim)


    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def task_conditioned_parameter_count(self) -> int:
        if self.arm == "single_shared":
            return 0
        total = self.head_weight.numel() + self.head_bias.numel()
        if self.router is not None:
            total += sum(p.numel() for p in self.router.parameters())
        if self.task_embed is not None:
            total += sum(p.numel() for p in self.task_embed.parameters())
        return int(total)

    def shared_trunk_parameter_count(self) -> int:
        return int(self.trainable_parameter_count() - self.task_conditioned_parameter_count())


DEAD_EXPERT_LOAD = 0.01

MIN_EFFECTIVE_EXPERTS = 2.0


def gate_utilization(
    weights: torch.Tensor, task_rows: torch.Tensor | None = None
) -> dict[str, float | list[float]]:
    if weights.ndim != 2:
        raise ValueError(f"weights must be [batch, experts], got shape {tuple(weights.shape)}")
    with torch.no_grad():
        probs = weights.detach().float()
        expert_count = probs.shape[1]
        log_e = torch.log(torch.tensor(float(max(expert_count, 2))))

        share = probs.mean(dim=0)
        share = share / share.sum().clamp_min(1e-12)
        safe_share = share.clamp_min(1e-12)
        marginal_entropy = float((-(safe_share * safe_share.log()).sum() / log_e).item())
        effective_experts = float(torch.exp(-(safe_share * safe_share.log()).sum()).item())
        live = int((share >= DEAD_EXPERT_LOAD / max(expert_count, 1)).sum().item())

        safe = probs.clamp_min(1e-12)
        row_entropy = float((-(safe * safe.log()).sum(dim=-1).mean() / log_e).item())

        result: dict[str, float | list[float]] = {
            "expert_count": int(expert_count),
            "live_expert_count": live,
            "effective_expert_count": effective_experts,
            "marginal_gate_entropy": marginal_entropy,
            "per_row_gate_entropy": row_entropy,
            "usage_shares": [float(x) for x in share.tolist()],
            "max_usage_share": float(share.max().item()),
            "min_usage_share": float(share.min().item()),
            "collapsed": bool(effective_experts < MIN_EFFECTIVE_EXPERTS),
        }

        if task_rows is not None and expert_count > 1:
            rows = task_rows.detach().long().reshape(-1)
            present = torch.unique(rows)
            if present.numel() >= 2:
                means = torch.stack([probs[rows == t].mean(dim=0) for t in present])
                grand = probs.mean(dim=0)
                between = ((means - grand[None]) ** 2).mean()
                total = ((probs - grand[None]) ** 2).mean()
                ratio = float((between / total.clamp_min(1e-12)).item()) if total > 0 else 0.0
                result["task_discriminability"] = min(max(ratio, 0.0), 1.0)
                result["tasks_observed"] = int(present.numel())
    return result


def sal_parameter_split(model: nn.Module, *, k_max: int, source_task_count: int) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    atom = sum(p.numel() for p in model.atom_out.parameters())
    codes = k_max * source_task_count
    return {
        "trainable_parameters": int(total),
        "task_conditioned_parameters": int(atom + codes),
        "shared_trunk_parameters": int(total - atom - codes),
    }
