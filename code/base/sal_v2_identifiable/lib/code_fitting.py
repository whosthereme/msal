from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CodeFit:

    code: torch.Tensor
    residual_mse: float
    baseline_mse: float
    explained_fraction: float
    per_atom_contribution: np.ndarray
    effective_rank: int
    condition_number: float

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code.detach().cpu().double().tolist(),
            "residual_mse": self.residual_mse,
            "baseline_mse": self.baseline_mse,
            "explained_fraction": self.explained_fraction,
            "per_atom_contribution": self.per_atom_contribution.tolist(),
            "effective_rank": int(self.effective_rank),
            "condition_number": self.condition_number,
        }


def solve_code(
    base: torch.Tensor,
    atoms: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    ridge: float = 1e-6,
    energy_fraction: float = 0.99,
) -> CodeFit:
    if atoms.ndim != 4:
        raise ValueError(f"expected (B, K, T, D) atoms, got {tuple(atoms.shape)}")
    if base.shape != target.shape:
        raise ValueError("base and target must have the same shape")
    if mask.shape != base.shape[:2]:
        raise ValueError("mask must be (batch, chunk)")

    weight = mask.double().unsqueeze(-1)
    residual = (target.double() - base.double()) * weight
    design = atoms.double() * weight.unsqueeze(1)

    flat_residual = residual.flatten(1)
    flat_design = design.permute(0, 2, 3, 1).flatten(1, 2)

    normal = torch.einsum("bdk,bdl->kl", flat_design, flat_design)
    rhs = torch.einsum("bdk,bd->k", flat_design, flat_residual)
    k = normal.shape[0]
    scale = float(torch.diagonal(normal).mean().clamp_min(1e-30).item())
    regularized = normal + ridge * scale * torch.eye(k, dtype=torch.float64, device=normal.device)
    code = torch.linalg.solve(regularized, rhs)

    valid = mask.double().sum() * base.shape[-1]
    baseline_mse = float((residual**2).sum() / valid.clamp_min(1.0))
    prediction = torch.einsum("bdk,k->bd", flat_design, code)
    residual_mse = float(((flat_residual - prediction) ** 2).sum() / valid.clamp_min(1.0))

    contribution = (code.cpu().numpy() ** 2) * torch.diagonal(normal).cpu().numpy()
    order = np.argsort(-contribution)
    total = contribution.sum()
    if total > 0:
        cumulative = np.cumsum(contribution[order]) / total
        effective_rank = int(np.searchsorted(cumulative, energy_fraction) + 1)
    else:
        effective_rank = 0
    eigenvalues = torch.linalg.eigvalsh(normal).clamp_min(0.0).cpu().numpy()
    condition = float(eigenvalues.max() / max(eigenvalues.min(), 1e-30))

    return CodeFit(
        code=code,
        residual_mse=residual_mse,
        baseline_mse=baseline_mse,
        explained_fraction=float(1.0 - residual_mse / max(baseline_mse, 1e-30)),
        per_atom_contribution=contribution,
        effective_rank=effective_rank,
        condition_number=condition,
    )


def admission_decision(
    *,
    candidate_residual: float,
    reference_residuals: list[float],
    quantile: float = 0.95,
    tolerance: float = 1.10,
) -> dict[str, Any]:
    if not reference_residuals:
        raise ValueError("at least one reference residual is required")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    if tolerance < 1.0:
        raise ValueError("tolerance must be >= 1")
    threshold = float(np.quantile(np.asarray(reference_residuals, dtype=np.float64), quantile))
    bound = threshold * tolerance
    admitted = candidate_residual <= bound
    return {
        "candidate_residual": float(candidate_residual),
        "reference_quantile": quantile,
        "reference_threshold": threshold,
        "admission_bound": bound,
        "code_only_admitted": bool(admitted),
        "atom_birth_considered": bool(not admitted),
        "excess_ratio": float(candidate_residual / max(bound, 1e-30)),
    }


def atom_birth_direction(
    base: torch.Tensor,
    atoms: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    code: torch.Tensor,
) -> dict[str, Any]:
    weight = mask.double().unsqueeze(-1)
    residual = (target.double() - base.double()) * weight
    design = atoms.double() * weight.unsqueeze(1)
    fitted = torch.einsum("bktd,k->btd", design, code.double())
    post = (residual - fitted).flatten(1)

    atom_flat = design.flatten(2)
    gram = torch.einsum("bkd,bld->kl", atom_flat, atom_flat)
    cross = torch.einsum("bkd,bd->k", atom_flat, post)
    k = gram.shape[0]
    scale = float(torch.diagonal(gram).mean().clamp_min(1e-30).item())
    coefficients = torch.linalg.solve(
        gram + 1e-9 * scale * torch.eye(k, dtype=torch.float64, device=gram.device), cross
    )
    orthogonal = post - torch.einsum("bkd,k->bd", atom_flat, coefficients)

    norm = float(orthogonal.norm().item())
    residual_norm = float(post.norm().item())
    signal_norm = float((target.double() * weight).norm().item())
    negligible = residual_norm <= 1e-8 * max(signal_norm, 1e-30)
    fraction = 0.0 if negligible else norm / max(residual_norm, 1e-30)
    return {
        "post_fit_residual_norm": residual_norm,
        "output_orthogonal_residual_norm": norm,
        "signal_norm": signal_norm,
        "residual_is_negligible": bool(negligible),
        "orthogonal_fraction": float(fraction),
        "duplicate_of_existing_span": bool(negligible or fraction < 0.10),
    }
