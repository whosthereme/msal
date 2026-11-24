from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def atom_gram(atom_fields: torch.Tensor) -> torch.Tensor:
    if atom_fields.ndim != 4:
        raise ValueError(f"expected (B, K, T, D) atom fields, got {tuple(atom_fields.shape)}")
    flat = atom_fields.flatten(2).double()
    return torch.einsum("bkd,bld->kl", flat, flat) / flat.shape[0]


def code_covariance(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 2:
        raise ValueError(f"expected (num_tasks, k) codes, got {tuple(codes.shape)}")
    centered = codes.double() - codes.double().mean(dim=0, keepdim=True)
    return centered.T @ centered / centered.shape[0]


def whitened_spectrum(gram: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    k = gram.shape[0]
    jitter = 1e-12 * float(torch.diagonal(gram).mean().item()) * torch.eye(
        k, dtype=torch.float64, device=gram.device
    )
    cholesky = torch.linalg.cholesky(gram.double() + jitter)
    whitened = cholesky.T @ covariance.double() @ cholesky
    whitened = 0.5 * (whitened + whitened.T)
    return torch.linalg.eigvalsh(whitened).flip(0)


def effective_capacity(spectrum: torch.Tensor, energy_fraction: float = 0.99) -> int:
    if not 0.0 < energy_fraction <= 1.0:
        raise ValueError("energy_fraction must lie in (0, 1]")
    values = spectrum.detach().cpu().numpy().astype(np.float64)
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total <= 0.0:
        return 0
    cumulative = np.cumsum(values) / total
    return int(np.searchsorted(cumulative, energy_fraction) + 1)


def capacity_report(
    gram: torch.Tensor,
    covariance: torch.Tensor,
    energy_fractions: tuple[float, ...] = (0.90, 0.95, 0.99, 0.999),
) -> dict[str, Any]:
    spectrum = whitened_spectrum(gram, covariance)
    values = np.clip(spectrum.detach().cpu().numpy().astype(np.float64), 0.0, None)
    total = float(values.sum())
    gain = torch.sqrt(torch.diagonal(gram).clamp_min(0.0)).detach().cpu().numpy()
    spread = torch.sqrt(torch.diagonal(covariance).clamp_min(0.0)).detach().cpu().numpy()
    ratio = gain / np.maximum(spread, 1e-30)
    normalized = gram.detach().cpu().numpy() / np.maximum(np.outer(gain, gain), 1e-30)
    k = gram.shape[0]
    off_diagonal = float(
        ((normalized - np.eye(k)) ** 2).sum() / max(k * (k - 1), 1)
    )
    return {
        "k_max": int(k),
        "whitened_spectrum": values.tolist(),
        "cumulative_output_energy": (np.cumsum(values) / max(total, 1e-30)).tolist(),
        "effective_capacity": {
            f"{fraction:g}": effective_capacity(spectrum, fraction)
            for fraction in energy_fractions
        },
        "atom_output_gain": gain.tolist(),
        "code_spread": spread.tolist(),
        "gain_over_spread": ratio.tolist(),
        "gain_dynamic_range": float(gain.max() / max(gain.min(), 1e-30)),
        "spread_dynamic_range": float(spread.max() / max(spread.min(), 1e-30)),
        "gain_over_spread_condition_number": float(ratio.max() / max(ratio.min(), 1e-30)),
        "normalized_gram_off_diagonality": off_diagonal,
        "per_atom_output_contribution": (gain * spread).tolist(),
    }


@dataclass(frozen=True)
class OutputGauge:

    transform: torch.Tensor
    inverse_transform: torch.Tensor
    spectrum: torch.Tensor
    code_mean: torch.Tensor


def solve_output_gauge(gram: torch.Tensor, codes: torch.Tensor) -> OutputGauge:
    covariance = code_covariance(codes)
    k = gram.shape[0]
    jitter = 1e-12 * float(torch.diagonal(gram).mean().item()) * torch.eye(
        k, dtype=torch.float64, device=gram.device
    )
    cholesky = torch.linalg.cholesky(gram.double() + jitter)
    whitened = cholesky.T @ covariance.double() @ cholesky
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    pivots = eigenvectors.abs().argmax(dim=0)
    signs = torch.sign(eigenvectors[pivots, torch.arange(k, device=eigenvectors.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    eigenvectors = eigenvectors * signs[None, :]
    inverse_transform = cholesky @ eigenvectors
    transform = torch.linalg.inv(inverse_transform)
    return OutputGauge(
        transform=transform,
        inverse_transform=inverse_transform,
        spectrum=eigenvalues,
        code_mean=codes.double().mean(dim=0),
    )


def subspace_alignment(basis_a: torch.Tensor, basis_b: torch.Tensor, rank: int) -> dict[str, float]:
    if rank < 1:
        raise ValueError("rank must be >= 1")
    left = basis_a[:, :rank].double()
    right = basis_b[:, :rank].double()
    left = left / left.norm(dim=0, keepdim=True).clamp_min(1e-30)
    right = right / right.norm(dim=0, keepdim=True).clamp_min(1e-30)
    left, _ = torch.linalg.qr(left)
    right, _ = torch.linalg.qr(right)
    singular = torch.linalg.svdvals(left.T @ right).clamp(0.0, 1.0)
    return {
        "rank": int(rank),
        "min_principal_cosine": float(singular.min().item()),
        "mean_principal_cosine": float(singular.mean().item()),
    }


def per_atom_alignment(basis_a: torch.Tensor, basis_b: torch.Tensor) -> list[float]:
    left = basis_a.double()
    right = basis_b.double()
    left = left / left.norm(dim=0, keepdim=True).clamp_min(1e-30)
    right = right / right.norm(dim=0, keepdim=True).clamp_min(1e-30)
    return [float(abs(torch.dot(left[:, k], right[:, k])).item()) for k in range(left.shape[1])]
