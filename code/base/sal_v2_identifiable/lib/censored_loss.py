from __future__ import annotations

import torch

ACTION_BOUND = 1.0

BOUND_TOLERANCE = 1e-6


def saturation_mask(actions: torch.Tensor) -> torch.Tensor:
    return (actions.abs() - ACTION_BOUND).abs() < BOUND_TOLERANCE


def implied_action(x_t: torch.Tensor, time: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    return x_t - time[:, None, None] * velocity


def censored_action_residual(
    predicted_action: torch.Tensor,
    observed_action: torch.Tensor,
) -> torch.Tensor:
    residual = predicted_action - observed_action
    saturated = saturation_mask(observed_action)
    outward = torch.sign(observed_action)
    inward_only = torch.clamp(residual * outward, max=0.0) * outward
    return torch.where(saturated, inward_only, residual)


def masked_mean(values: torch.Tensor, masks: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weight = masks[:, :, None].to(values.dtype) * weights.view(1, 1, -1)
    total = weight.sum()
    if float(total) == 0.0:
        return values.sum() * 0.0
    return (values * weight).sum() / total


def censored_flow_loss(
    prediction: torch.Tensor,
    batch_x_t: torch.Tensor,
    batch_time: torch.Tensor,
    observed_action: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    predicted_action = implied_action(batch_x_t, batch_time, prediction)
    residual = censored_action_residual(predicted_action, observed_action)
    return masked_mean(residual**2, masks, weights)


def unsaturated_only_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_action: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    keep = (~saturation_mask(observed_action)).to(prediction.dtype)
    weight = masks[:, :, None].to(prediction.dtype) * weights.view(1, 1, -1) * keep
    total = weight.sum()
    if float(total) == 0.0:
        return prediction.sum() * 0.0
    return ((prediction - target) ** 2 * weight).sum() / total


def unsaturated_action_mse(
    prediction: torch.Tensor,
    batch_x_t: torch.Tensor,
    batch_time: torch.Tensor,
    observed_action: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_action = implied_action(batch_x_t, batch_time, prediction)
    keep = (~saturation_mask(observed_action)).to(prediction.dtype)
    weight = masks[:, :, None].to(prediction.dtype) * weights.view(1, 1, -1) * keep
    total = weight.sum()
    squared = (predicted_action - observed_action) ** 2
    if float(total) == 0.0:
        return squared.sum() * 0.0, total
    return (squared * weight).sum() / total, total


def post_clip_action_mse(
    prediction: torch.Tensor,
    batch_x_t: torch.Tensor,
    batch_time: torch.Tensor,
    observed_action: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    predicted_action = implied_action(batch_x_t, batch_time, prediction)
    clipped = predicted_action.clamp(-ACTION_BOUND, ACTION_BOUND)
    return masked_mean((clipped - observed_action) ** 2, masks, weights)
