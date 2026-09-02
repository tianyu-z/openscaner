"""Losses for four-corner heatmaps with auxiliary document masks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


def focal_weighted_mse(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return channel-balanced MSE weighted toward uncertain/wrong pixels."""
    logits = logits.float()
    targets = targets.float()
    probabilities = torch.sigmoid(logits)
    focal_weights = (
        targets * (1.0 - probabilities).square()
        + (1.0 - targets) * probabilities.square()
    )
    per_channel = (focal_weights * (probabilities - targets).square()).mean(
        dim=(0, 2, 3)
    )
    return per_channel.mean()


def positive_negative_balanced_bce(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Balance stable positive and negative BCE per sample and corner channel."""
    logits = logits.float()
    targets = targets.float()
    dimensions = (2, 3)
    positive_mass = targets.sum(dim=dimensions).clamp_min(1.0)
    negative_targets = 1.0 - targets
    negative_mass = negative_targets.sum(dim=dimensions).clamp_min(1.0)
    positive = (targets * F.softplus(-logits)).sum(dim=dimensions) / positive_mass
    negative = (
        negative_targets * F.softplus(logits)
    ).sum(dim=dimensions) / negative_mass
    return (0.5 * (positive + negative)).mean()


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return batch-mean soft Dice loss for mask logits."""
    probabilities = torch.sigmoid(logits.float())
    targets = targets.float()
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def corner_heatmap_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the total corner objective and its finite scalar components."""
    corner_focal_mse = focal_weighted_mse(
        predictions["corner_heatmaps"], targets["heatmaps"]
    )
    corner_balanced_bce = positive_negative_balanced_bce(
        predictions["corner_heatmaps"], targets["heatmaps"]
    )
    heatmap_loss = corner_focal_mse + corner_balanced_bce
    mask_bce = F.binary_cross_entropy_with_logits(
        predictions["mask_logits"].float(), targets["mask"].float()
    )
    mask_dice = soft_dice_loss(predictions["mask_logits"], targets["mask"])
    mask_loss = 0.25 * (mask_bce + mask_dice)
    return {
        "total": heatmap_loss + mask_loss,
        "corner_heatmaps": heatmap_loss,
        "corner_focal_mse": corner_focal_mse,
        "corner_balanced_bce": corner_balanced_bce,
        "mask": mask_loss,
        "mask_bce": mask_bce,
        "mask_dice": mask_dice,
    }
