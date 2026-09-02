"""Exact training objective for the local corner refiner."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from openscaner.models.local_corner_refiner import (
    differentiable_local_soft_argmax,
)
from openscaner.training.corner_loss import (
    focal_weighted_mse,
    positive_negative_balanced_bce,
)


_PREDICTION_KEYS = frozenset(("corner_logits", "edge_logits", "confidence"))
_TARGET_KEYS = frozenset(
    ("corner_heatmap", "edge_maps", "corner_xy", "corner_valid")
)


def _require_exact_keys(
    values: Mapping[str, torch.Tensor],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(values) != expected:
        raise ValueError(f"{name} must contain exact keys {sorted(expected)}")


def _require_floating_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    return value


def _validate_inputs(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping")
    if not isinstance(targets, Mapping):
        raise TypeError("targets must be a mapping")
    _require_exact_keys(predictions, _PREDICTION_KEYS, "predictions")
    _require_exact_keys(targets, _TARGET_KEYS, "targets")

    corner_logits = _require_floating_tensor(
        predictions["corner_logits"], "predictions['corner_logits']"
    )
    edge_logits = _require_floating_tensor(
        predictions["edge_logits"], "predictions['edge_logits']"
    )
    confidence = _require_floating_tensor(
        predictions["confidence"], "predictions['confidence']"
    )
    corner_heatmap = _require_floating_tensor(
        targets["corner_heatmap"], "targets['corner_heatmap']"
    )
    edge_maps = _require_floating_tensor(
        targets["edge_maps"], "targets['edge_maps']"
    )
    corner_xy = _require_floating_tensor(
        targets["corner_xy"], "targets['corner_xy']"
    )
    corner_valid = targets["corner_valid"]
    if not isinstance(corner_valid, torch.Tensor):
        raise TypeError("targets['corner_valid'] must be a torch.Tensor")
    if corner_valid.is_complex():
        raise TypeError("targets['corner_valid'] must be bool or numeric")

    named_tensors = {
        "predictions['corner_logits']": corner_logits,
        "predictions['edge_logits']": edge_logits,
        "predictions['confidence']": confidence,
        "targets['corner_heatmap']": corner_heatmap,
        "targets['edge_maps']": edge_maps,
        "targets['corner_xy']": corner_xy,
        "targets['corner_valid']": corner_valid,
    }
    if any(value.device != corner_logits.device for value in named_tensors.values()):
        raise ValueError(
            "all predictions and targets must use the same device as corner_logits"
        )

    if (
        corner_logits.ndim != 4
        or corner_logits.shape[0] < 1
        or corner_logits.shape[1] != 1
        or corner_logits.shape[2] < 3
        or corner_logits.shape[3] < 3
    ):
        raise ValueError("predictions['corner_logits'] has invalid shape")
    batch_size, _, height, width = corner_logits.shape
    if edge_logits.shape != (batch_size, 2, height, width):
        raise ValueError("predictions['edge_logits'] has invalid shape")
    if confidence.shape != (batch_size, 1):
        raise ValueError("predictions['confidence'] has invalid shape")
    if corner_heatmap.shape != corner_logits.shape:
        raise ValueError("targets['corner_heatmap'] has invalid shape")
    if edge_maps.shape != edge_logits.shape:
        raise ValueError("targets['edge_maps'] has invalid shape")
    if corner_xy.shape != (batch_size, 2):
        raise ValueError("targets['corner_xy'] has invalid shape")
    if corner_valid.shape != (batch_size,):
        raise ValueError("targets['corner_valid'] has invalid shape")

    for name, value in named_tensors.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    for name, value in (
        ("targets['corner_heatmap']", corner_heatmap),
        ("targets['edge_maps']", edge_maps),
        ("predictions['confidence']", confidence),
    ):
        if torch.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{name} must contain values in [0,1]")
    if not torch.all((corner_valid == 0) | (corner_valid == 1)):
        raise ValueError("targets['corner_valid'] must be a binary mask")
    valid_coordinates = corner_xy[corner_valid.bool()]
    if valid_coordinates.numel() > 0 and torch.any(
        (valid_coordinates < 0.0) | (valid_coordinates > 1.0)
    ):
        raise ValueError(
            "targets['corner_xy'] values marked valid must be in [0,1]"
        )


def local_corner_refiner_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the weighted refiner objective and all scalar components."""
    _validate_inputs(predictions, targets)

    corner_logits = predictions["corner_logits"]
    corner_targets = targets["corner_heatmap"]
    corner_focal_mse = focal_weighted_mse(corner_logits, corner_targets)
    corner_balanced_bce = positive_negative_balanced_bce(
        corner_logits, corner_targets
    )
    corner_heatmaps = corner_focal_mse + corner_balanced_bce

    decoded = differentiable_local_soft_argmax(corner_logits).float()
    coordinate_targets = targets["corner_xy"].float()
    per_sample_residual = F.smooth_l1_loss(
        decoded, coordinate_targets, reduction="none"
    ).mean(dim=1)
    valid = targets["corner_valid"].ne(0).to(dtype=per_sample_residual.dtype)
    unweighted_residual = (per_sample_residual * valid).sum() / valid.sum().clamp_min(
        1.0
    )
    residual = 0.5 * unweighted_residual

    edge_logits = predictions["edge_logits"]
    edge_targets = targets["edge_maps"]
    edge_bce = positive_negative_balanced_bce(edge_logits, edge_targets)
    edge_probabilities = torch.sigmoid(edge_logits.float())
    float_edge_targets = edge_targets.float()
    intersection = (edge_probabilities * float_edge_targets).sum(dim=(2, 3))
    denominator = edge_probabilities.sum(dim=(2, 3)) + float_edge_targets.sum(
        dim=(2, 3)
    )
    edge_dice = 1.0 - (
        (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean()
    edges = 0.25 * 0.5 * (edge_bce + edge_dice)

    total = corner_heatmaps + residual + edges
    components = {
        "total": total,
        "corner_heatmaps": corner_heatmaps,
        "corner_focal_mse": corner_focal_mse,
        "corner_balanced_bce": corner_balanced_bce,
        "residual": residual,
        "edges": edges,
        "edge_bce": edge_bce,
        "edge_dice": edge_dice,
    }
    if any(
        value.ndim != 0 or not torch.isfinite(value)
        for value in components.values()
    ):
        raise RuntimeError("local corner refiner loss produced a non-finite scalar")
    return components


__all__ = ["local_corner_refiner_loss"]
