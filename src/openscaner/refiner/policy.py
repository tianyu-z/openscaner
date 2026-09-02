"""Frozen local-corner-refiner policy and shared runtime selection."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

from openscaner.geometry import validate_quad
from openscaner.refiner.geometry import (
    HEATMAP_SIZE,
    PATCH_SIZE,
    PatchTransform,
    _validate_ordered_quad,
    decode_local_heatmaps,
)

POLICY_SCHEMA_VERSION = 1


def _unit_float(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0 or converted > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    if positive and converted == 0.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return converted


@dataclass(frozen=True, slots=True)
class RefinerPolicy:
    """The complete, versioned local-corner-refiner acceptance policy."""

    schema_version: int
    radius_ratio: float
    minimum_confidence: float
    maximum_residual_ratio: float
    fallback: Literal["docaligner"]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != POLICY_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be exactly {POLICY_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "radius_ratio",
            _unit_float(self.radius_ratio, "radius_ratio", positive=True),
        )
        object.__setattr__(
            self,
            "minimum_confidence",
            _unit_float(self.minimum_confidence, "minimum_confidence"),
        )
        object.__setattr__(
            self,
            "maximum_residual_ratio",
            _unit_float(
                self.maximum_residual_ratio,
                "maximum_residual_ratio",
            ),
        )
        if self.fallback != "docaligner":
            raise ValueError("fallback must be exactly 'docaligner'")


@dataclass(frozen=True, slots=True)
class RefinerSelection:
    """One full-coverage refinement result or an exact DocAligner fallback."""

    corners: np.ndarray
    fallback_used: bool
    aggregate_confidence: float
    corner_confidences: np.ndarray
    residual_ratios: np.ndarray

    @property
    def refined_corners(self) -> np.ndarray:
        """Compatibility name for consumers that distinguish selected corners."""
        return self.corners


def _source_dimensions(source_shape: object) -> tuple[int, int]:
    if not isinstance(source_shape, tuple) or len(source_shape) not in (2, 3):
        raise TypeError("source_shape must be an H x W or H x W x C tuple")
    height, width = source_shape[:2]
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (height, width)
    ):
        raise TypeError("source_shape height and width must be integers")
    if int(height) < 1 or int(width) < 1:
        raise ValueError("source_shape height and width must be positive")
    return int(height), int(width)


def _basic_coarse_quad(coarse: object) -> np.ndarray:
    if not isinstance(coarse, np.ndarray):
        raise TypeError("coarse must be a numpy array")
    if coarse.shape != (4, 2):
        raise ValueError("coarse must have shape (4, 2)")
    if not (
        np.issubdtype(coarse.dtype, np.integer)
        or np.issubdtype(coarse.dtype, np.floating)
    ):
        raise TypeError("coarse must use a real numeric dtype")
    numeric = np.asarray(coarse, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("coarse must contain only finite values")
    try:
        _validate_ordered_quad(numeric)
    except (TypeError, ValueError) as error:
        raise ValueError("coarse must be a valid TL/TR/BR/BL quadrilateral") from error
    return numeric


def _coarse_quad(coarse: object, source_shape: tuple[int, ...]) -> np.ndarray:
    numeric = _basic_coarse_quad(coarse)
    valid, _ = validate_quad(numeric, source_shape, reorder=False)
    if not valid:
        raise ValueError("coarse must be a valid TL/TR/BR/BL source quadrilateral")
    return numeric


def _validate_transforms(
    transforms: object,
    policy: RefinerPolicy,
) -> tuple[PatchTransform, PatchTransform, PatchTransform, PatchTransform]:
    if not isinstance(transforms, tuple) or len(transforms) != 4:
        raise ValueError("transforms must be an ordered tuple of four PatchTransform values")
    converted: list[PatchTransform] = []
    for index, transform in enumerate(transforms):
        if not isinstance(transform, PatchTransform):
            raise TypeError("transforms must contain PatchTransform values")
        if transform.corner_index != index:
            raise ValueError("transforms must be ordered TL/TR/BR/BL")
        if not np.isclose(
            transform.radius_ratio,
            policy.radius_ratio,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("transform radius_ratio must equal policy radius_ratio")
        converted.append(transform)
    return tuple(converted)  # type: ignore[return-value]


def _model_output(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    unit_interval: bool = False,
) -> np.ndarray | None:
    if not isinstance(value, np.ndarray):
        return None
    if value.shape != shape or value.dtype != np.dtype(np.float32):
        return None
    if not np.isfinite(value).all():
        return None
    if unit_interval and (np.any(value < 0.0) or np.any(value > 1.0)):
        return None
    return np.asarray(value, dtype=np.float64)


def _fallback(coarse: np.ndarray) -> RefinerSelection:
    return RefinerSelection(
        corners=np.array(coarse, copy=True),
        fallback_used=True,
        aggregate_confidence=0.0,
        corner_confidences=np.zeros(4, dtype=np.float64),
        residual_ratios=np.zeros(4, dtype=np.float64),
    )


def _sample_edge_evidence(edge_logits: np.ndarray, point: np.ndarray) -> float:
    """Sample both incident edge heads without allowing either to move a point."""
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("decoded point must be finite x,y")
    scale = (HEATMAP_SIZE - 1) / float(PATCH_SIZE - 1)
    start_x = float(point[0] * scale)
    start_y = float(point[1] * scale)
    x_samples = np.linspace(start_x, float(HEATMAP_SIZE - 1), num=16)
    y_samples = np.linspace(start_y, float(HEATMAP_SIZE - 1), num=16)
    values: list[float] = []
    for channel, coordinates in (
        (0, ((x, start_y) for x in x_samples)),
        (1, ((start_x, y) for y in y_samples)),
    ):
        logits = edge_logits[channel]
        for x, y in coordinates:
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, HEATMAP_SIZE - 1)
            y1 = min(y0 + 1, HEATMAP_SIZE - 1)
            x_weight = x - x0
            y_weight = y - y0
            top = (1.0 - x_weight) * logits[y0, x0] + x_weight * logits[y0, x1]
            bottom = (1.0 - x_weight) * logits[y1, x0] + x_weight * logits[y1, x1]
            logit = (1.0 - y_weight) * top + y_weight * bottom
            if logit >= 0.0:
                probability = 1.0 / (1.0 + np.exp(-logit))
            else:
                exponential = np.exp(logit)
                probability = exponential / (1.0 + exponential)
            values.append(float(probability))
    evidence = float(np.mean(values, dtype=np.float64))
    if not np.isfinite(evidence):
        raise RuntimeError("edge evidence is non-finite")
    return evidence


def apply_refiner_outputs(
    source_shape: tuple[int, ...],
    coarse: np.ndarray,
    transforms: tuple[PatchTransform, PatchTransform, PatchTransform, PatchTransform],
    corner_logits: object,
    edge_logits: object,
    model_confidence: object,
    policy: RefinerPolicy,
) -> RefinerSelection:
    """Select a refined quadrilateral or the exact DocAligner coarse fallback.

    This is intentionally the sole production acceptance gate.  Its static
    inputs are rejected eagerly, while malformed model outputs fail closed to
    a copy-equal DocAligner quadrilateral for complete calibration coverage.
    """
    if not isinstance(policy, RefinerPolicy):
        raise TypeError("policy must be a RefinerPolicy")
    _basic_coarse_quad(coarse)
    try:
        _source_dimensions(source_shape)
    except (TypeError, ValueError):
        return _fallback(coarse)
    coarse_numeric = _coarse_quad(coarse, source_shape)
    try:
        ordered_transforms = _validate_transforms(transforms, policy)
    except (TypeError, ValueError):
        return _fallback(coarse)
    corners = _model_output(
        corner_logits,
        name="corner_logits",
        shape=(4, 1, HEATMAP_SIZE, HEATMAP_SIZE),
    )
    edges = _model_output(
        edge_logits,
        name="edge_logits",
        shape=(4, 2, HEATMAP_SIZE, HEATMAP_SIZE),
    )
    confidence = _model_output(
        model_confidence,
        name="model_confidence",
        shape=(4, 1),
        unit_interval=True,
    )
    if corners is None or edges is None or confidence is None:
        return _fallback(coarse)
    try:
        decoded = decode_local_heatmaps(corners.astype(np.float32))
    except (TypeError, ValueError, RuntimeError, FloatingPointError):
        return _fallback(coarse)
    if (
        decoded.shape != (4, 2)
        or not np.isfinite(decoded).all()
        or np.any(decoded < 0.0)
        or np.any(decoded > float(PATCH_SIZE - 1))
    ):
        return _fallback(coarse)
    try:
        refined = np.stack(
            [
                transform.patch_to_source_points(decoded[index : index + 1])[0]
                for index, transform in enumerate(ordered_transforms)
            ]
        ).astype(np.float64)
    except (TypeError, ValueError, RuntimeError, FloatingPointError):
        return _fallback(coarse)
    height, width = _source_dimensions(source_shape)
    if (
        refined.shape != (4, 2)
        or not np.isfinite(refined).all()
        or np.any(refined[:, 0] < 0.0)
        or np.any(refined[:, 0] >= width)
        or np.any(refined[:, 1] < 0.0)
        or np.any(refined[:, 1] >= height)
    ):
        return _fallback(coarse)
    residuals = np.asarray(
        [
            np.linalg.norm(refined[index] - coarse_numeric[index])
            / transform.local_side
            for index, transform in enumerate(ordered_transforms)
        ],
        dtype=np.float64,
    )
    if (
        not np.isfinite(residuals).all()
        or np.any(residuals > policy.maximum_residual_ratio)
    ):
        return _fallback(coarse)
    valid, _ = validate_quad(refined, source_shape, reorder=False)
    if not valid:
        return _fallback(coarse)
    try:
        edge_evidence = np.asarray(
            [
                _sample_edge_evidence(edges[index], decoded[index])
                for index in range(4)
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError, RuntimeError, FloatingPointError, OverflowError):
        return _fallback(coarse)
    corner_confidences = 0.5 * confidence[:, 0] + 0.5 * edge_evidence
    aggregate_confidence = float(np.mean(corner_confidences, dtype=np.float64))
    if (
        not np.isfinite(corner_confidences).all()
        or not np.isfinite(aggregate_confidence)
        or aggregate_confidence < policy.minimum_confidence
    ):
        return _fallback(coarse)
    return RefinerSelection(
        corners=refined.astype(np.float32),
        fallback_used=False,
        aggregate_confidence=aggregate_confidence,
        corner_confidences=corner_confidences,
        residual_ratios=residuals,
    )


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "RefinerPolicy",
    "RefinerSelection",
    "apply_refiner_outputs",
]
