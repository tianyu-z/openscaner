"""Deterministic, bounded candidate generation for document-boundary fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from openscaner.fusion.signals import CornerModelSignals, DocAlignerSignals
from openscaner.postprocess import quad_from_mask


CandidateFamily = Literal["docaligner", "corner_heatmap", "mask", "hybrid"]

_CANDIDATE_FAMILIES = frozenset(
    {"docaligner", "corner_heatmap", "mask", "hybrid"}
)
_MAXIMUM_CANDIDATES = 64
_MINIMUM_AREA_RATIO = 0.01
_MASK_PROBABILITY_THRESHOLDS = (
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.975,
)
_MAXIMUM_COMPONENTS_PER_THRESHOLD = 2
_DUPLICATE_DECIMALS = 6


def _has_model_order_geometry(corners: np.ndarray) -> bool:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return False
    edges = np.roll(corners, -1, axis=0) - corners
    cross_products = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    scale = max(1.0, float(np.max(np.abs(corners))))
    epsilon = np.finfo(np.float64).eps * scale * scale * 32.0
    if np.any(cross_products <= epsilon):
        return False

    tl, tr, br, bl = corners
    center = corners.mean(axis=0)
    semantic_epsilon = np.finfo(np.float64).eps * scale * 32.0
    return bool(
        0.5 * (tl[1] + tr[1]) < center[1] - semantic_epsilon
        and 0.5 * (bl[1] + br[1]) > center[1] + semantic_epsilon
        and 0.5 * (tl[0] + bl[0]) < center[0] - semantic_epsilon
        and 0.5 * (tr[0] + br[0]) > center[0] + semantic_epsilon
    )


@dataclass(frozen=True)
class FusionCandidate:
    family: CandidateFamily
    corners: np.ndarray
    source_index: int

    def __post_init__(self) -> None:
        if self.family not in _CANDIDATE_FAMILIES:
            raise ValueError("family is not a supported fusion candidate family")
        if isinstance(self.source_index, bool) or not isinstance(self.source_index, int):
            raise TypeError("source_index must be a non-negative integer")
        if self.source_index < 0:
            raise ValueError("source_index must be a non-negative integer")

        source = np.asarray(self.corners)
        if source.shape != (4, 2):
            raise ValueError("corners must have shape (4, 2)")
        if not np.issubdtype(source.dtype, np.number):
            raise TypeError("corners must be numeric")
        with np.errstate(over="ignore", invalid="ignore"):
            corners = np.array(source, dtype=np.float32, copy=True)
        if not _has_model_order_geometry(corners.astype(np.float64)):
            raise ValueError(
                "corners must be finite, convex, and ordered TL, TR, BR, BL"
            )
        immutable_corners = np.frombuffer(
            corners.tobytes(order="C"),
            dtype=np.float32,
        ).reshape(4, 2)
        object.__setattr__(self, "corners", immutable_corners)


def _validate_generation_limits(
    maximum_candidates: int,
    maximum_corner_displacement_ratio: float,
) -> tuple[int, float]:
    if isinstance(maximum_candidates, bool) or not isinstance(maximum_candidates, int):
        raise TypeError("maximum_candidates must be an integer")
    if not 1 <= maximum_candidates <= _MAXIMUM_CANDIDATES:
        raise ValueError(
            f"maximum_candidates must be in the range [1, {_MAXIMUM_CANDIDATES}]"
        )
    if isinstance(maximum_corner_displacement_ratio, bool) or not isinstance(
        maximum_corner_displacement_ratio, (int, float, np.integer, np.floating)
    ):
        raise TypeError("maximum_corner_displacement_ratio must be numeric")
    displacement = float(maximum_corner_displacement_ratio)
    if not np.isfinite(displacement) or not 0.0 <= displacement <= 1.0:
        raise ValueError(
            "maximum_corner_displacement_ratio must be finite and in [0, 1]"
        )
    return maximum_candidates, displacement


def _image_size(image: object) -> tuple[int, int]:
    source = np.asarray(image)
    if source.ndim not in (2, 3) or source.size == 0:
        raise ValueError("image must be a non-empty image array")
    height, width = source.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("image dimensions must each be at least two pixels")
    return height, width


def _valid_source_corners(
    corners: object,
    *,
    height: int,
    width: int,
) -> np.ndarray | None:
    try:
        source = np.asarray(corners)
    except Exception:
        return None
    if source.shape != (4, 2) or not np.issubdtype(source.dtype, np.number):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        quad = np.array(source, dtype=np.float32, copy=True)
    if not _has_model_order_geometry(quad.astype(np.float64)):
        return None
    if (
        np.any(quad[:, 0] < 0.0)
        or np.any(quad[:, 0] > width - 1)
        or np.any(quad[:, 1] < 0.0)
        or np.any(quad[:, 1] > height - 1)
    ):
        return None
    if cv2.contourArea(quad) / float(width * height) < _MINIMUM_AREA_RATIO:
        return None
    return quad


def _mask_quads(
    probabilities: np.ndarray,
    *,
    height: int,
    width: int,
    maximum_fits: int,
):
    mask_height, mask_width = probabilities.shape
    scale = np.asarray(
        [(width - 1) / (mask_width - 1), (height - 1) / (mask_height - 1)],
        dtype=np.float32,
    )
    attempted_fits = 0
    for threshold_index, threshold in enumerate(_MASK_PROBABILITY_THRESHOLDS):
        binary = (probabilities >= threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        component_indices = sorted(
            range(1, count),
            key=lambda index: (-int(stats[index, cv2.CC_STAT_AREA]), index),
        )[:_MAXIMUM_COMPONENTS_PER_THRESHOLD]
        for component_rank, component_index in enumerate(component_indices):
            if (
                int(stats[component_index, cv2.CC_STAT_AREA])
                / float(mask_height * mask_width)
                < _MINIMUM_AREA_RATIO
            ):
                continue
            if attempted_fits >= maximum_fits:
                return
            attempted_fits += 1
            component = (labels == component_index).astype(np.uint8)
            fitted = quad_from_mask(component, min_area_ratio=_MINIMUM_AREA_RATIO)
            if fitted is None:
                continue
            corners = _valid_source_corners(
                fitted.corners * scale,
                height=height,
                width=width,
            )
            if corners is None:
                continue
            source_index = (
                threshold_index * _MAXIMUM_COMPONENTS_PER_THRESHOLD + component_rank
            )
            yield source_index, corners


def generate_candidates(
    image,
    docaligner,
    corner_model,
    *,
    maximum_candidates,
    maximum_corner_displacement_ratio,
) -> tuple[FusionCandidate, ...]:
    """Generate stable source-only quadrilateral candidates under fixed caps."""
    maximum_candidates, displacement_ratio = _validate_generation_limits(
        maximum_candidates,
        maximum_corner_displacement_ratio,
    )
    height, width = _image_size(image)
    normalization = np.asarray([width - 1, height - 1], dtype=np.float64)
    candidates: list[FusionCandidate] = []
    seen: set[tuple[float, ...]] = set()

    def add(family: CandidateFamily, corners: object, source_index: int) -> bool:
        valid = _valid_source_corners(corners, height=height, width=width)
        if valid is None:
            return False
        key = tuple(
            float(value)
            for value in np.round(
                valid.astype(np.float64) / normalization,
                _DUPLICATE_DECIMALS,
            ).flat
        )
        if key in seen:
            return False
        seen.add(key)
        candidates.append(FusionCandidate(family, valid, source_index))
        return True

    coarse = None
    if isinstance(docaligner, DocAlignerSignals) and docaligner.corners is not None:
        coarse = _valid_source_corners(
            docaligner.corners,
            height=height,
            width=width,
        )
        if coarse is not None:
            add("docaligner", coarse, 0)
            if len(candidates) >= maximum_candidates:
                return tuple(candidates)

    if not isinstance(corner_model, CornerModelSignals):
        return tuple(candidates)

    heatmap_corners = _valid_source_corners(
        corner_model.normalized_corners
        * np.asarray([width - 1, height - 1], dtype=np.float32),
        height=height,
        width=width,
    )
    if heatmap_corners is not None:
        add("corner_heatmap", heatmap_corners, 0)
        if len(candidates) >= maximum_candidates:
            return tuple(candidates)

    probabilities = np.asarray(corner_model.mask_probabilities)
    if (
        probabilities.shape != (96, 96)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        return tuple(candidates)

    diagonal = float(np.hypot(width, height))
    maximum_displacement = displacement_ratio * diagonal
    for source_index, mask_corners in _mask_quads(
        probabilities,
        height=height,
        width=width,
        maximum_fits=maximum_candidates - len(candidates),
    ):
        add("mask", mask_corners, source_index)
        if len(candidates) >= maximum_candidates:
            return tuple(candidates)
        if coarse is None:
            continue
        for corner_index in range(4):
            displacement = float(
                np.linalg.norm(mask_corners[corner_index] - coarse[corner_index])
            )
            if displacement > maximum_displacement + 1e-7:
                continue
            hybrid = coarse.copy()
            hybrid[corner_index] = mask_corners[corner_index]
            add("hybrid", hybrid, source_index)
            if len(candidates) >= maximum_candidates:
                return tuple(candidates)
    return tuple(candidates)


__all__ = ["FusionCandidate", "generate_candidates"]
