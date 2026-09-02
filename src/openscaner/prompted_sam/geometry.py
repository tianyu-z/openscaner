"""Deterministic crop, prompt, and coordinate geometry for prompted MobileSAM."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from openscaner.geometry import order_quad, validate_quad
from openscaner.postprocess import MaskQuad, quad_from_mask


def _finite_quad(corners: object) -> np.ndarray:
    quad = np.asarray(corners)
    if (
        quad.shape != (4, 2)
        or not np.issubdtype(quad.dtype, np.number)
        or not np.isfinite(quad).all()
    ):
        raise ValueError("corners must be a finite 4x2 array")
    return order_quad(quad).astype(np.float32)


@dataclass(frozen=True, slots=True)
class CropTransform:
    """Half-open source crop bounds and exact translation helpers."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(type(value) is not int for value in values):
            raise TypeError("crop bounds must be integers")
        if self.left < 0 or self.top < 0 or self.right <= self.left or self.bottom <= self.top:
            raise ValueError("crop bounds must describe a non-empty positive rectangle")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def _points(self, points: object) -> np.ndarray:
        array = np.asarray(points)
        if (
            array.ndim < 1
            or array.shape[-1] != 2
            or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all()
        ):
            raise ValueError("points must be a finite array ending in two coordinates")
        return array.astype(np.float32)

    def to_crop(self, points: object) -> np.ndarray:
        return self._points(points) - np.array([self.left, self.top], dtype=np.float32)

    def to_source(self, points: object) -> np.ndarray:
        return self._points(points) + np.array([self.left, self.top], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """One bounded set of MobileSAM prompts in crop coordinates."""

    box: np.ndarray
    points: np.ndarray
    labels: np.ndarray
    family: str

    def __post_init__(self) -> None:
        box = np.asarray(self.box, dtype=np.float32)
        points = np.asarray(self.points, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int32)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("prompt box must contain four finite coordinates")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("prompt box must have positive area")
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("prompt points must be a finite Nx2 array")
        if labels.shape != (len(points),) or not set(labels.tolist()) <= {0, 1}:
            raise ValueError("prompt labels must contain one binary label per point")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("prompt family must be non-empty")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "labels", labels)


def crop_from_quad(
    image_shape: tuple[int, ...],
    corners: object,
    *,
    margin_ratio: float,
) -> CropTransform:
    if (
        isinstance(margin_ratio, bool)
        or not isinstance(margin_ratio, (int, float, np.integer, np.floating))
        or not np.isfinite(float(margin_ratio))
        or float(margin_ratio) < 0.0
    ):
        raise ValueError("margin_ratio must be a finite non-negative number")
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = image_shape[:2]
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("image_shape must contain positive integer dimensions")

    quad = _finite_quad(corners)
    valid, reason = validate_quad(quad, (height, width), reorder=False)
    if not valid:
        raise ValueError(f"invalid coarse quadrilateral: {reason}")
    lower = quad.min(axis=0)
    upper = quad.max(axis=0)
    span = upper - lower
    expanded_lower = np.floor(lower - float(margin_ratio) * span).astype(np.int64)
    expanded_upper = np.ceil(upper + float(margin_ratio) * span).astype(np.int64) + 1
    left = max(0, int(expanded_lower[0]))
    top = max(0, int(expanded_lower[1]))
    right = min(width, int(expanded_upper[0]))
    bottom = min(height, int(expanded_upper[1]))
    return CropTransform(left=left, top=top, right=right, bottom=bottom)


def prompt_from_quad(
    corners: object,
    crop: CropTransform,
    *,
    foreground_scale: float,
    include_background: bool,
) -> PromptBundle:
    if not isinstance(crop, CropTransform):
        raise TypeError("crop must be a CropTransform")
    if (
        isinstance(foreground_scale, bool)
        or not isinstance(foreground_scale, (int, float, np.integer, np.floating))
        or not np.isfinite(float(foreground_scale))
        or not 0.0 < float(foreground_scale) < 1.0
    ):
        raise ValueError("foreground_scale must be finite and strictly between zero and one")
    if type(include_background) is not bool:
        raise TypeError("include_background must be boolean")

    local_quad = crop.to_crop(_finite_quad(corners))
    if (
        np.any(local_quad[:, 0] < 0.0)
        or np.any(local_quad[:, 0] > crop.width - 1)
        or np.any(local_quad[:, 1] < 0.0)
        or np.any(local_quad[:, 1] > crop.height - 1)
    ):
        raise ValueError("coarse quadrilateral must lie inside the crop")
    centroid = local_quad.mean(axis=0)
    foreground = np.concatenate(
        (
            centroid[None],
            centroid[None] + float(foreground_scale) * (local_quad - centroid[None]),
        ),
        axis=0,
    ).astype(np.float32)
    if any(cv2.pointPolygonTest(local_quad, point, False) <= 0 for point in foreground):
        raise ValueError("foreground prompts must lie strictly inside the coarse quadrilateral")

    points = foreground
    labels = np.ones(len(foreground), dtype=np.int32)
    family = "box_five_positive"
    if include_background:
        background = np.array(
            [
                [0.0, 0.0],
                [float(crop.width - 1), 0.0],
                [float(crop.width - 1), float(crop.height - 1)],
                [0.0, float(crop.height - 1)],
            ],
            dtype=np.float32,
        )
        if any(cv2.pointPolygonTest(local_quad, point, False) >= 0 for point in background):
            raise ValueError("background prompts must lie strictly outside the coarse quadrilateral")
        points = np.concatenate((points, background), axis=0)
        labels = np.concatenate((labels, np.zeros(4, dtype=np.int32)))
        family = "box_five_positive_four_background"

    lower = local_quad.min(axis=0)
    upper = local_quad.max(axis=0)
    box = np.array([lower[0], lower[1], upper[0], upper[1]], dtype=np.float32)
    return PromptBundle(box=box, points=points, labels=labels, family=family)


def mask_quad_in_source(
    mask: np.ndarray,
    crop: CropTransform,
    source_image: np.ndarray,
    *,
    foreground_points: np.ndarray | None = None,
) -> MaskQuad | None:
    if not isinstance(crop, CropTransform):
        raise TypeError("crop must be a CropTransform")
    source = np.asarray(source_image)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        raise ValueError("source_image must be a non-empty three-channel image")
    height, width = source.shape[:2]
    if crop.right > width or crop.bottom > height:
        raise ValueError("crop must lie inside source_image")

    candidate_mask = np.asarray(mask)
    if foreground_points is not None:
        points = np.asarray(foreground_points)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.issubdtype(points.dtype, np.number)
            or not np.isfinite(points).all()
        ):
            raise ValueError("foreground_points must be a finite Nx2 array")
        if candidate_mask.ndim != 2 or candidate_mask.size == 0:
            return None
        binary = (candidate_mask > 0).astype(np.uint8)
        count, labels = cv2.connectedComponents(binary, connectivity=8)
        selected_labels: set[int] = set()
        for point in points:
            x = int(np.clip(np.rint(point[0]), 0, binary.shape[1] - 1))
            y = int(np.clip(np.rint(point[1]), 0, binary.shape[0] - 1))
            y0, y1 = max(0, y - 2), min(binary.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(binary.shape[1], x + 3)
            selected_labels.update(
                int(value)
                for value in np.unique(labels[y0:y1, x0:x1])
                if 0 < int(value) < count
            )
        if selected_labels:
            selected = np.isin(labels, tuple(sorted(selected_labels))).astype(np.uint8)
            selected_points = cv2.findNonZero(selected)
            if selected_points is None or len(selected_points) < 4:
                return None
            hull = cv2.convexHull(selected_points)
            joined = np.zeros_like(selected)
            cv2.fillConvexPoly(joined, hull, 1)
            candidate_mask = joined

    local_image = source[crop.top : crop.bottom, crop.left : crop.right]
    recovered = quad_from_mask(
        candidate_mask,
        source_image=local_image,
        source_color_order="BGR",
        min_area_ratio=0.01,
    )
    if recovered is None:
        geometry_only = quad_from_mask(candidate_mask, min_area_ratio=0.01)
        if geometry_only is not None:
            recovered = MaskQuad(
                corners=geometry_only.corners,
                area_ratio=geometry_only.area_ratio,
                confidence=0.0,
                geometric_confidence=geometry_only.geometric_confidence,
                edge_confidence=0.0,
            )
    if recovered is None:
        return None
    source_corners = crop.to_source(recovered.corners)
    valid, _ = validate_quad(source_corners, source.shape, reorder=False)
    if not valid:
        return None
    area_ratio = abs(cv2.contourArea(source_corners)) / float(height * width)
    return MaskQuad(
        corners=source_corners,
        area_ratio=area_ratio,
        confidence=recovered.confidence,
        geometric_confidence=recovered.geometric_confidence,
        edge_confidence=recovered.edge_confidence,
    )


__all__ = [
    "CropTransform",
    "PromptBundle",
    "crop_from_quad",
    "mask_quad_in_source",
    "prompt_from_quad",
]
