"""Geometry for canonical document-corner patches and local heatmaps."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import cv2
import numpy as np

from openscaner.geometry import _order_quad_indices

PATCH_SIZE = 256
HEATMAP_SIZE = 64

_LOCAL_SOFT_ARGMAX_RADIUS = 2
_LOCAL_SOFTMAX_LOGIT_RANGE = 80.0
_MAX_BASIS_CONDITION = 1_000_000.0
_FLOAT64_GEOMETRY_ERROR_FACTOR = 32.0


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _corner_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("corner_index must be an integer")
    converted = int(value)
    if converted < 0 or converted >= 4:
        raise ValueError("corner_index must be in [0, 3]")
    return converted


def _patch_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("patch_size must be an integer")
    converted = int(value)
    if converted < 2 or converted > np.iinfo(np.int32).max:
        raise ValueError("patch_size must be between 2 and INT32_MAX")
    return converted


def _affine_matrix(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.shape != (2, 3):
        raise ValueError(f"{name} must have shape (2, 3)")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")

    converted = np.array(value, dtype=np.float32, copy=True)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values")
    condition = float(np.linalg.cond(converted[:, :2]))
    if not np.isfinite(condition) or condition >= _MAX_BASIS_CONDITION:
        raise ValueError(f"{name} linear part is singular or near-singular")
    return np.frombuffer(
        converted.tobytes(order="C"),
        dtype=np.float32,
    ).reshape(2, 3)


def _transform_points(points: object, matrix: np.ndarray) -> np.ndarray:
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("points must have shape (..., 2)")
    if not (
        np.issubdtype(points.dtype, np.integer)
        or np.issubdtype(points.dtype, np.floating)
    ):
        raise TypeError("points must use a real numeric dtype")

    converted = np.asarray(points, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError("points must contain only finite values")
    return converted @ matrix[:, :2].T + matrix[:, 2]


@dataclass(frozen=True, eq=False, slots=True)
class PatchTransform:
    """Immutable affine mapping between source and canonical patch pixels.

    Matrices use canonical float32 coefficients. Consequently, point round-trip
    error is bounded by float32 precision and coordinate scale, not a universal
    absolute tolerance.
    """

    corner_index: int
    radius_ratio: float
    local_side: float
    source_to_patch: np.ndarray
    patch_to_source: np.ndarray

    def __post_init__(self) -> None:
        corner_index = _corner_index(self.corner_index)
        radius_ratio = _positive_real(self.radius_ratio, "radius_ratio")
        local_side = _positive_real(self.local_side, "local_side")
        source_to_patch = _affine_matrix(self.source_to_patch, "source_to_patch")
        patch_to_source = _affine_matrix(self.patch_to_source, "patch_to_source")

        expected = cv2.invertAffineTransform(source_to_patch).astype(np.float32)
        if not np.array_equal(patch_to_source, expected):
            raise ValueError("source_to_patch and patch_to_source must be inverses")

        object.__setattr__(self, "corner_index", corner_index)
        object.__setattr__(self, "radius_ratio", radius_ratio)
        object.__setattr__(self, "local_side", local_side)
        object.__setattr__(self, "source_to_patch", source_to_patch)
        object.__setattr__(self, "patch_to_source", patch_to_source)

    def source_to_patch_points(self, points: np.ndarray) -> np.ndarray:
        """Map source points to canonical patch pixel coordinates."""
        return _transform_points(points, self.source_to_patch)

    def patch_to_source_points(self, points: np.ndarray) -> np.ndarray:
        """Map canonical patch points back to source pixel coordinates."""
        return _transform_points(points, self.patch_to_source)


def _numeric_quad(coarse_corners: object) -> np.ndarray:
    try:
        corners = np.asarray(coarse_corners)
    except (TypeError, ValueError) as error:
        raise TypeError("coarse_corners must be a real numeric array") from error
    if corners.shape != (4, 2):
        raise ValueError("coarse_corners must have shape (4, 2)")
    if not (
        np.issubdtype(corners.dtype, np.integer)
        or np.issubdtype(corners.dtype, np.floating)
    ):
        raise TypeError("coarse_corners must use a real numeric dtype")

    quad = np.asarray(corners, dtype=np.float64)
    if not np.isfinite(quad).all():
        raise ValueError("coarse_corners must contain only finite values")
    return quad


def _validate_ordered_quad(
    coarse_corners: object,
    image_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    quad = _numeric_quad(coarse_corners)
    if (quad < 0.0).any():
        raise ValueError("invalid quadrilateral: out_of_bounds")
    if image_shape is not None:
        height, width = image_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("invalid quadrilateral: invalid_image")
        if (quad[:, 0] >= width).any() or (quad[:, 1] >= height).any():
            raise ValueError("invalid quadrilateral: out_of_bounds")

    edges = np.roll(quad, -1, axis=0) - quad
    coordinate_scale = max(1.0, float(np.max(np.abs(quad))))
    length_epsilon = (
        _FLOAT64_GEOMETRY_ERROR_FACTOR
        * np.finfo(np.float64).eps
        * coordinate_scale
    )
    edge_lengths = np.linalg.norm(edges, axis=1)
    if np.any(edge_lengths <= length_epsilon):
        raise ValueError("invalid quadrilateral: degenerate_edge")

    edge_scale = float(np.max(edge_lengths))
    cross_epsilon = length_epsilon * max(edge_scale, length_epsilon)

    def orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
        first_edge = second - first
        second_edge = third - first
        return float(
            first_edge[0] * second_edge[1]
            - first_edge[1] * second_edge[0]
        )

    def segments_cross(
        first: np.ndarray,
        second: np.ndarray,
        third: np.ndarray,
        fourth: np.ndarray,
    ) -> bool:
        first_side = orientation(first, second, third)
        second_side = orientation(first, second, fourth)
        third_side = orientation(third, fourth, first)
        fourth_side = orientation(third, fourth, second)
        return bool(
            (
                (first_side > cross_epsilon and second_side < -cross_epsilon)
                or (first_side < -cross_epsilon and second_side > cross_epsilon)
            )
            and (
                (third_side > cross_epsilon and fourth_side < -cross_epsilon)
                or (third_side < -cross_epsilon and fourth_side > cross_epsilon)
            )
        )

    if segments_cross(quad[0], quad[1], quad[2], quad[3]) or segments_cross(
        quad[1], quad[2], quad[3], quad[0]
    ):
        raise ValueError("invalid quadrilateral: self_intersection")

    next_edges = np.roll(edges, -1, axis=0)
    cross_products = (
        edges[:, 0] * next_edges[:, 1]
        - edges[:, 1] * next_edges[:, 0]
    )
    if np.all(cross_products < -cross_epsilon):
        raise ValueError("invalid quadrilateral: reversed_winding")
    if np.any(cross_products <= cross_epsilon):
        raise ValueError("invalid quadrilateral: not_convex")

    twice_area = orientation(quad[0], quad[1], quad[2]) + orientation(
        quad[0],
        quad[2],
        quad[3],
    )
    if twice_area <= 2.0 * cross_epsilon:
        raise ValueError("invalid quadrilateral: degenerate_area")

    with np.errstate(over="ignore"):
        ordering_quad = quad.astype(np.float32)
    if (
        not np.isfinite(ordering_quad).all()
        or np.unique(ordering_quad, axis=0).shape[0] != 4
        or not np.array_equal(
            _order_quad_indices(ordering_quad),
            np.arange(4),
        )
    ):
        raise ValueError("invalid quadrilateral: invalid_semantic_order")
    return quad


def build_patch_transform(
    coarse_corners: np.ndarray,
    corner_index: int,
    *,
    radius_ratio: float,
    patch_size: int = PATCH_SIZE,
) -> PatchTransform:
    """Build the affine transform for one ordered document-corner role."""
    index = _corner_index(corner_index)
    ratio = _positive_real(radius_ratio, "radius_ratio")
    size = _patch_size(patch_size)
    quad = _validate_ordered_quad(coarse_corners)

    origin = quad[index]
    x_edge = quad[(index + 1) % 4] - origin
    y_edge = quad[(index - 1) % 4] - origin
    x_length = float(np.linalg.norm(x_edge))
    y_length = float(np.linalg.norm(y_edge))
    if not np.isfinite(x_length) or not np.isfinite(y_length):
        raise ValueError("adjacent edge lengths must be finite")
    if x_length <= 0.0 or y_length <= 0.0:
        raise ValueError("adjacent edge lengths must be positive")

    local_side = min(x_length, y_length)
    radius = ratio * local_side
    half = (size - 1) / 2.0
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("patch radius must be finite and positive")

    basis = np.column_stack((x_edge / x_length, y_edge / y_length))
    condition = float(np.linalg.cond(basis))
    if not np.isfinite(condition) or condition >= _MAX_BASIS_CONDITION:
        raise ValueError("adjacent edge basis is singular or near-singular")

    linear = (half / radius) * np.linalg.inv(basis)
    translation = np.array([half, half], dtype=np.float64) - linear @ origin
    source_to_patch = np.column_stack((linear, translation)).astype(np.float32)
    patch_to_source = cv2.invertAffineTransform(source_to_patch).astype(np.float32)
    if not (
        np.isfinite(source_to_patch).all()
        and np.isfinite(patch_to_source).all()
    ):
        raise ValueError("patch transform matrices must be finite")

    return PatchTransform(
        corner_index=index,
        radius_ratio=ratio,
        local_side=local_side,
        source_to_patch=source_to_patch,
        patch_to_source=patch_to_source,
    )


def _validate_source_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8 BGR pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")
    return image


def extract_corner_patches(
    image: np.ndarray,
    coarse_corners: np.ndarray,
    *,
    radius_ratio: float,
) -> tuple[np.ndarray, tuple[PatchTransform, PatchTransform, PatchTransform, PatchTransform]]:
    """Extract four canonical uint8 BGR patches without clipping coordinates."""
    source = _validate_source_image(image)
    quad = _validate_ordered_quad(coarse_corners, source.shape)
    transforms = tuple(
        build_patch_transform(quad, index, radius_ratio=radius_ratio)
        for index in range(4)
    )
    patches = np.stack(
        [
            cv2.warpAffine(
                source,
                transform.source_to_patch,
                (PATCH_SIZE, PATCH_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(127, 127, 127),
            )
            for transform in transforms
        ]
    )
    return patches, transforms  # type: ignore[return-value]


def _normalize_decode_logits(values: np.ndarray) -> np.ndarray:
    compute_dtype = np.float64 if values.dtype == np.float64 else np.float32
    promoted = values.astype(compute_dtype, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = promoted - promoted.max()
    shifted = np.where(np.isneginf(shifted), -_LOCAL_SOFTMAX_LOGIT_RANGE, shifted)
    if not np.isfinite(shifted).all():
        raise RuntimeError("local heatmap normalization produced non-finite values")
    return np.clip(shifted, -_LOCAL_SOFTMAX_LOGIT_RANGE, 0.0)


def _quadratic_border_pad(values: np.ndarray) -> np.ndarray:
    left_near = 3.0 * values[:, :1] - 3.0 * values[:, 1:2] + values[:, 2:3]
    left_far = 6.0 * values[:, :1] - 8.0 * values[:, 1:2] + 3.0 * values[:, 2:3]
    right_near = 3.0 * values[:, -1:] - 3.0 * values[:, -2:-1] + values[:, -3:-2]
    right_far = (
        6.0 * values[:, -1:]
        - 8.0 * values[:, -2:-1]
        + 3.0 * values[:, -3:-2]
    )
    horizontal = np.concatenate(
        (left_far, left_near, values, right_near, right_far),
        axis=1,
    )

    top_near = 3.0 * horizontal[:1] - 3.0 * horizontal[1:2] + horizontal[2:3]
    top_far = 6.0 * horizontal[:1] - 8.0 * horizontal[1:2] + 3.0 * horizontal[2:3]
    bottom_near = (
        3.0 * horizontal[-1:]
        - 3.0 * horizontal[-2:-1]
        + horizontal[-3:-2]
    )
    bottom_far = (
        6.0 * horizontal[-1:]
        - 8.0 * horizontal[-2:-1]
        + 3.0 * horizontal[-3:-2]
    )
    return np.concatenate(
        (top_far, top_near, horizontal, bottom_near, bottom_far),
        axis=0,
    )


def decode_local_heatmaps(logits: np.ndarray, radius: int = 2) -> np.ndarray:
    """Decode N local heatmaps into unclamped canonical patch x,y pixels."""
    if isinstance(radius, bool) or not isinstance(radius, Integral):
        raise TypeError("radius must be an integer")
    if int(radius) != _LOCAL_SOFT_ARGMAX_RADIUS:
        raise ValueError("radius must be 2 for the canonical 5x5 decoder")
    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a numpy array")
    if (
        logits.ndim != 4
        or logits.shape[0] < 1
        or logits.shape[1] != 1
        or logits.shape[2] <= 2
        or logits.shape[3] <= 2
    ):
        raise ValueError(
            "logits must have shape N x 1 x H x W with N > 0 and H,W > 2"
        )
    if not np.issubdtype(logits.dtype, np.floating):
        raise TypeError("logits must use a floating-point dtype")
    if not np.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")

    batch_size, _, height, width = logits.shape
    decoded = np.empty((batch_size, 2), dtype=np.float32)
    for batch_index in range(batch_size):
        values = logits[batch_index, 0]
        peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)
        normalized = _normalize_decode_logits(values)
        padded = _quadratic_border_pad(normalized)
        window = padded[
            peak_y : peak_y + 2 * _LOCAL_SOFT_ARGMAX_RADIUS + 1,
            peak_x : peak_x + 2 * _LOCAL_SOFT_ARGMAX_RADIUS + 1,
        ]
        stable = window - window.max()
        weights = np.exp(stable)
        weights /= weights.sum()

        coordinate_dtype = normalized.dtype
        local_x = np.arange(
            peak_x - _LOCAL_SOFT_ARGMAX_RADIUS,
            peak_x + _LOCAL_SOFT_ARGMAX_RADIUS + 1,
            dtype=coordinate_dtype,
        )
        local_y = np.arange(
            peak_y - _LOCAL_SOFT_ARGMAX_RADIUS,
            peak_y + _LOCAL_SOFT_ARGMAX_RADIUS + 1,
            dtype=coordinate_dtype,
        )
        yy, xx = np.meshgrid(local_y, local_x, indexing="ij")
        x = float(np.sum(weights * xx)) * (PATCH_SIZE - 1) / (width - 1)
        y = float(np.sum(weights * yy)) * (PATCH_SIZE - 1) / (height - 1)
        decoded[batch_index] = (x, y)

    if not np.isfinite(decoded).all():
        raise RuntimeError("local heatmap decoding produced non-finite output")
    return decoded
