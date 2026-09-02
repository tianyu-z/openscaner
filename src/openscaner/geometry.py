from __future__ import annotations

import cv2
import numpy as np


def _order_quad_indices(quad: np.ndarray) -> np.ndarray:
    """Return repository-order indices for a validated float32 quad."""
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    ordered_indices = np.argsort(angles)
    start = int(np.argmin(quad[ordered_indices].sum(axis=1)))
    return np.roll(ordered_indices, -start)


def order_quad(points: np.ndarray) -> np.ndarray:
    """Return four points in top-left, top-right, bottom-right, bottom-left order."""
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError("quadrilateral must contain exactly four 2D points")
    if not np.isfinite(quad).all():
        raise ValueError("quadrilateral points must be finite")

    ordered = quad[_order_quad_indices(quad)]
    return ordered.astype(np.float32)


def cyclic_quad_mean_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Compare ordered quads over cyclic starts without reversing winding."""
    actual_quad = np.asarray(actual, dtype=np.float32)
    expected_quad = np.asarray(expected, dtype=np.float32)
    if actual_quad.shape != (4, 2) or expected_quad.shape != (4, 2):
        raise ValueError("quadrilaterals must each contain exactly four 2D points")
    if not np.isfinite(actual_quad).all() or not np.isfinite(expected_quad).all():
        raise ValueError("quadrilateral points must be finite")
    return min(
        float(
            np.linalg.norm(
                np.roll(actual_quad, shift, axis=0) - expected_quad,
                axis=1,
            ).mean()
        )
        for shift in range(4)
    )


def validate_quad(
    points: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    min_area_ratio: float = 0.01,
    reorder: bool = True,
) -> tuple[bool, str]:
    try:
        quad = order_quad(points) if reorder else np.asarray(points, dtype=np.float32)
    except ValueError:
        return False, "invalid_shape"
    if quad.shape != (4, 2) or not cv2.isContourConvex(quad.astype(np.int32)):
        return False, "not_convex"

    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        return False, "invalid_image"
    if (quad[:, 0] < 0).any() or (quad[:, 0] >= width).any() or (quad[:, 1] < 0).any() or (quad[:, 1] >= height).any():
        return False, "out_of_bounds"
    area_ratio = abs(cv2.contourArea(quad)) / float(width * height)
    if area_ratio < min_area_ratio:
        return False, "too_small"
    return True, "ok"


def warp_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    quad = order_quad(corners)
    valid, reason = validate_quad(quad, image.shape, min_area_ratio=0.0, reorder=False)
    if not valid:
        raise ValueError(f"invalid quadrilateral: {reason}")

    tl, tr, br, bl = quad
    width = max(1, round((np.linalg.norm(tl - tr) + np.linalg.norm(bl - br)) / 2.0))
    height = max(1, round((np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2.0))
    source = np.float32([tl, tr, bl, br])
    destination = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_CUBIC)
