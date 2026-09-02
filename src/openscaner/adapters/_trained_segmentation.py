"""Shared CPU inference path for locally trained binary segmenters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import BytesIO
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from openscaner.adapters.base import AdapterOutput, AdapterUnavailable
from openscaner.geometry import order_quad, validate_quad
from openscaner.postprocess import quad_from_mask
from openscaner.training.data import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

_QUADRILATERAL_MASK_SIZE = 192
_PROBABILITY_THRESHOLDS = (
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
)
_REDUCTION_EPSILON_RATIOS = tuple(np.linspace(0.002, 0.08, 40))
_MAX_COMPONENTS_PER_THRESHOLD = 2
_MAX_REDUCED_VERTICES = 12
_MAX_COMBINATIONS_PER_COMPONENT = 128
_MAX_QUADRILATERAL_CANDIDATES = 12
_SOFT_DICE_SCORE_WEIGHT = 0.25


@dataclass(frozen=True)
class _MaskCandidate:
    mask: np.ndarray
    threshold: float
    probability_agreement: float


@dataclass(frozen=True)
class _CompactCandidate:
    corners: np.ndarray
    threshold: float
    probability_agreement: float
    preselection_score: float


def _bounded_quadrilateral_combinations(
    vertex_count: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield globally distributed four-vertex sets under a fixed work cap."""
    yielded = 0
    for indices in combinations(range(vertex_count), 4):
        if vertex_count > 8:
            cyclic_gaps = (
                indices[1] - indices[0],
                indices[2] - indices[1],
                indices[3] - indices[2],
                vertex_count + indices[0] - indices[3],
            )
            if min(cyclic_gaps) < 2:
                continue
        yield indices
        yielded += 1
        if yielded >= _MAX_COMBINATIONS_PER_COMPONENT:
            return


def _convex_polygon_row_spans(
    corners: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inclusive integer x spans for every pixel row inside a convex quad."""
    minimum_y = max(0, int(np.ceil(float(corners[:, 1].min()))))
    maximum_y = min(height - 1, int(np.floor(float(corners[:, 1].max()))))
    if minimum_y > maximum_y:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty

    rows = np.arange(minimum_y, maximum_y + 1, dtype=np.int32)
    left_edges = np.full(rows.shape, np.inf, dtype=np.float64)
    right_edges = np.full(rows.shape, -np.inf, dtype=np.float64)
    row_values = rows.astype(np.float64)
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        start_x, start_y = (float(value) for value in start)
        end_x, end_y = (float(value) for value in end)
        if start_y == end_y:
            active = row_values == start_y
            intersections = np.full(rows.shape, min(start_x, end_x))
            opposite = np.full(rows.shape, max(start_x, end_x))
            left_edges[active] = np.minimum(left_edges[active], intersections[active])
            right_edges[active] = np.maximum(right_edges[active], opposite[active])
            continue
        active = (row_values >= min(start_y, end_y)) & (
            row_values <= max(start_y, end_y)
        )
        intersections = start_x + (
            (row_values - start_y) * (end_x - start_x) / (end_y - start_y)
        )
        left_edges[active] = np.minimum(left_edges[active], intersections[active])
        right_edges[active] = np.maximum(right_edges[active], intersections[active])

    valid = np.isfinite(left_edges) & np.isfinite(right_edges)
    rows = rows[valid]
    left = np.ceil(left_edges[valid] - 1e-9).astype(np.int32)
    right = np.floor(right_edges[valid] + 1e-9).astype(np.int32)
    left = np.clip(left, 0, width - 1)
    right = np.clip(right, 0, width - 1)
    nonempty = left <= right
    return rows[nonempty], left[nonempty], right[nonempty]


def _score_convex_polygon(
    corners: np.ndarray,
    probability_prefix: np.ndarray,
    component_prefix: np.ndarray,
    *,
    probability_mass: float,
    component_area: int,
) -> tuple[float, float]:
    """Return exact full-pixel component IoU and soft Dice without a mask."""
    height, padded_width = probability_prefix.shape
    rows, left, right = _convex_polygon_row_spans(
        corners,
        height=height,
        width=padded_width - 1,
    )
    candidate_area = int((right - left + 1).sum(dtype=np.int64))
    intersection = int(
        (
            component_prefix[rows, right + 1]
            - component_prefix[rows, left]
        ).sum(dtype=np.int64)
    )
    candidate_probability_mass = float(
        (
            probability_prefix[rows, right + 1]
            - probability_prefix[rows, left]
        ).sum(dtype=np.float64)
    )
    union = component_area + candidate_area - intersection
    component_iou = intersection / max(union, 1)
    soft_dice = float(
        2.0
        * candidate_probability_mass
        / max(probability_mass + candidate_area, 1e-8)
    )
    assert np.isfinite(soft_dice) and 0.0 <= soft_dice <= 1.0
    return component_iou, soft_dice


def _shortlist_component_indices(
    labels: np.ndarray,
    stats: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_area_ratio: float,
) -> list[int]:
    """Select a bounded union of cheap, complementary component rankings."""
    component_indices = [
        index
        for index in range(1, len(stats))
        if int(stats[index, cv2.CC_STAT_AREA]) / labels.size >= min_area_ratio
    ]
    if not component_indices:
        return []

    probability_sums = np.bincount(
        labels.reshape(-1),
        weights=probabilities.reshape(-1),
        minlength=len(stats),
    )
    areas = {
        index: int(stats[index, cv2.CC_STAT_AREA])
        for index in component_indices
    }
    mean_probabilities = {
        index: float(probability_sums[index] / areas[index])
        for index in component_indices
    }
    rectangular_support = {
        index: areas[index]
        / max(
            int(stats[index, cv2.CC_STAT_WIDTH])
            * int(stats[index, cv2.CC_STAT_HEIGHT]),
            1,
        )
        for index in component_indices
    }
    rankings = (
        sorted(component_indices, key=lambda index: (-areas[index], index)),
        sorted(
            component_indices,
            key=lambda index: (-mean_probabilities[index], -areas[index], index),
        ),
        sorted(
            component_indices,
            key=lambda index: (-rectangular_support[index], -areas[index], index),
        ),
    )

    selected: list[int] = []
    for rank in range(len(component_indices)):
        for ranking in rankings:
            component_index = ranking[rank]
            if component_index not in selected:
                selected.append(component_index)
                if len(selected) == _MAX_COMPONENTS_PER_THRESHOLD:
                    return selected
    return selected


def _quadrilateral_mask_candidates(
    probabilities: np.ndarray,
    *,
    min_area_ratio: float,
) -> list[_MaskCandidate]:
    """Fit robust quadrilaterals to learned components across probability levels."""
    candidates: dict[tuple[int, ...], _CompactCandidate] = {}
    height, width = probabilities.shape
    probability_prefix = np.zeros((height, width + 1), dtype=np.float64)
    np.cumsum(
        probabilities,
        axis=1,
        dtype=np.float64,
        out=probability_prefix[:, 1:],
    )
    probability_mass = float(probability_prefix[:, -1].sum(dtype=np.float64))
    output_scale = np.array(
        [_QUADRILATERAL_MASK_SIZE / width, _QUADRILATERAL_MASK_SIZE / height],
        dtype=np.float32,
    )

    for threshold in _PROBABILITY_THRESHOLDS:
        binary = (probabilities >= threshold).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        component_indices = _shortlist_component_indices(
            labels,
            stats,
            probabilities,
            min_area_ratio=min_area_ratio,
        )
        for component_index in component_indices:
            component_area = int(stats[component_index, cv2.CC_STAT_AREA])
            component = (labels == component_index).astype(np.uint8)
            component_prefix = np.zeros((height, width + 1), dtype=np.int32)
            np.cumsum(
                component,
                axis=1,
                dtype=np.int32,
                out=component_prefix[:, 1:],
            )
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                continue
            hull = cv2.convexHull(max(contours, key=cv2.contourArea))
            perimeter = cv2.arcLength(hull, True)
            if perimeter <= 0.0:
                continue
            reduced = None
            for epsilon_ratio in _REDUCTION_EPSILON_RATIOS:
                polygon = cv2.approxPolyDP(
                    hull,
                    float(epsilon_ratio * perimeter),
                    True,
                ).reshape(-1, 2)
                if 4 <= len(polygon) <= _MAX_REDUCED_VERTICES:
                    reduced = polygon
                    break
            if reduced is None:
                continue

            for indices in _bounded_quadrilateral_combinations(len(reduced)):
                try:
                    corners = order_quad(reduced[list(indices)])
                except ValueError:
                    continue
                valid, _ = validate_quad(
                    corners,
                    binary.shape,
                    min_area_ratio=min_area_ratio,
                    reorder=False,
                )
                if not valid:
                    continue
                component_iou, soft_dice = _score_convex_polygon(
                    corners,
                    probability_prefix,
                    component_prefix,
                    probability_mass=probability_mass,
                    component_area=component_area,
                )
                preselection_score = component_iou + (
                    _SOFT_DICE_SCORE_WEIGHT * soft_dice
                )
                scaled_corners = np.rint(corners * output_scale).astype(np.int32)
                key = tuple(int(value) for value in scaled_corners.flat)
                previous = candidates.get(key)
                compact_candidate = _CompactCandidate(
                    corners=scaled_corners,
                    threshold=threshold,
                    probability_agreement=soft_dice,
                    preselection_score=preselection_score,
                )
                if previous is not None:
                    if preselection_score > previous.preselection_score:
                        candidates[key] = compact_candidate
                    continue
                if len(candidates) < _MAX_QUADRILATERAL_CANDIDATES:
                    candidates[key] = compact_candidate
                    continue
                worst_key, worst = min(
                    candidates.items(),
                    key=lambda item: (item[1].preselection_score, item[0]),
                )
                if preselection_score > worst.preselection_score:
                    del candidates[worst_key]
                    candidates[key] = compact_candidate

    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1].preselection_score, item[0]),
        reverse=True,
    )
    mask_candidates = []
    for _, candidate in ranked:
        mask = np.zeros(
            (_QUADRILATERAL_MASK_SIZE, _QUADRILATERAL_MASK_SIZE),
            dtype=np.uint8,
        )
        cv2.fillConvexPoly(mask, candidate.corners, 1, cv2.LINE_8)
        mask_candidates.append(
            _MaskCandidate(
                mask=mask,
                threshold=candidate.threshold,
                probability_agreement=candidate.probability_agreement,
            )
        )
    return mask_candidates


def load_checkpoint(
    model_source: Path | bytes,
    *,
    model_name: str,
    builder: Callable[[], object],
):
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterUnavailable(
            "trained segmenter requires the torch runtime"
        ) from error
    try:
        checkpoint_input = (
            BytesIO(model_source) if isinstance(model_source, bytes) else model_source
        )
        checkpoint = torch.load(
            checkpoint_input,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("model_name") != model_name
        ):
            raise ValueError("checkpoint model identity does not match adapter")
        state_dict = checkpoint["state_dict"]
        image_size = int(checkpoint["image_size"])
        if image_size != IMAGE_SIZE:
            raise ValueError(f"checkpoint image size must be {IMAGE_SIZE}")
        model = builder()
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise AdapterUnavailable(
            f"{model_name} checkpoint is incompatible with its architecture"
        ) from error
    return model


def run_binary_segmenter(
    image: np.ndarray,
    *,
    model_path: Path | bytes,
    cpu_threads: int,
    loader: Callable[[Path | bytes], object],
    horizontal_flip_tta: bool = False,
) -> AdapterOutput:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        raise ValueError("image must be a non-empty three-channel image")
    if source.dtype != np.uint8:
        raise ValueError("image must use uint8 values")
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterUnavailable(
            "trained segmenter requires the torch runtime"
        ) from error

    torch.set_num_threads(cpu_threads)
    # Positive limits are advisory on OpenCV's macOS GCD backend and can leave
    # all system workers active. Sequential mode is the only portable upper
    # bound, while Torch still uses the requested CPU thread count.
    cv2.setNumThreads(0)
    model = loader(model_path)
    model.to("cpu")
    model.eval()
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalized = rgb.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1)).copy()).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        if not isinstance(logits, torch.Tensor) or logits.shape != (
            1,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise AdapterUnavailable(
                "trained segmenter returned an incompatible output"
            )
        probability_tensor = torch.sigmoid(logits)
        if horizontal_flip_tta:
            flipped_logits = model(torch.flip(tensor, dims=(-1,)))
            if not isinstance(flipped_logits, torch.Tensor) or flipped_logits.shape != (
                1,
                1,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ):
                raise AdapterUnavailable(
                    "trained segmenter returned an incompatible output"
                )
            probability_tensor = 0.5 * (
                probability_tensor
                + torch.flip(torch.sigmoid(flipped_logits), dims=(-1,))
            )
        probabilities = probability_tensor.squeeze().cpu().numpy()

    mask = (probabilities >= 0.5).astype(np.uint8)
    candidate_masks = _quadrilateral_mask_candidates(
        probabilities,
        min_area_ratio=0.01,
    )
    recovered_candidates = [
        (recovered, candidate)
        for candidate in candidate_masks
        if (
            recovered := quad_from_mask(
                candidate.mask,
                source_image=source,
                source_color_order="BGR",
                min_area_ratio=0.01,
            )
        )
        is not None
    ]
    selected = (
        max(
            recovered_candidates,
            key=lambda item: item[0].confidence * item[1].probability_agreement,
        )
        if recovered_candidates
        else None
    )
    diagnostics = {
        "accepted_candidates": len(recovered_candidates),
        "input_size": IMAGE_SIZE,
        "foreground_fraction": float(mask.mean()),
        "mean_probability": float(probabilities.mean()),
        "quadrilateral_candidates": len(candidate_masks),
    }
    if selected is None:
        return AdapterOutput(
            corners=None,
            confidence=0.0,
            backend="torch:cpu",
            diagnostics=diagnostics,
        )

    recovered, selected_candidate = selected
    selected_mask = selected_candidate.mask
    if selected_mask.shape != probabilities.shape:
        selected_mask = cv2.resize(
            selected_mask,
            (probabilities.shape[1], probabilities.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    foreground_confidence = float(
        probabilities[np.asarray(selected_mask, dtype=bool)].mean()
    )
    diagnostics["selected_threshold"] = selected_candidate.threshold
    diagnostics["selected_probability_agreement"] = (
        selected_candidate.probability_agreement
    )
    confidence = float(np.clip(foreground_confidence * recovered.confidence, 0.0, 1.0))
    return AdapterOutput(
        corners=recovered.corners,
        confidence=confidence,
        backend="torch:cpu",
        diagnostics=diagnostics,
    )
