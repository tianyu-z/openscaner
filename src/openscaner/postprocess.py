from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from openscaner.geometry import order_quad, validate_quad


_MIN_GEOMETRIC_CONFIDENCE = 0.75
_MIN_EDGE_CONFIDENCE = 0.25
_MIN_ABSOLUTE_EDGE_RESPONSE = 0.01
_MIN_EDGE_CONTINUITY = 0.20
_MIN_EDGE_POLARITY = 0.62
_MIN_ALIGNED_PROMINENCE = 0.09
_MIN_ALIGNED_T_STAT = 4.0
_MIN_RETURNED_EDGE_CONTINUITY = 0.15
_MIN_RETURNED_EDGE_POLARITY = 0.53
_MIN_RETURNED_ALIGNED_T_STAT = 1.0
_EDGE_STRENGTH_REFERENCE = 0.04
_MIN_CORNER_AREA_FRACTION = 0.01
_MIN_TURN_SINE = 0.04
_MIN_SIDE_COVERAGE = 0.60
_MIN_SIDE_RUN_PERIMETER_RATIO = 0.045
_MIN_SOURCE_SUPPORT_RATIO = 0.20
_SIDE_SUPPORT_BINS = 16
_STRUCTURE_EPSILON_RATIO = 0.008
_MAX_SEARCH_OFFSETS = 129
SourceColorOrder = Literal["BGR", "RGB", "GRAY"]


@dataclass(frozen=True)
class MaskQuad:
    """A document quadrilateral and calibrated confidence values in ``[0, 1]``.

    ``geometric_confidence`` combines hull-area agreement and contour-side
    coverage. ``edge_confidence`` is the third-strongest of four independently
    normalized source-edge scores at the returned sides, subject to aggregate
    physical side coverage, so one occluded side is allowed.
    Overall ``confidence`` is their product when source evidence is available.
    """

    corners: np.ndarray
    area_ratio: float
    confidence: float
    geometric_confidence: float
    edge_confidence: float | None


def _as_binary_mask(mask: np.ndarray) -> np.ndarray | None:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or array.size == 0:
        return None

    if np.issubdtype(array.dtype, np.floating):
        finite = np.where(np.isfinite(array), array, 0.0)
        if finite.min() >= 0.0 and finite.max() <= 1.0:
            foreground = finite > 0.5
        else:
            foreground = finite > 0.0
    else:
        foreground = array > 0
    return foreground.astype(np.uint8)


def _valid_quad(
    points: np.ndarray,
    image_shape: tuple[int, int],
    min_area_ratio: float,
) -> np.ndarray | None:
    try:
        corners = order_quad(points)
    except ValueError:
        return None
    valid, _ = validate_quad(corners, image_shape, min_area_ratio=min_area_ratio, reorder=False)
    if not valid:
        return None

    area = abs(cv2.contourArea(corners))
    for index, corner in enumerate(corners):
        previous_edge = corners[index - 1] - corner
        next_edge = corners[(index + 1) % 4] - corner
        cross = abs(float(previous_edge[0] * next_edge[1] - previous_edge[1] * next_edge[0]))
        lengths = float(np.linalg.norm(previous_edge) * np.linalg.norm(next_edge))
        if lengths <= 0.0 or cross / lengths < _MIN_TURN_SINE:
            return None
        if cross / area < _MIN_CORNER_AREA_FRACTION:
            return None
    return corners


def _polygon_quad(
    hull: np.ndarray,
    image_shape: tuple[int, int],
    min_area_ratio: float,
) -> np.ndarray | None:
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0.0:
        return None

    hull_area = cv2.contourArea(hull)
    candidates: list[tuple[float, np.ndarray]] = []
    for epsilon_ratio in (0.005, 0.008, 0.012, 0.016, 0.02, 0.03, 0.04, 0.06, 0.08):
        polygon = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(polygon) != 4:
            continue
        corners = _valid_quad(polygon.reshape(4, 2), image_shape, min_area_ratio)
        if corners is None:
            continue
        area_error = abs(cv2.contourArea(corners) - hull_area)
        candidates.append((area_error, corners))

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _side_support(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return None
    unit = direction / length
    relative = points - start
    projection = relative @ unit
    distance = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
    tolerance = max(1.0, min(2.0, 0.01 * length))
    supported = (projection >= 0.0) & (projection <= length) & (distance <= tolerance)
    return projection, supported, length


def _minimum_side_coverage(contour: np.ndarray, corners: np.ndarray) -> float:
    points = contour.reshape(-1, 2).astype(np.float64)
    coverages: list[float] = []
    for index, start in enumerate(corners.astype(np.float64)):
        end = corners[(index + 1) % 4].astype(np.float64)
        support = _side_support(points, start, end)
        if support is None:
            return 0.0
        projection, supported, length = support
        bin_count = min(_SIDE_SUPPORT_BINS, max(4, int(np.ceil(length))))
        bin_indices = np.floor(np.clip(projection[supported] / length, 0.0, 1.0 - 1e-8) * bin_count)
        coverages.append(float(len(np.unique(bin_indices))) / bin_count)
    return min(coverages)


def _minimum_side_run_ratio(contour: np.ndarray, corners: np.ndarray) -> float:
    """Return the shortest side's longest contiguous straight contour run."""
    points = contour.reshape(-1, 2).astype(np.float64)
    perimeter = float(cv2.arcLength(contour, True))
    if len(points) < 4 or perimeter <= 1e-8:
        return 0.0

    run_ratios: list[float] = []
    for index, start in enumerate(corners.astype(np.float64)):
        end = corners[(index + 1) % 4].astype(np.float64)
        support = _side_support(points, start, end)
        if support is None:
            return 0.0
        _, supported, _ = support
        if np.all(supported):
            run_ratios.append(1.0)
            continue

        first_gap = int(np.flatnonzero(~supported)[0])
        ordered_indices = (first_gap + 1 + np.arange(len(points))) % len(points)
        longest = 0.0
        current = 0.0
        previous_index: int | None = None
        for point_index in ordered_indices:
            point_index = int(point_index)
            if supported[point_index]:
                if previous_index is not None and supported[previous_index]:
                    current += float(np.linalg.norm(points[point_index] - points[previous_index]))
                longest = max(longest, current)
            else:
                current = 0.0
            previous_index = point_index
        run_ratios.append(longest / perimeter)
    return min(run_ratios)


def _reduced_hull(hull: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(hull, True)
    for epsilon_ratio in np.linspace(0.002, 0.08, 40):
        polygon = cv2.approxPolyDP(hull, float(epsilon_ratio * perimeter), True).reshape(-1, 2)
        if 4 <= len(polygon) <= 16:
            return polygon.astype(np.float32)

    points = hull.reshape(-1, 2)
    if len(points) < 4:
        return None
    indices = np.linspace(0, len(points) - 1, min(16, len(points)), dtype=int)
    return points[indices].astype(np.float32)


def _cyclic_segment(points: np.ndarray, start: int, end: int) -> np.ndarray:
    if start <= end:
        return points[start : end + 1]
    return np.concatenate((points[start:], points[: end + 1]))


def _support_line(
    points: np.ndarray,
    hull_points: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    if len(points) < 2:
        return None

    direction_x, direction_y, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
    normal = np.array([-direction_y, direction_x], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm <= 1e-8:
        return None
    normal /= norm

    offset = float(np.mean(points @ normal))
    if float(center @ normal) > offset:
        normal = -normal
    offset = float(np.max(hull_points @ normal))
    return normal, offset


def _line_intersection(
    first: tuple[np.ndarray, float],
    second: tuple[np.ndarray, float],
) -> np.ndarray | None:
    coefficients = np.stack((first[0], second[0]))
    if abs(np.linalg.det(coefficients)) <= 1e-6:
        return None
    return np.linalg.solve(coefficients, np.array([first[1], second[1]], dtype=np.float64))


def _gradient_components(
    image: np.ndarray,
    color_order: SourceColorOrder | None,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("source_image must not be empty")
    if source.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("source_image must use uint8, uint16, float32, or float64 values")

    is_float = np.issubdtype(source.dtype, np.floating)
    if is_float:
        if not np.isfinite(source).all():
            raise ValueError("source_image values must be finite")
        source_min = float(source.min())
        source_max = float(source.max())
        if source_min < 0.0 or source_max > 255.0:
            raise ValueError("source_image float values must be in the range [0, 255]")
        scale = 1.0 if source_max <= 1.0 else 1.0 / 255.0
    else:
        scale = 1.0 / float(np.iinfo(source.dtype).max)

    if source.ndim == 2:
        if color_order not in (None, "GRAY"):
            raise ValueError("two-dimensional source_image requires source_color_order='GRAY'")
        gray_source = source
    elif source.ndim == 3 and source.shape[2] == 3:
        if color_order == "BGR":
            conversion = cv2.COLOR_BGR2GRAY
        elif color_order == "RGB":
            conversion = cv2.COLOR_RGB2GRAY
        else:
            raise ValueError("three-channel source_image requires source_color_order='BGR' or 'RGB'")
        conversion_source = source if source.dtype != np.float64 else source.astype(np.float32)
        gray_source = cv2.cvtColor(conversion_source, conversion)
    else:
        raise ValueError("source_image must be a grayscale or three-channel image")

    gray = gray_source.astype(np.float32, copy=False)
    if scale != 1.0:
        if is_float:
            gray = gray * scale
        else:
            gray *= scale
    if is_float and not np.isfinite(gray).all():
        raise ValueError("source_image conversion produced non-finite values")
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_x *= 0.25
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_y *= 0.25
    if is_float and (not np.isfinite(gradient_x).all() or not np.isfinite(gradient_y).all()):
        raise ValueError("source_image gradients must be finite")
    return gradient_x, gradient_y


def _sample_directional_gradient(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    normal: np.ndarray,
) -> tuple[float, float, float, float, float]:
    length = float(np.linalg.norm(end - start))
    sample_count = min(512, max(24, int(np.ceil(length / 2.0))))
    positions = np.linspace(0.08, 0.92, sample_count, dtype=np.float32)[:, None]
    samples = start + positions * (end - start)
    sampled_x = cv2.remap(
        gradient_x,
        samples[:, 0].astype(np.float32),
        samples[:, 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_y = cv2.remap(
        gradient_y,
        samples[:, 0].astype(np.float32),
        samples[:, 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    values = (sampled_x * normal[0] + sampled_y * normal[1]).reshape(-1)
    magnitudes = np.abs(values)
    line_peak = float(np.percentile(magnitudes, 90))
    if line_peak <= 1e-8:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    supported = magnitudes >= max(0.001, 0.25 * line_peak)
    continuity = float(np.mean(supported))
    if not np.any(supported):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    positive_fraction = float(np.mean(values[supported] > 0.0))
    negative_fraction = float(np.mean(values[supported] < 0.0))
    polarity = max(positive_fraction, negative_fraction)
    absolute_response = float(magnitudes.mean())

    winsor_limit = float(np.percentile(magnitudes, 85))
    winsorized = np.clip(values, -winsor_limit, winsor_limit)
    aligned_response = float(winsorized.mean())
    standard_error = float(winsorized.std()) / np.sqrt(len(winsorized))
    t_statistic = abs(aligned_response) / max(standard_error, 1e-8)
    return absolute_response, continuity, polarity, aligned_response, t_statistic


def _search_offsets(radius: float) -> list[float]:
    if radius <= 0.0:
        return [0.0]
    per_direction = min((_MAX_SEARCH_OFFSETS - 1) // 2, max(1, int(np.ceil(radius / 0.5))))
    offsets = [0.0]
    for distance in np.linspace(radius / per_direction, radius, per_direction):
        offsets.extend((-float(distance), float(distance)))
    return offsets


def _parallel_edge_profile(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    normal: np.ndarray,
    offsets: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = [
        _sample_directional_gradient(
            gradient_x,
            gradient_y,
            start + normal * offset,
            end + normal * offset,
            normal,
        )
        for offset in offsets
    ]
    profile = np.asarray(samples, dtype=np.float64)
    return profile[:, 0], profile[:, 1], profile[:, 2], profile[:, 3], profile[:, 4]


def _line_edge_confidence(
    responses: np.ndarray,
    continuities: np.ndarray,
    polarities: np.ndarray,
    aligned_responses: np.ndarray,
    t_statistics: np.ndarray,
    *,
    index: int,
    sample_count: int,
    offset_fraction: float,
) -> float:
    response = float(responses[index])
    continuity = float(continuities[index])
    polarity = float(polarities[index])
    aligned_strengths = np.abs(aligned_responses)
    aligned_response = float(aligned_strengths[index])
    t_statistic = float(t_statistics[index])
    aligned_baseline = float(np.percentile(aligned_strengths, 75))
    prominence = (aligned_response - aligned_baseline) / max(aligned_baseline, 1e-8)
    small_sample_penalty = 1.5 * max(0.0, (256 - sample_count) / 256)
    required_t_statistic = _MIN_ALIGNED_T_STAT + small_sample_penalty + 0.05 * offset_fraction
    if (
        response < _MIN_ABSOLUTE_EDGE_RESPONSE
        or continuity < _MIN_EDGE_CONTINUITY
        or polarity < _MIN_EDGE_POLARITY
        or prominence < _MIN_ALIGNED_PROMINENCE
        or t_statistic < required_t_statistic
    ):
        return 0.0

    strength = min(1.0, response / _EDGE_STRENGTH_REFERENCE)
    continuity_strength = min(1.0, continuity / 0.60)
    polarity_strength = min(1.0, max(0.0, (polarity - 0.50) / 0.40))
    prominence_strength = min(1.0, prominence / 0.50)
    significance_strength = min(1.0, t_statistic / 8.0)
    return float(
        np.mean(
            (
                strength,
                continuity_strength,
                polarity_strength,
                prominence_strength,
                significance_strength,
            )
        )
    )


def _returned_line_confidence(
    responses: np.ndarray,
    continuities: np.ndarray,
    polarities: np.ndarray,
    aligned_responses: np.ndarray,
    t_statistics: np.ndarray,
    *,
    index: int,
    sample_count: int,
) -> float:
    """Score evidence at a retained line, rather than elsewhere in its search band."""
    response = float(responses[index])
    continuity = float(continuities[index])
    polarity = float(polarities[index])
    aligned_strengths = np.abs(aligned_responses)
    aligned_response = float(aligned_strengths[index])
    t_statistic = float(t_statistics[index])
    band_peak = float(aligned_strengths.max())
    band_lower_quartile = float(np.percentile(aligned_strengths, 25))
    band_has_local_structure = band_peak >= 1.25 * max(band_lower_quartile, 1e-8)
    small_sample_penalty = 1.5 * max(0.0, (256 - sample_count) / 256)
    required_t_statistic = _MIN_RETURNED_ALIGNED_T_STAT + small_sample_penalty
    if (
        response < _MIN_ABSOLUTE_EDGE_RESPONSE
        or continuity < _MIN_RETURNED_EDGE_CONTINUITY
        or polarity < _MIN_RETURNED_EDGE_POLARITY
        or t_statistic < required_t_statistic
        or not band_has_local_structure
    ):
        return 0.0

    strength = min(1.0, response / _EDGE_STRENGTH_REFERENCE)
    continuity_strength = min(1.0, continuity / 0.60)
    polarity_strength = min(1.0, max(0.0, (polarity - 0.50) / 0.40))
    location_strength = min(1.0, aligned_response / max(band_peak, 1e-8))
    significance_strength = min(1.0, t_statistic / 8.0)
    return float(
        np.mean(
            (
                strength,
                continuity_strength,
                polarity_strength,
                location_strength,
                significance_strength,
            )
        )
    )


def _refine_quad(
    corners: np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray],
    search_radius: float,
    max_refinement_offset: float,
    min_area_ratio: float,
) -> tuple[np.ndarray, float] | None:
    gradient_x, gradient_y = gradients
    lines: list[tuple[np.ndarray, float]] = []
    line_confidences: list[float] = []
    supported_lengths: list[float] = []
    coarse_confidences: list[float] = []
    coarse_supported_lengths: list[float] = []
    perimeter = 0.0
    for index, start in enumerate(corners.astype(np.float64)):
        end = corners[(index + 1) % 4].astype(np.float64)
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= 1e-8:
            return None
        perimeter += float(length)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64) / length

        offsets = _search_offsets(search_radius)
        profile = _parallel_edge_profile(
            gradient_x,
            gradient_y,
            start,
            end,
            normal,
            offsets,
        )
        offset_array = np.asarray(offsets)
        sample_count = min(512, max(24, int(np.ceil(float(length) / 2.0))))
        candidate_confidences = {
            int(candidate_index): _line_edge_confidence(
                *profile,
                index=int(candidate_index),
                sample_count=sample_count,
                offset_fraction=float(abs(offset_array[candidate_index]) / max(search_radius, 1e-8)),
            )
            for candidate_index in range(len(offset_array))
        }
        supported_indices = [
            candidate_index
            for candidate_index, confidence in candidate_confidences.items()
            if confidence > 0.0
        ]
        coarse_confidence = _returned_line_confidence(
            *profile,
            index=0,
            sample_count=sample_count,
        )
        coarse_supported_length = (
            float(profile[1][0] * length) if coarse_confidence > 0.0 else 0.0
        )
        returned_confidence = coarse_confidence
        supported_length = coarse_supported_length
        best_offset = 0.0
        if supported_indices:
            movement_indices = [
                candidate_index
                for candidate_index in supported_indices
                if abs(offset_array[candidate_index]) <= max_refinement_offset + 1e-8
            ]
            aligned_strengths = np.abs(profile[3])
            movement_evidence: tuple[int, ...] | None = None
            if max(profile[0][supported_indices]) >= _MIN_ABSOLUTE_EDGE_RESPONSE:
                peak_index = max(
                    supported_indices,
                    key=lambda candidate_index: aligned_strengths[candidate_index],
                )
                paired_indices = [
                    candidate_index
                    for candidate_index in supported_indices
                    if 1.5 <= abs(offset_array[candidate_index] - offset_array[peak_index]) <= 6.0
                    and profile[3][candidate_index] * profile[3][peak_index] < 0.0
                ]
                paired_index = (
                    max(paired_indices, key=lambda candidate_index: aligned_strengths[candidate_index])
                    if paired_indices
                    else None
                )
                if (
                    paired_index is not None
                    and aligned_strengths[paired_index] >= 0.75 * aligned_strengths[peak_index]
                ):
                    best_offset = float(
                        0.5 * (offset_array[peak_index] + offset_array[paired_index])
                    )
                    movement_evidence = (peak_index, paired_index)

            if movement_evidence is None and movement_indices:
                peak_index = max(
                    movement_indices,
                    key=lambda candidate_index: aligned_strengths[candidate_index],
                )
                competing_indices = [
                    candidate_index
                    for candidate_index in movement_indices
                    if abs(offset_array[candidate_index] - offset_array[peak_index]) > 2.0
                ]
                competing_response = max(
                    [float(aligned_strengths[0])]
                    + [float(aligned_strengths[candidate_index]) for candidate_index in competing_indices]
                )
                if (
                    profile[0][peak_index] >= 0.10
                    and aligned_strengths[peak_index] >= 1.25 * max(competing_response, 1e-8)
                ):
                    best_offset = float(offset_array[peak_index])
                    movement_evidence = (peak_index,)

            if movement_evidence is not None:
                returned_confidence = min(
                    candidate_confidences[candidate_index]
                    for candidate_index in movement_evidence
                )
                supported_length = float(
                    min(profile[1][candidate_index] for candidate_index in movement_evidence)
                    * length
                )

        lines.append((normal, float(normal @ start + best_offset)))
        line_confidences.append(returned_confidence)
        supported_lengths.append(supported_length)
        coarse_confidences.append(coarse_confidence)
        coarse_supported_lengths.append(coarse_supported_length)

    intersections = []
    for index in range(4):
        corner = _line_intersection(lines[index - 1], lines[index])
        if corner is None:
            return None
        intersections.append(corner)
    refined = _valid_quad(np.asarray(intersections), gradient_x.shape, min_area_ratio)
    if refined is None:
        refined = corners.astype(np.float32)
        line_confidences = coarse_confidences
        supported_lengths = coarse_supported_lengths
    mean_displacement = float(np.linalg.norm(refined - corners, axis=1).mean())
    if mean_displacement > 1.25 * search_radius:
        refined = corners.astype(np.float32)
        line_confidences = coarse_confidences
        supported_lengths = coarse_supported_lengths
    line_confidences.sort()
    edge_confidence = float(line_confidences[1])
    if sum(supported_lengths) / max(perimeter, 1e-8) < _MIN_SOURCE_SUPPORT_RATIO:
        edge_confidence = 0.0
    return refined, edge_confidence


def _support_line_quad(
    hull: np.ndarray,
    image_shape: tuple[int, int],
    min_area_ratio: float,
) -> np.ndarray | None:
    reduced = _reduced_hull(hull)
    if reduced is None:
        return None

    hull_points = hull.reshape(-1, 2).astype(np.float64)
    reduced_indices = [int(np.argmin(np.linalg.norm(hull_points - point, axis=1))) for point in reduced]
    if len(set(reduced_indices)) != len(reduced_indices):
        return None

    center = hull_points.mean(axis=0)
    edge_candidates: list[tuple[float, int, tuple[np.ndarray, float]]] = []
    for index, start in enumerate(reduced_indices):
        next_index = (index + 1) % len(reduced_indices)
        end = reduced_indices[next_index]
        side_points = _cyclic_segment(hull_points, start, end)
        line = _support_line(side_points, hull_points, center)
        if line is None:
            continue
        length = float(np.linalg.norm(reduced[next_index] - reduced[index]))
        edge_candidates.append((length, index, line))
    if len(edge_candidates) < 4:
        return None

    dominant_edges = sorted(
        sorted(edge_candidates, key=lambda candidate: candidate[0], reverse=True)[:4],
        key=lambda candidate: candidate[1],
    )
    lines = [candidate[2] for candidate in dominant_edges]

    intersections = []
    for index in range(4):
        corner = _line_intersection(lines[index - 1], lines[index])
        if corner is None:
            return None
        intersections.append(corner)
    return _valid_quad(np.asarray(intersections), image_shape, min_area_ratio)


def _quad_for_component(
    component: np.ndarray,
    image_shape: tuple[int, int],
    min_area_ratio: float,
) -> tuple[np.ndarray, float] | None:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    if len(hull) < 4:
        return None
    perimeter = cv2.arcLength(hull, True)
    structural_polygon = cv2.approxPolyDP(hull, _STRUCTURE_EPSILON_RATIO * perimeter, True)
    if len(structural_polygon) < 4:
        return None

    hull_area = abs(cv2.contourArea(hull))
    if hull_area <= 0.0:
        return None

    candidates = [
        (True, _polygon_quad(hull, image_shape, min_area_ratio)),
        (False, _support_line_quad(hull, image_shape, min_area_ratio)),
    ]
    scored: list[tuple[float, float, np.ndarray]] = []
    for is_polygon, corners in candidates:
        if corners is None:
            continue
        quad_area = abs(cv2.contourArea(corners))
        if quad_area <= 0.0:
            continue
        side_coverage = _minimum_side_coverage(contour, corners)
        side_run_ratio = _minimum_side_run_ratio(contour, corners)
        if (
            side_coverage < _MIN_SIDE_COVERAGE
            or side_run_ratio < _MIN_SIDE_RUN_PERIMETER_RATIO
        ):
            continue
        area_agreement = min(hull_area, quad_area) / max(hull_area, quad_area)
        geometric_confidence = min(area_agreement, side_coverage)
        # Direct four-vertex fits avoid support-line extrapolation. Prefer one
        # when its fidelity is within a small calibration margin, while still
        # allowing a materially better support-line candidate to win.
        selection_score = area_agreement * side_coverage + (0.050 if is_polygon else 0.0)
        scored.append((selection_score, geometric_confidence, corners))

    if not scored:
        return None
    _, confidence, corners = max(scored, key=lambda candidate: candidate[0])
    return corners, float(confidence)


def quad_from_mask(
    mask: np.ndarray,
    *,
    source_image: np.ndarray | None = None,
    source_color_order: SourceColorOrder | None = None,
    min_area_ratio: float = 0.01,
) -> MaskQuad | None:
    """Recover the largest plausible document quadrilateral from a mask.

    Without ``source_image``, returned corners use mask coordinates. With a
    source image, corners are scaled and gradient-refined in source coordinates.
    Three-channel source images require an explicit ``source_color_order`` of
    ``"RGB"`` or ``"BGR"``; two-dimensional images are treated as grayscale.
    Invalid source images raise ``ValueError``.
    """
    if not 0.0 <= min_area_ratio < 1.0:
        raise ValueError("min_area_ratio must be in [0, 1)")

    if source_image is None:
        if source_color_order is not None:
            raise ValueError("source_color_order requires source_image")
        gradients = None
    else:
        gradients = _gradient_components(source_image, source_color_order)

    binary = _as_binary_mask(mask)
    if binary is None or not np.any(binary):
        return None

    height, width = binary.shape
    image_area = float(height * width)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_indices = sorted(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )

    for component_index in component_indices:
        component_area = int(stats[component_index, cv2.CC_STAT_AREA])
        if component_area / image_area < min_area_ratio:
            break
        component = (labels == component_index).astype(np.uint8)
        fit = _quad_for_component(component, binary.shape, min_area_ratio)
        if fit is None:
            continue
        corners, geometric_confidence = fit
        if geometric_confidence < _MIN_GEOMETRIC_CONFIDENCE:
            continue

        result_area = image_area
        edge_confidence = None
        confidence = geometric_confidence
        if gradients is not None:
            source_height, source_width = gradients[0].shape
            scale = np.array([source_width / width, source_height / height], dtype=np.float32)
            scaled_corners = corners * scale
            scale_max = float(scale.max())
            search_radius = max(8.0, min(16.0, 4.0 * scale_max))
            max_refinement_offset = (
                3.0 if scale_max <= 1.5 else min(16.0, max(8.0, scale_max))
            )
            refinement = _refine_quad(
                scaled_corners,
                gradients,
                search_radius=search_radius,
                max_refinement_offset=max_refinement_offset,
                min_area_ratio=min_area_ratio,
            )
            if refinement is None:
                continue
            corners, edge_confidence = refinement
            if edge_confidence < _MIN_EDGE_CONFIDENCE:
                continue
            confidence = geometric_confidence * edge_confidence
            result_area = float(source_height * source_width)
        area_ratio = abs(cv2.contourArea(corners)) / result_area
        return MaskQuad(
            corners=corners,
            area_ratio=float(area_ratio),
            confidence=float(confidence),
            geometric_confidence=float(geometric_confidence),
            edge_confidence=edge_confidence,
        )
    return None
