"""MIT-licensed OSS Document Scanner contour detector port.

Defaults mirror the application path at ossappscollective/OSS-DocumentScanner
revision 444bd810e77c571d66d2eecb3e13371e722ee538.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from openscaner.adapters.base import AdapterOutput

RESIZE_THRESHOLD = 300
AREA_SCALE_MIN_FACTOR = 0.10
BORDER_SIZE = 10
USE_CHANNEL = -1
CANNY_SIGMA_X = 0.0
CANNY_FACTOR = 2.0
MORPHOLOGY_ANCHOR_SIZE = 4
DILATE_ANCHOR_SIZE = 3
THRESHOLD = 160
THRESHOLD_MAX = 256
MEDIAN_BLUR_VALUE = 9
BILATERAL_FILTER_VALUE = 18
CONTOURS_APPROX_EPSILON_FACTOR = 0.02
EXPECTED_MAX_COSINE = 0.4
EXPECTED_OPTIMAL_MAX_COSINE = 0.3
EXPECTED_AREA_FACTOR = 0.20
MIN_DISTANCE_FROM_BORDER_FACTOR = 0.0
HOUGH_LINES_THRESHOLD = 0
HOUGH_LINES_MIN_LINE_LENGTH = 55
HOUGH_LINES_MAX_LINE_GAP = 0


@dataclass(frozen=True)
class ContourDiagnostics:
    contours_seen: int
    rejected_by_size: int
    rejected_by_shape: int
    rejected_by_border: int
    rejected_by_angle: int
    accepted_contours: int
    selected_area_ratio: float | None
    selected_max_cosine: float | None


@dataclass(frozen=True)
class _Candidate:
    points: np.ndarray
    area: float
    max_cosine: float
    weight: int

    @property
    def sort_factor(self) -> float:
        return self.area + self.weight * (1.0 - self.max_cosine)


@dataclass
class _MutableDiagnostics:
    contours_seen: int = 0
    rejected_by_size: int = 0
    rejected_by_shape: int = 0
    rejected_by_border: int = 0
    rejected_by_angle: int = 0
    accepted_contours: int = 0


def _validate_image(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] < 3 or source.size == 0:
        raise ValueError("image must be a non-empty color image")
    if source.dtype != np.uint8:
        raise ValueError("image must use uint8 values")
    return source[:, :, :3]


def _angle(first: np.ndarray, second: np.ndarray, vertex: np.ndarray) -> float:
    first_delta = first.astype(np.float64) - vertex
    second_delta = second.astype(np.float64) - vertex
    denominator = np.sqrt(
        float(first_delta @ first_delta) * float(second_delta @ second_delta) + 1e-10
    )
    return float(first_delta @ second_delta) / denominator


def _find_squares(
    binary: np.ndarray,
    width: int,
    height: int,
    weight: int,
    diagnostics: _MutableDiagnostics,
) -> list[_Candidate]:
    found_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(found_contours)
    contours.sort(key=cv2.contourArea, reverse=True)
    diagnostics.contours_seen += len(contours)
    margin = BORDER_SIZE
    max_allowed_area = (width - 2 * BORDER_SIZE) * (height - 2 * BORDER_SIZE) * 0.92
    candidates: list[_Candidate] = []

    for contour in contours:
        arc_length = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if (
            arc_length < 100
            or area < width * height * AREA_SCALE_MIN_FACTOR
            or area >= max_allowed_area
        ):
            diagnostics.rejected_by_size += 1
            continue

        approximation = cv2.approxPolyDP(
            contour,
            arc_length * CONTOURS_APPROX_EPSILON_FACTOR,
            True,
        )
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            diagnostics.rejected_by_shape += 1
            continue

        points = approximation.reshape(4, 2)
        if any(
            point[0] < margin
            or point[0] >= width - margin
            or point[1] < margin
            or point[1] >= height - margin
            for point in points
        ):
            diagnostics.rejected_by_border += 1
            continue

        max_cosine = max(
            abs(_angle(points[index % 4], points[index - 2], points[(index - 1) % 4]))
            for index in range(2, 6)
        )
        if max_cosine >= EXPECTED_MAX_COSINE:
            diagnostics.rejected_by_angle += 1
            continue

        candidates.append(
            _Candidate(
                points=points.copy(),
                area=float(area),
                max_cosine=float(max_cosine),
                weight=weight,
            )
        )
        diagnostics.accepted_contours += 1
    return candidates


def _sort_points_like_upstream(points: np.ndarray) -> np.ndarray:
    by_y = points[np.argsort(points[:, 1], kind="stable")].copy()
    by_y[:2] = by_y[:2][np.argsort(by_y[:2, 0], kind="stable")]
    by_y[2:] = by_y[2:][np.argsort(-by_y[2:, 0], kind="stable")]
    return by_y


def _freeze_diagnostics(
    mutable: _MutableDiagnostics,
    selected: _Candidate | None,
    image_area: float,
) -> ContourDiagnostics:
    return ContourDiagnostics(
        contours_seen=mutable.contours_seen,
        rejected_by_size=mutable.rejected_by_size,
        rejected_by_shape=mutable.rejected_by_shape,
        rejected_by_border=mutable.rejected_by_border,
        rejected_by_angle=mutable.rejected_by_angle,
        accepted_contours=mutable.accepted_contours,
        selected_area_ratio=None if selected is None else selected.area / image_area,
        selected_max_cosine=None if selected is None else selected.max_cosine,
    )


def detect_with_diagnostics(
    image: np.ndarray,
    *,
    cpu_threads: int,
) -> tuple[AdapterOutput, ContourDiagnostics]:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    cv2.setNumThreads(cpu_threads)
    source = _validate_image(image)
    source_height, source_width = source.shape[:2]
    largest_dimension = max(source_width, source_height)
    resize_scale = 1.0
    resized = source
    if largest_dimension > RESIZE_THRESHOLD:
        resize_scale = largest_dimension / float(RESIZE_THRESHOLD)
        resized = cv2.resize(
            source,
            (
                int(np.floor(source_width / resize_scale)),
                int(np.floor(source_height / resize_scale)),
            ),
        )
    resized = cv2.copyMakeBorder(
        resized,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    blurred = cv2.medianBlur(resized, MEDIAN_BLUR_VALUE)
    morphology = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MORPHOLOGY_ANCHOR_SIZE, MORPHOLOGY_ANCHOR_SIZE),
    )
    dilation = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (DILATE_ANCHOR_SIZE, DILATE_ANCHOR_SIZE),
    )
    height, width = resized.shape[:2]
    diagnostics = _MutableDiagnostics()
    candidates: list[_Candidate] = []
    weight = 3_000_000

    stop = False
    for channel_index in range(min(resized.shape[2], 3) - 1, -1, -1):
        channel = blurred[:, :, channel_index]
        _, binary = cv2.threshold(channel, THRESHOLD, THRESHOLD_MAX, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morphology)
        binary = cv2.dilate(binary, dilation)
        candidates.extend(_find_squares(binary, width, height, weight, diagnostics))
        weight -= 1
        candidates.sort(key=lambda candidate: candidate.sort_factor, reverse=True)
        if candidates and (
            candidates[0].max_cosine < EXPECTED_OPTIMAL_MAX_COSINE
            and candidates[0].area > width * height * EXPECTED_AREA_FACTOR
        ):
            break

        for threshold in range(60, 9, -10):
            binary = cv2.Canny(
                channel,
                threshold * CANNY_FACTOR,
                CANNY_FACTOR * threshold * 2,
            )
            binary = cv2.dilate(binary, dilation)
            candidates.extend(_find_squares(binary, width, height, weight, diagnostics))
            weight -= 1
            candidates.sort(key=lambda candidate: candidate.sort_factor, reverse=True)
            if candidates and (
                candidates[0].max_cosine < EXPECTED_OPTIMAL_MAX_COSINE
                and candidates[0].area > width * height * EXPECTED_AREA_FACTOR
            ):
                stop = True
                break
        if stop:
            break

    selected = candidates[0] if candidates else None
    frozen = _freeze_diagnostics(diagnostics, selected, float(width * height))
    if selected is None:
        return AdapterOutput(
            corners=None,
            confidence=0.0,
            backend="opencv:cpu",
            diagnostics=asdict(frozen),
        ), frozen

    points = (selected.points.astype(np.float32) - BORDER_SIZE) * resize_scale
    points = _sort_points_like_upstream(points)
    confidence = float(np.clip(1.0 - selected.max_cosine, 0.0, 1.0))
    return AdapterOutput(
        corners=points,
        confidence=confidence,
        backend="opencv:cpu",
        diagnostics=asdict(frozen),
    ), frozen


def run(image, model_dir, cpu_threads):
    del model_dir
    output, _ = detect_with_diagnostics(image, cpu_threads=cpu_threads)
    return output
