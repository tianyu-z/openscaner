"""Source-only feature scoring and verified fusion-policy loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import secrets
import stat
from types import MappingProxyType
from typing import Literal

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX fail-closed path
    _fcntl = None

import cv2
import numpy as np

from openscaner.adapters.base import AdapterUnavailable
from openscaner.fusion.candidates import (
    FusionCandidate,
    _MAXIMUM_CANDIDATES,
    _valid_source_corners,
)
from openscaner.fusion.artifacts import FUSION_POLICY_ARTIFACT
from openscaner.fusion.signals import FusionSignals


_WEIGHT_NAMES = (
    "mask_agreement",
    "corner_response",
    "edge_support",
    "coarse_agreement",
)
_FEATURE_NAMES = frozenset(_WEIGHT_NAMES)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "maximum_candidates",
        "maximum_corner_displacement_ratio",
        "weights",
        "minimum_score",
        "fallback",
    }
)
_POLICY_METADATA_FIELDS = frozenset(
    {"filename", "schema_version", "sha256", "size_bytes"}
)
_POLICY_SCHEMA_VERSION = FUSION_POLICY_ARTIFACT.schema_version
_MANIFEST_FILENAME = "manifest.json"
_MAXIMUM_MANIFEST_BYTES = 4 * 1024 * 1024
_MAXIMUM_POLICY_BYTES = 64 * 1024
_ADAPTER_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?\Z")


def _bounded_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and in [0, 1]") from error
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class FusionFeatures:
    mask_agreement: float
    corner_response: float
    edge_support: float
    coarse_agreement: float
    stability: float

    def __post_init__(self) -> None:
        for name in (
            "mask_agreement",
            "corner_response",
            "edge_support",
            "coarse_agreement",
            "stability",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_float(getattr(self, name), name=name),
            )


@dataclass(frozen=True)
class FusionPolicy:
    schema_version: int
    maximum_candidates: int
    maximum_corner_displacement_ratio: float
    weights: dict[str, float]
    minimum_score: float
    fallback: Literal["docaligner"]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _POLICY_SCHEMA_VERSION
        ):
            raise ValueError(
                f"policy schema_version must be exactly {_POLICY_SCHEMA_VERSION}"
            )
        if isinstance(self.maximum_candidates, bool) or not isinstance(
            self.maximum_candidates, int
        ):
            raise TypeError("maximum_candidates must be an integer")
        if not 1 <= self.maximum_candidates <= _MAXIMUM_CANDIDATES:
            raise ValueError(
                f"maximum_candidates must be in the range [1, {_MAXIMUM_CANDIDATES}]"
            )
        displacement = _bounded_float(
            self.maximum_corner_displacement_ratio,
            name="maximum_corner_displacement_ratio",
        )
        if not isinstance(self.weights, dict) or set(self.weights) != _FEATURE_NAMES:
            raise ValueError(f"weights must contain exactly {sorted(_FEATURE_NAMES)}")
        weights = {
            name: _bounded_float(value, name=f"weights[{name}]")
            for name, value in self.weights.items()
        }
        if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("weights must sum to 1 within 1e-6")
        minimum_score = _bounded_float(self.minimum_score, name="minimum_score")
        if self.fallback != "docaligner":
            raise ValueError("fallback must be exactly 'docaligner'")

        object.__setattr__(self, "maximum_corner_displacement_ratio", displacement)
        object.__setattr__(self, "weights", MappingProxyType(weights))
        object.__setattr__(self, "minimum_score", minimum_score)


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: FusionCandidate
    score: float
    features: FusionFeatures
    fallback_used: bool

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, FusionCandidate):
            raise TypeError("candidate must be a FusionCandidate")
        if not isinstance(self.features, FusionFeatures):
            raise TypeError("features must be FusionFeatures")
        object.__setattr__(self, "score", _bounded_float(self.score, name="score"))
        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used must be a bool")


@dataclass(frozen=True)
class _ScoringContext:
    source: np.ndarray
    edge_magnitude: np.ndarray


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = float(np.exp(value))
    return exponential / (1.0 + exponential)


def _bilinear(array: np.ndarray, x: float, y: float) -> float:
    height, width = array.shape
    x = float(np.clip(x, 0.0, width - 1.0))
    y = float(np.clip(y, 0.0, height - 1.0))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    x_weight = x - x0
    y_weight = y - y0
    top = (1.0 - x_weight) * float(array[y0, x0]) + x_weight * float(
        array[y0, x1]
    )
    bottom = (1.0 - x_weight) * float(array[y1, x0]) + x_weight * float(
        array[y1, x1]
    )
    return (1.0 - y_weight) * top + y_weight * bottom


def _mask_agreement(corners: np.ndarray, probabilities: np.ndarray, shape) -> float:
    height, width = shape[:2]
    mask_height, mask_width = probabilities.shape
    scale = np.asarray(
        [(mask_width - 1) / (width - 1), (mask_height - 1) / (height - 1)],
        dtype=np.float32,
    )
    polygon = np.rint(corners * scale).astype(np.int32)
    inside = np.zeros((mask_height, mask_width), dtype=np.uint8)
    cv2.fillConvexPoly(inside, polygon, 1, cv2.LINE_8)
    inside_values = probabilities[inside != 0]
    outside_values = probabilities[inside == 0]
    if inside_values.size == 0:
        return 0.0
    inside_agreement = float(inside_values.mean(dtype=np.float64))
    outside_agreement = (
        1.0 - float(outside_values.mean(dtype=np.float64))
        if outside_values.size
        else 1.0
    )
    return float(np.clip(0.5 * (inside_agreement + outside_agreement), 0.0, 1.0))


def _corner_response(corners: np.ndarray, heatmaps: np.ndarray, shape) -> float:
    height, width = shape[:2]
    scale = np.asarray(
        [
            (heatmaps.shape[2] - 1) / (width - 1),
            (heatmaps.shape[1] - 1) / (height - 1),
        ],
        dtype=np.float64,
    )
    responses = [
        _sigmoid(
            _bilinear(
                heatmaps[index],
                float(corners[index, 0] * scale[0]),
                float(corners[index, 1] * scale[1]),
            )
        )
        for index in range(4)
    ]
    return float(np.clip(np.mean(responses), 0.0, 1.0))


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError("image must be grayscale or a three-channel BGR image")
    if not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise ValueError("image must contain real numeric values")
    if np.issubdtype(image.dtype, np.integer):
        if np.issubdtype(image.dtype, np.signedinteger) and image.min() < 0:
            raise ValueError("integer image values must be non-negative")
        scale = 1.0 / float(np.iinfo(image.dtype).max)
    else:
        planes = (
            (image,)
            if image.ndim == 2
            else tuple(image[:, :, index] for index in range(3))
        )
        if any(not np.isfinite(plane).all() for plane in planes):
            raise ValueError("image must contain finite numeric values")
        minimum = float(image.min())
        maximum = float(image.max())
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError("floating image values must be in [0, 1] or [0, 255]")
        scale = 1.0 if maximum <= 1.0 else 1.0 / 255.0

    if image.ndim == 2:
        gray = image.astype(np.float32, copy=True)
    elif image.dtype in (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    ):
        native_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = native_gray.astype(np.float32, copy=False)
    else:
        gray = np.empty(image.shape[:2], dtype=np.float32)
        scratch = np.empty_like(gray)
        np.multiply(image[:, :, 0], np.float32(0.114), out=gray, casting="unsafe")
        np.multiply(
            image[:, :, 1],
            np.float32(0.587),
            out=scratch,
            casting="unsafe",
        )
        np.add(gray, scratch, out=gray)
        np.multiply(
            image[:, :, 2],
            np.float32(0.299),
            out=scratch,
            casting="unsafe",
        )
        np.add(gray, scratch, out=gray)
    if scale != 1.0:
        np.multiply(gray, np.float32(scale), out=gray)
    return gray


def _source_image(image) -> np.ndarray:
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("image must be a non-empty image array")
    if source.ndim != 2 and not (source.ndim == 3 and source.shape[2] == 3):
        raise ValueError("image must be grayscale or a three-channel BGR image")
    return source


def _prepare_scoring_context(image) -> _ScoringContext:
    source = _source_image(image)
    gray = cv2.GaussianBlur(_grayscale(source), (3, 3), 0)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    return _ScoringContext(source=source, edge_magnitude=magnitude)


def _edge_support(corners: np.ndarray, magnitude: np.ndarray) -> float:
    side_scores: list[float] = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        direction = end.astype(np.float64) - start.astype(np.float64)
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            side_scores.append(0.0)
            continue
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64) / length
        sample_count = min(256, max(24, int(np.ceil(length))))
        fractions = np.linspace(0.05, 0.95, sample_count, dtype=np.float32)[:, None]
        base = start + fractions * (end - start)
        offset_scores = []
        for offset in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            points = base + normal.astype(np.float32) * offset
            sampled = cv2.remap(
                magnitude,
                points[:, 0].astype(np.float32),
                points[:, 1].astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).reshape(-1)
            offset_scores.append(float(sampled.mean(dtype=np.float64)))
        side_scores.append(float(np.clip(max(offset_scores) / 0.25, 0.0, 1.0)))
    return float(sorted(side_scores, reverse=True)[2])


def _coarse_agreement(
    corners: np.ndarray,
    signals: FusionSignals,
    *,
    height: int,
    width: int,
) -> float:
    coarse_source = signals.docaligner.corners
    if coarse_source is None:
        return 0.0
    coarse = _valid_source_corners(coarse_source, height=height, width=width)
    if coarse is None:
        return 0.0
    mean_distance = float(np.linalg.norm(corners - coarse, axis=1).mean())
    diagonal = float(np.hypot(width, height))
    return float(np.clip(1.0 - mean_distance / diagonal, 0.0, 1.0))


def _stability(corners: np.ndarray) -> float:
    edges = np.roll(corners.astype(np.float64), -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-8):
        return 0.0
    cross_products = np.abs(
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    turn_sines = cross_products / (lengths * np.roll(lengths, -1))
    turn_score = float(np.clip(float(turn_sines.min()) / 0.25, 0.0, 1.0))
    side_score = float(
        np.clip(float(lengths.min() / lengths.max()) / 0.15, 0.0, 1.0)
    )
    diagonals = np.asarray(
        [
            np.linalg.norm(corners[2] - corners[0]),
            np.linalg.norm(corners[3] - corners[1]),
        ],
        dtype=np.float64,
    )
    diagonal_score = float(
        np.clip(
            float(diagonals.min() / max(diagonals.max(), 1e-8)) / 0.25,
            0.0,
            1.0,
        )
    )
    return min(turn_score, side_score, diagonal_score)


def _score_candidate_with_context(
    candidate,
    signals,
    policy,
    context: _ScoringContext,
) -> tuple[float, FusionFeatures]:
    if not isinstance(candidate, FusionCandidate):
        raise TypeError("candidate must be a FusionCandidate")
    if not isinstance(signals, FusionSignals):
        raise TypeError("signals must be FusionSignals")
    if not isinstance(policy, FusionPolicy):
        raise TypeError("policy must be FusionPolicy")
    source = context.source
    height, width = source.shape[:2]
    corners = _valid_source_corners(
        candidate.corners,
        height=height,
        width=width,
    )
    if corners is None:
        features = FusionFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
        return 0.0, features

    probabilities = np.clip(signals.corner_model.mask_probabilities, 0.0, 1.0)
    heatmaps = signals.corner_model.corner_heatmaps
    features = FusionFeatures(
        mask_agreement=_mask_agreement(corners, probabilities, source.shape),
        corner_response=_corner_response(corners, heatmaps, source.shape),
        edge_support=_edge_support(corners, context.edge_magnitude),
        coarse_agreement=_coarse_agreement(
            corners,
            signals,
            height=height,
            width=width,
        ),
        stability=_stability(corners),
    )
    weighted = sum(
        policy.weights[name] * getattr(features, name) for name in _WEIGHT_NAMES
    )
    score = float(np.clip(weighted * features.stability, 0.0, 1.0))
    return score, features


def score_candidate(
    candidate,
    image,
    signals,
    policy,
) -> tuple[float, FusionFeatures]:
    context = _prepare_scoring_context(image)
    return _score_candidate_with_context(candidate, signals, policy, context)


def select_candidate(
    candidates,
    image,
    signals,
    policy,
) -> ScoredCandidate | None:
    if not isinstance(policy, FusionPolicy):
        raise TypeError("policy must be a FusionPolicy")
    if not isinstance(signals, FusionSignals):
        raise TypeError("signals must be FusionSignals")
    source = _source_image(image)
    height, width = source.shape[:2]
    coarse_source = signals.docaligner.corners
    valid_coarse = (
        None
        if coarse_source is None
        else _valid_source_corners(coarse_source, height=height, width=width)
    )
    raw_fallback = (
        None
        if valid_coarse is None
        else FusionCandidate("docaligner", valid_coarse, 0)
    )

    considered = tuple(islice(iter(candidates), policy.maximum_candidates))
    if any(not isinstance(candidate, FusionCandidate) for candidate in considered):
        raise TypeError("candidates must contain only FusionCandidate values")
    if raw_fallback is None:
        fallback_candidate = None
    else:
        fallback_candidate = next(
            (
                candidate
                for candidate in considered
                if candidate.family == "docaligner"
                and np.array_equal(candidate.corners, raw_fallback.corners)
            ),
            None,
        )
        if fallback_candidate is None:
            fallback_candidate = raw_fallback
            if len(considered) == policy.maximum_candidates:
                considered = considered[:-1]

    context = None

    def score_for_selection(
        candidate: FusionCandidate,
    ) -> tuple[float, FusionFeatures]:
        nonlocal context
        if (
            _valid_source_corners(
                candidate.corners,
                height=height,
                width=width,
            )
            is None
        ):
            return 0.0, FusionFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
        if context is None:
            context = _prepare_scoring_context(source)
        return _score_candidate_with_context(candidate, signals, policy, context)

    scored = [
        (
            candidate,
            *score_for_selection(candidate),
        )
        for candidate in considered
    ]
    fallback_scored = next(
        (
            item
            for item in scored
            if item[0] is fallback_candidate
        ),
        None,
    )
    if fallback_candidate is not None and fallback_scored is None:
        fallback_score, fallback_features = score_for_selection(fallback_candidate)
        fallback_scored = (
            fallback_candidate,
            fallback_score,
            fallback_features,
        )
    if not scored:
        if fallback_scored is None:
            return None
        fallback_candidate, fallback_score, fallback_features = fallback_scored
        return ScoredCandidate(
            fallback_candidate,
            fallback_score,
            fallback_features,
            True,
        )

    best_candidate, best_score, best_features = scored[0]
    for candidate, score, features in scored[1:]:
        if score > best_score:
            best_candidate, best_score, best_features = candidate, score, features
    if best_score >= policy.minimum_score:
        return ScoredCandidate(best_candidate, best_score, best_features, False)
    if fallback_scored is None:
        return None
    fallback_candidate, fallback_score, fallback_features = fallback_scored
    return ScoredCandidate(
        fallback_candidate,
        fallback_score,
        fallback_features,
        True,
    )


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _decode_json(payload: bytes, *, description: str):
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        OverflowError,
    ) as error:
        raise AdapterUnavailable(f"fusion policy {description} is invalid") from error


def _snapshot_lexical_directory_path(
    path: Path,
) -> tuple[tuple[int, int], ...]:
    if not path.is_absolute():
        raise AdapterUnavailable(
            "fusion policy model directory must resolve to an absolute path"
        )
    current = Path(path.anchor)
    components = [current]
    for component in path.parts[1:]:
        if component in ("", ".", ".."):
            raise AdapterUnavailable(
                "fusion policy model directory contains an invalid component"
            )
        current = current / component
        components.append(current)

    identities: list[tuple[int, int]] = []
    for component_path in components:
        metadata = os.lstat(component_path)
        if stat.S_ISLNK(metadata.st_mode):
            raise AdapterUnavailable(
                "fusion policy model directory must not contain symlinks"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise AdapterUnavailable(
                "fusion policy model directory path must contain only directories"
            )
        identities.append((metadata.st_dev, metadata.st_ino))
    return tuple(identities)


def _open_model_directory(model_dir: Path) -> int:
    if any(component == ".." for component in model_dir.parts):
        raise AdapterUnavailable(
            "fusion policy model directory must not contain '..'"
        )
    try:
        lexical_path = Path(os.path.abspath(os.fspath(model_dir)))
    except (OSError, TypeError, ValueError) as error:
        raise AdapterUnavailable(
            "fusion policy model directory could not be resolved safely"
        ) from error
    try:
        before_components = _snapshot_lexical_directory_path(lexical_path)
    except AdapterUnavailable:
        raise
    except OSError as error:
        raise AdapterUnavailable(
            "fusion policy model directory could not be resolved safely"
        ) from error
    try:
        resolved_path = model_dir.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise AdapterUnavailable(
            "fusion policy model directory could not be resolved safely"
        ) from error
    if lexical_path != resolved_path:
        raise AdapterUnavailable(
            "fusion policy model directory must not contain symlinks"
        )

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow or not getattr(os, "O_DIRECTORY", 0):
        raise AdapterUnavailable(
            "fusion policy model directory identity cannot be proven"
        )
    descriptor = None
    try:
        descriptor = os.open(resolved_path, directory_flags | no_follow)
        after = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(after.st_mode)
            or before_components[-1] != (after.st_dev, after.st_ino)
        ):
            raise AdapterUnavailable(
                "fusion policy model directory changed while opening"
            )
        after_components = _snapshot_lexical_directory_path(lexical_path)
        if after_components != before_components:
            raise AdapterUnavailable(
                "fusion policy model directory changed while opening"
            )

        opened_path: str | None = None
        get_path = getattr(_fcntl, "F_GETPATH", None) if _fcntl is not None else None
        if get_path is not None:
            raw_path = _fcntl.fcntl(descriptor, get_path, b"\0" * 1024)
            if isinstance(raw_path, bytes) and b"\0" in raw_path:
                opened_path = os.fsdecode(raw_path.split(b"\0", 1)[0])
        elif os.path.exists("/proc/self/fd"):
            opened_path = os.readlink(f"/proc/self/fd/{descriptor}")
        if (
            opened_path is None
            or Path(os.path.normpath(opened_path)) != resolved_path
        ):
            raise AdapterUnavailable(
                "fusion policy model directory identity cannot be proven"
            )
        result = descriptor
        descriptor = None
        return result
    except AdapterUnavailable:
        raise
    except (OSError, ValueError) as error:
        raise AdapterUnavailable(
            "fusion policy model directory could not be opened safely"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_regular_file_at(
    directory_descriptor: int,
    *,
    filename: str,
    description: str,
    maximum_size: int,
    expected_size: int | None = None,
) -> bytes:
    descriptor = None
    try:
        before = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(before.st_mode):
            raise AdapterUnavailable(
                f"fusion policy {description} must not be a symlink"
            )
        if not stat.S_ISREG(before.st_mode):
            raise AdapterUnavailable(
                f"fusion policy {description} is not a regular file"
            )
        if before.st_size > maximum_size:
            raise AdapterUnavailable(f"fusion policy {description} is too large")
        if expected_size is not None and before.st_size != expected_size:
            raise AdapterUnavailable(
                f"fusion policy {description} size mismatch"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(filename, flags, dir_fd=directory_descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (before.st_dev, before.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise AdapterUnavailable(
                f"fusion policy {description} changed while opening"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_size + 1)
        after = os.fstat(descriptor)
        if (
            len(payload) > maximum_size
            or len(payload) != metadata.st_size
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
            or after.st_ctime_ns != metadata.st_ctime_ns
        ):
            raise AdapterUnavailable(
                f"fusion policy {description} changed while reading"
            )
    except AdapterUnavailable:
        raise
    except OSError as error:
        raise AdapterUnavailable(
            f"fusion policy {description} could not be read"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return payload


def _read_manifest(directory_descriptor: int) -> dict[str, object]:
    payload = _read_regular_file_at(
        directory_descriptor,
        filename=_MANIFEST_FILENAME,
        description="manifest",
        maximum_size=_MAXIMUM_MANIFEST_BYTES,
    )
    manifest = _decode_json(payload, description="manifest")
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 3
        or not isinstance(manifest.get("models"), list)
        or not all(isinstance(entry, dict) for entry in manifest["models"])
    ):
        raise AdapterUnavailable("fusion policy manifest schema is incompatible")
    return manifest


def load_verified_policy(model_dir, *, adapter_name) -> tuple[FusionPolicy, str]:
    if not isinstance(adapter_name, str) or not _ADAPTER_NAME.fullmatch(adapter_name):
        raise ValueError("adapter_name must be a safe non-empty adapter identifier")
    directory = Path(model_dir)
    directory_descriptor = _open_model_directory(directory)
    try:
        manifest = _read_manifest(directory_descriptor)
        return _load_policy_from_manifest(
            directory_descriptor,
            manifest=manifest,
            adapter_name=adapter_name,
        )
    finally:
        os.close(directory_descriptor)


def _load_policy_from_manifest(
    directory_descriptor: int,
    *,
    manifest: dict[str, object],
    adapter_name: str,
) -> tuple[FusionPolicy, str]:
    matches = [
        entry for entry in manifest["models"] if entry.get("adapter") == adapter_name
    ]
    if len(matches) != 1:
        raise AdapterUnavailable(
            "fusion policy manifest adapter entry is absent or duplicated"
        )
    metadata = matches[0].get("fusion_policy")
    if not isinstance(metadata, dict) or set(metadata) != _POLICY_METADATA_FIELDS:
        raise AdapterUnavailable("fusion policy artifact metadata is invalid")

    expected_filename = (
        FUSION_POLICY_ARTIFACT.filename
        if adapter_name == FUSION_POLICY_ARTIFACT.adapter
        else f"{adapter_name}_policy.json"
    )
    filename = metadata.get("filename")
    digest = metadata.get("sha256")
    size_bytes = metadata.get("size_bytes")
    if (
        filename != expected_filename
        or not isinstance(filename, str)
        or Path(filename).name != filename
    ):
        raise AdapterUnavailable("fusion policy artifact filename is invalid")
    if (
        type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != _POLICY_SCHEMA_VERSION
    ):
        raise AdapterUnavailable("fusion policy artifact schema is incompatible")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise AdapterUnavailable("fusion policy artifact SHA-256 is invalid")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or not 1 <= size_bytes <= _MAXIMUM_POLICY_BYTES
    ):
        raise AdapterUnavailable("fusion policy artifact size is invalid")
    payload = _read_regular_file_at(
        directory_descriptor,
        filename=filename,
        description="artifact",
        maximum_size=_MAXIMUM_POLICY_BYTES,
        expected_size=size_bytes,
    )
    actual_digest = hashlib.sha256(payload).hexdigest()
    if not secrets.compare_digest(actual_digest, digest):
        raise AdapterUnavailable("fusion policy artifact SHA-256 checksum mismatch")
    document = _decode_json(payload, description="artifact")
    if not isinstance(document, dict) or set(document) != _POLICY_FIELDS:
        raise AdapterUnavailable("fusion policy artifact fields are incompatible")
    try:
        policy = FusionPolicy(**document)
    except (TypeError, ValueError, OverflowError) as error:
        raise AdapterUnavailable("fusion policy artifact values are incompatible") from error
    return policy, actual_digest


__all__ = [
    "FusionFeatures",
    "FusionPolicy",
    "ScoredCandidate",
    "load_verified_policy",
    "score_candidate",
    "select_candidate",
]
