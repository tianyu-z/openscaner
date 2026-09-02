"""Verified full-resolution SmartDoc data for local corner refinement."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import secrets
import stat
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from openscaner.adapters import docaligner
from openscaner.fusion.artifacts import ModelArtifactIdentity
from openscaner.refiner.geometry import (
    HEATMAP_SIZE,
    PATCH_SIZE,
    PatchTransform,
    _validate_ordered_quad,
    extract_corner_patches,
)
from openscaner.training.smartdoc import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SMARTDOC_ARCHIVE_SHA256,
    SMARTDOC_TRAIN_BACKGROUNDS,
    SMARTDOC_VALIDATION_BACKGROUND,
    SMARTDOC_VERSION,
    SmartDocRecord,
    _apply_blur,
    _apply_photometric_augmentation,
    _apply_spatial_shadow,
    assert_protected_hashes_absent,
    load_smartdoc_records,
    read_verified_smartdoc_image,
    split_smartdoc_records,
    validate_smartdoc_record_paths,
    validate_smartdoc_source_paths,
    verify_smartdoc_markers,
    verify_smartdoc_source,
)

PATCH_RADIUS_RATIOS = (0.18, 0.24, 0.30)
VIEW_NAMES = ("clean", "augmented")
CACHE_SCHEMA_VERSION = 1
MAX_CACHE_RECORDS = 20_000
MAX_CACHE_BYTES = 32 * 1024 * 1024
DEFAULT_SEED = 20260825

_SPLITS = ("train", "validation")


def _immutable_quad(value: object, name: str) -> np.ndarray:
    try:
        converted = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric quadrilateral") from error
    if converted.shape != (4, 2):
        raise ValueError(f"{name} must have shape (4, 2)")
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(converted.tobytes(order="C"), dtype=np.float32).reshape(4, 2)


def _lower_hex(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _record_identity(record: SmartDocRecord) -> tuple[str, str, int]:
    return record.background, record.sequence, record.frame_index


def augmentation_identity(
    record: SmartDocRecord,
    view: str,
    *,
    seed: int = DEFAULT_SEED,
) -> str:
    """Return the root-independent deterministic identity for one generated view."""
    if not isinstance(record, SmartDocRecord):
        raise TypeError("record must be a SmartDocRecord")
    if view not in VIEW_NAMES:
        raise ValueError(f"view must be one of {VIEW_NAMES}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    payload = "\0".join(
        (
            str(seed),
            record.background,
            record.sequence,
            str(record.frame_index),
            view,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_generator(identity: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(bytes.fromhex(identity)[:16], "little"))


def _apply_local_corner_occluder(
    image: np.ndarray,
    corners: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Cover one uniformly selected corner with color sampled from a local ring."""
    result = image.copy()
    corner_index = int(generator.integers(0, 4))
    corner = corners[corner_index].astype(np.float64)
    next_length = float(np.linalg.norm(corners[(corner_index + 1) % 4] - corner))
    previous_length = float(np.linalg.norm(corners[(corner_index - 1) % 4] - corner))
    radius_ratio = float(generator.uniform(0.06, 0.12))
    radius = max(1, int(round(radius_ratio * min(next_length, previous_length))))

    height, width = result.shape[:2]
    minimum_x = max(0, int(np.floor(corner[0] - 2.0 * radius)))
    maximum_x = min(width - 1, int(np.ceil(corner[0] + 2.0 * radius)))
    minimum_y = max(0, int(np.floor(corner[1] - 2.0 * radius)))
    maximum_y = min(height - 1, int(np.ceil(corner[1] + 2.0 * radius)))
    yy, xx = np.mgrid[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
    squared = (xx - corner[0]) ** 2 + (yy - corner[1]) ** 2
    ring = np.column_stack(
        np.nonzero(
            (squared >= (1.25 * radius) ** 2)
            & (squared <= (2.0 * radius) ** 2)
        )
    )
    if len(ring) == 0:
        sample_x = int(np.clip(round(corner[0] + radius), 0, width - 1))
        sample_y = int(np.clip(round(corner[1]), 0, height - 1))
    else:
        selected = ring[int(generator.integers(0, len(ring)))]
        sample_y = minimum_y + int(selected[0])
        sample_x = minimum_x + int(selected[1])
    fill = tuple(int(channel) for channel in image[sample_y, sample_x])
    center = tuple(np.rint(corner).astype(np.int32))
    cv2.circle(result, center, radius, fill, thickness=-1, lineType=cv2.LINE_8)
    return result


def _apply_refiner_augmentation(
    image: np.ndarray,
    corners: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    augmented = _apply_photometric_augmentation(image, generator)
    augmented = _apply_blur(augmented, generator)
    augmented = _apply_spatial_shadow(augmented, generator)
    if float(generator.random()) < 0.5:
        augmented = _apply_local_corner_occluder(augmented, corners, generator)
    return augmented


def generate_refiner_view(
    record: SmartDocRecord,
    view: str,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, str]:
    """Regenerate one full-resolution deterministic cache view."""
    identity = augmentation_identity(record, view, seed=seed)
    source = read_verified_smartdoc_image(record)
    _validate_ordered_quad(record.corners, source.shape)
    if view == "clean":
        return source.copy(), identity
    generated = _apply_refiner_augmentation(
        source,
        record.corners,
        _identity_generator(identity),
    )
    if generated.dtype != np.uint8 or generated.shape != source.shape:
        raise RuntimeError("refiner augmentation changed the image representation")
    return generated.copy(), identity


def _png_bytes(image: np.ndarray) -> bytes:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8 BGR pixels")
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("image must be a non-empty uint8 BGR image")
    success, encoded = cv2.imencode(
        ".png",
        image,
        (cv2.IMWRITE_PNG_COMPRESSION, 3),
    )
    if not success:
        raise RuntimeError("unable to losslessly encode refiner view")
    return encoded.tobytes()


def refiner_cache_key(
    image: np.ndarray,
    *,
    seed: int,
    augmentation_identity: str,
    view: str,
    model_identity: ModelArtifactIdentity,
) -> str:
    """Bind exact generated PNG bytes and all inference-affecting identities."""
    if view not in VIEW_NAMES:
        raise ValueError(f"view must be one of {VIEW_NAMES}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    _lower_hex(augmentation_identity, "augmentation_identity")
    if not isinstance(model_identity, ModelArtifactIdentity):
        raise TypeError("model_identity must be a ModelArtifactIdentity")
    assert_protected_hashes_absent((model_identity.sha256,))
    metadata = {
        "augmentation_identity": augmentation_identity,
        "model": {
            "filename": model_identity.filename,
            "sha256": model_identity.sha256,
            "size_bytes": model_identity.size_bytes,
        },
        "seed": seed,
        "view": view,
    }
    encoded_metadata = json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(encoded_metadata).to_bytes(8, "big"))
    digest.update(encoded_metadata)
    digest.update(_png_bytes(image))
    return digest.hexdigest()


@dataclass(frozen=True, eq=False, slots=True)
class RefinerCacheRecord:
    split: str
    view: str
    background: str
    sequence: str
    frame_index: int
    cache_key: str
    coarse_corners: np.ndarray | None = field(compare=False)
    ground_truth_corners: np.ndarray = field(compare=False)
    original_image_sha256: str
    original_image_size_bytes: int
    augmentation_identity: str
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.split, str):
            raise TypeError("split must be a string")
        if self.split not in _SPLITS:
            raise ValueError("split must be train or validation")
        if not isinstance(self.view, str):
            raise TypeError("view must be a string")
        if self.view not in VIEW_NAMES:
            raise ValueError(f"view must be one of {VIEW_NAMES}")
        if not isinstance(self.background, str):
            raise TypeError("background must be a string")
        expected_split = (
            "train"
            if self.background in SMARTDOC_TRAIN_BACKGROUNDS
            else "validation"
            if self.background == SMARTDOC_VALIDATION_BACKGROUND
            else None
        )
        if expected_split is None or self.split != expected_split:
            raise ValueError("SmartDoc background does not match split")
        if not isinstance(self.sequence, str) or not self.sequence:
            raise ValueError("sequence must be a non-empty string")
        if (
            self.sequence in {".", ".."}
            or "/" in self.sequence
            or "\\" in self.sequence
            or "\0" in self.sequence
        ):
            raise ValueError("sequence must not contain path concepts")
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        cache_key = _lower_hex(self.cache_key, "cache_key")
        image_sha256 = _lower_hex(
            self.original_image_sha256, "original_image_sha256"
        )
        identity = _lower_hex(self.augmentation_identity, "augmentation_identity")
        image_size = _positive_int(
            self.original_image_size_bytes, "original_image_size_bytes"
        )
        ground_truth = _immutable_quad(
            self.ground_truth_corners, "ground_truth_corners"
        )
        coarse = (
            None
            if self.coarse_corners is None
            else _immutable_quad(self.coarse_corners, "coarse_corners")
        )
        _validate_ordered_quad(ground_truth)
        if coarse is not None:
            _validate_ordered_quad(coarse)
        if (coarse is None) != (self.exclusion_reason is not None):
            raise ValueError(
                "exclusion_reason must be present exactly for failed detections"
            )
        if self.exclusion_reason is not None and self.exclusion_reason not in {
            "missing_detection",
            "invalid_detection",
        }:
            raise ValueError("unknown exclusion_reason")
        assert_protected_hashes_absent((cache_key, image_sha256, identity))
        object.__setattr__(self, "cache_key", cache_key)
        object.__setattr__(self, "original_image_sha256", image_sha256)
        object.__setattr__(self, "original_image_size_bytes", image_size)
        object.__setattr__(self, "augmentation_identity", identity)
        object.__setattr__(self, "ground_truth_corners", ground_truth)
        object.__setattr__(self, "coarse_corners", coarse)

    def __reduce__(self):
        return (
            RefinerCacheRecord,
            (
                self.split,
                self.view,
                self.background,
                self.sequence,
                self.frame_index,
                self.cache_key,
                self.coarse_corners,
                self.ground_truth_corners,
                self.original_image_sha256,
                self.original_image_size_bytes,
                self.augmentation_identity,
                self.exclusion_reason,
            ),
        )


@dataclass(frozen=True, slots=True)
class RefinerCache:
    smartdoc_archive_sha256: str
    smartdoc_version: str
    stride: int
    seed: int
    model_identity: ModelArtifactIdentity
    records: tuple[RefinerCacheRecord, ...]
    schema_version: int = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != CACHE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported refiner cache schema version")
        archive_sha256 = _lower_hex(
            self.smartdoc_archive_sha256, "smartdoc_archive_sha256"
        )
        if self.smartdoc_version != "2.0.0":
            raise ValueError("unsupported SmartDoc version")
        stride = _positive_int(self.stride, "stride")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.model_identity, ModelArtifactIdentity):
            raise TypeError("model_identity must be a ModelArtifactIdentity")
        if self.model_identity.adapter != "docaligner":
            raise ValueError("model identity must identify DocAligner")
        records = tuple(self.records)
        if not records:
            raise ValueError("refiner cache records must not be empty")
        if len(records) > MAX_CACHE_RECORDS:
            raise ValueError("refiner cache has too many records")
        if any(not isinstance(record, RefinerCacheRecord) for record in records):
            raise TypeError("records must contain RefinerCacheRecord values")
        records = tuple(
            sorted(
                records,
                key=lambda record: (
                    record.background,
                    record.sequence,
                    record.frame_index,
                    VIEW_NAMES.index(record.view),
                ),
            )
        )
        identities: set[tuple[str, str, int, str]] = set()
        keys: set[str] = set()
        views: dict[tuple[str, str, int], set[str]] = defaultdict(set)
        source_bindings: dict[
            tuple[str, str, int], tuple[str, str, int, bytes]
        ] = {}
        for record in records:
            if record.frame_index % stride != 0:
                raise ValueError("cache record was not selected by the declared stride")
            identity = (
                record.background,
                record.sequence,
                record.frame_index,
                record.view,
            )
            if identity in identities:
                raise ValueError("duplicate refiner cache record")
            if record.cache_key in keys:
                raise ValueError("duplicate refiner cache key")
            identities.add(identity)
            keys.add(record.cache_key)
            views[identity[:3]].add(record.view)
            source_binding = (
                record.split,
                record.original_image_sha256,
                record.original_image_size_bytes,
                record.ground_truth_corners.tobytes(),
            )
            previous_binding = source_bindings.setdefault(
                identity[:3], source_binding
            )
            if previous_binding != source_binding:
                raise ValueError(
                    "clean and augmented cache pair source is inconsistent"
                )
            _validate_ordered_quad(record.ground_truth_corners)
            if record.coarse_corners is not None:
                _validate_ordered_quad(record.coarse_corners)
            expected_augmentation_identity = hashlib.sha256(
                "\0".join(
                    (
                        str(self.seed),
                        record.background,
                        record.sequence,
                        str(record.frame_index),
                        record.view,
                    )
                ).encode("utf-8")
            ).hexdigest()
            if record.augmentation_identity != expected_augmentation_identity:
                raise ValueError("augmentation identity differs from cache metadata")
        if any(value != set(VIEW_NAMES) for value in views.values()):
            raise ValueError("each SmartDoc record must have exactly two cache views")
        assert_protected_hashes_absent(
            (archive_sha256, self.model_identity.sha256)
        )
        object.__setattr__(self, "smartdoc_archive_sha256", archive_sha256)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "records", records)

    def __reduce__(self):
        return (
            RefinerCache,
            (
                self.smartdoc_archive_sha256,
                self.smartdoc_version,
                self.stride,
                self.seed,
                self.model_identity,
                self.records,
                self.schema_version,
            ),
        )


def _split_for_background(background: str) -> str:
    if background in SMARTDOC_TRAIN_BACKGROUNDS:
        return "train"
    if background == SMARTDOC_VALIDATION_BACKGROUND:
        return "validation"
    raise ValueError(f"unknown SmartDoc background: {background}")


def build_refiner_cache(
    records: Sequence[SmartDocRecord],
    *,
    smartdoc_archive_sha256: str,
    smartdoc_version: str,
    stride: int,
    seed: int = DEFAULT_SEED,
    session: object,
    model_identity: ModelArtifactIdentity,
    infer_fn: Callable[
        [object, np.ndarray], Any
    ] = docaligner._infer_docaligner_signals,
) -> RefinerCache:
    """Build an immutable cache with exactly one DocAligner inference per view."""
    archive_sha256 = _lower_hex(
        smartdoc_archive_sha256, "smartdoc_archive_sha256"
    )
    if smartdoc_version != "2.0.0":
        raise ValueError("unsupported SmartDoc version")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(model_identity, ModelArtifactIdentity):
        raise TypeError("model_identity must be a ModelArtifactIdentity")
    if model_identity.adapter != "docaligner":
        raise ValueError("model identity must identify DocAligner")
    assert_protected_hashes_absent((archive_sha256, model_identity.sha256))
    stored = tuple(records)
    if not stored:
        raise ValueError("records must be non-empty")
    if any(not isinstance(record, SmartDocRecord) for record in stored):
        raise TypeError("records must contain SmartDocRecord values")
    validate_smartdoc_record_paths(stored)
    if len(stored) * len(VIEW_NAMES) > MAX_CACHE_RECORDS:
        raise ValueError("refiner cache would contain too many records")
    assert_protected_hashes_absent(
        record.image_sha256
        for record in stored
        if record.image_sha256 is not None
    )
    selected_stride = _positive_int(stride, "stride")
    if any(record.frame_index % selected_stride != 0 for record in stored):
        raise ValueError("records must be selected by the declared stride")
    identities = [_record_identity(record) for record in stored]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate SmartDoc record")
    ordered = sorted(stored, key=_record_identity)
    cached: list[RefinerCacheRecord] = []
    for record in ordered:
        split = _split_for_background(record.background)
        if record.image_sha256 is None or record.image_size_bytes is None:
            raise ValueError("SmartDoc record is missing verified image provenance")
        for view in VIEW_NAMES:
            image, identity = generate_refiner_view(record, view, seed=seed)
            key = refiner_cache_key(
                image,
                seed=seed,
                augmentation_identity=identity,
                view=view,
                model_identity=model_identity,
            )
            signals = infer_fn(session, image)
            raw_corners = getattr(signals, "corners", None)
            exclusion_reason: str | None = None
            coarse = docaligner.accepted_docaligner_corners(
                raw_corners,
                image.shape,
            )
            if raw_corners is None:
                exclusion_reason = "missing_detection"
            elif coarse is None:
                exclusion_reason = "invalid_detection"
            else:
                try:
                    _validate_ordered_quad(coarse, image.shape)
                except (TypeError, ValueError):
                    coarse = None
                    exclusion_reason = "invalid_detection"
            cached.append(
                RefinerCacheRecord(
                    split=split,
                    view=view,
                    background=record.background,
                    sequence=record.sequence,
                    frame_index=record.frame_index,
                    cache_key=key,
                    coarse_corners=coarse,
                    ground_truth_corners=record.corners,
                    original_image_sha256=record.image_sha256,
                    original_image_size_bytes=record.image_size_bytes,
                    augmentation_identity=identity,
                    exclusion_reason=exclusion_reason,
                )
            )
    return RefinerCache(
        smartdoc_archive_sha256=archive_sha256,
        smartdoc_version=smartdoc_version,
        stride=selected_stride,
        seed=seed,
        model_identity=model_identity,
        records=tuple(cached),
    )


def _model_document(identity: ModelArtifactIdentity) -> dict[str, object]:
    return {
        "adapter": identity.adapter,
        "filename": identity.filename,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }


def _record_document(record: RefinerCacheRecord) -> dict[str, object]:
    return {
        "augmentation_identity": record.augmentation_identity,
        "background": record.background,
        "cache_key": record.cache_key,
        "coarse_corners": (
            None if record.coarse_corners is None else record.coarse_corners.tolist()
        ),
        "exclusion_reason": record.exclusion_reason,
        "frame_index": record.frame_index,
        "ground_truth_corners": record.ground_truth_corners.tolist(),
        "original_image_sha256": record.original_image_sha256,
        "original_image_size_bytes": record.original_image_size_bytes,
        "sequence": record.sequence,
        "split": record.split,
        "view": record.view,
    }


def _cache_document(cache: RefinerCache) -> dict[str, object]:
    if not isinstance(cache, RefinerCache):
        raise TypeError("cache must be a RefinerCache")
    return {
        "docaligner_model": _model_document(cache.model_identity),
        "records": [_record_document(record) for record in cache.records],
        "schema_version": cache.schema_version,
        "seed": cache.seed,
        "smartdoc_archive_sha256": cache.smartdoc_archive_sha256,
        "smartdoc_version": cache.smartdoc_version,
        "stride": cache.stride,
    }


def refiner_cache_bytes(cache: RefinerCache) -> bytes:
    """Serialize a cache as canonical finite sorted JSON with one trailing newline."""
    payload = (
        json.dumps(
            _cache_document(cache),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    if len(payload) > MAX_CACHE_BYTES:
        raise ValueError("refiner cache exceeds maximum serialized size")
    return payload


def _cache_absolute_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if not absolute.name:
        raise ValueError("cache target must name a file")
    return absolute


def _open_cache_directory(path: Path, *, create: bool) -> int:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open("/", directory_flags)
    except OSError as error:
        raise ValueError("unable to open cache directory") from error
    try:
        for component in path.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=descriptor,
                )
            except OSError as error:
                if not create or error.errno != errno.ENOENT:
                    raise ValueError(
                        "cache path ancestor changed or is not a directory"
                    ) from error
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as mkdir_error:
                    raise ValueError("unable to create cache directory") from mkdir_error
                try:
                    next_descriptor = os.open(
                        component,
                        directory_flags,
                        dir_fd=descriptor,
                    )
                except OSError as open_error:
                    raise ValueError(
                        "cache path ancestor changed or is not a directory"
                    ) from open_error
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _cache_target_status(
    parent_descriptor: int,
    name: str,
    *,
    allow_missing: bool,
) -> os.stat_result | None:
    try:
        status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise ValueError("cache target must be a regular file") from None
    except OSError as error:
        raise ValueError("unable to inspect cache target") from error
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise ValueError(
            "cache target must be a singly-linked regular non-symlinked file"
        )
    return status


def _assert_cache_parent_unchanged(
    parent: Path,
    expected: os.stat_result,
) -> None:
    descriptor = _open_cache_directory(parent, create=False)
    try:
        actual = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("cache path ancestor changed during operation")


def save_refiner_cache(
    cache: RefinerCache,
    path: Path,
    *,
    phase_hook: Callable[[str, Path], None] | None = None,
) -> None:
    """Atomically publish canonical cache bytes from the destination directory."""
    payload = refiner_cache_bytes(cache)
    target = _cache_absolute_path(Path(path))
    parent_descriptor = _open_cache_directory(target.parent, create=True)
    temporary_name: str | None = None
    temporary_descriptor = -1
    try:
        parent_status = os.fstat(parent_descriptor)
        original_target = _cache_target_status(
            parent_descriptor,
            target.name,
            allow_missing=True,
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for _ in range(16):
            candidate = f".{target.name}.{secrets.token_hex(8)}.tmp"
            try:
                temporary_descriptor = os.open(
                    candidate,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_name is None:
            raise RuntimeError("unable to allocate unique cache temporary file")
        with os.fdopen(temporary_descriptor, "wb") as stream:
            temporary_descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if phase_hook is not None:
            phase_hook("before_replace", target.parent / temporary_name)
        _assert_cache_parent_unchanged(target.parent, parent_status)
        current_target = _cache_target_status(
            parent_descriptor,
            target.name,
            allow_missing=True,
        )
        original_identity = (
            None
            if original_target is None
            else (original_target.st_dev, original_target.st_ino)
        )
        current_identity = (
            None
            if current_target is None
            else (current_target.st_dev, current_target.st_ino)
        )
        if current_identity != original_identity:
            raise ValueError("cache target changed during publication")
        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_document_keys(
    document: dict[str, object], expected: set[str], description: str
) -> None:
    if set(document) != expected:
        raise ValueError(f"malformed {description} keys")


def _read_cache_payload(path: Path) -> bytes:
    source = _cache_absolute_path(path)
    parent_descriptor = _open_cache_directory(source.parent, create=False)
    parent_status = os.fstat(parent_descriptor)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                source.name,
                flags,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise ValueError(
                "cache target must be a regular non-symlinked file"
            ) from error
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                "cache target must be a singly-linked regular non-symlinked file"
            )
        if before.st_size > MAX_CACHE_BYTES:
            raise ValueError("cache target is oversized")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(MAX_CACHE_BYTES + 1)
            after = os.fstat(stream.fileno())
        target_after = _cache_target_status(
            parent_descriptor,
            source.name,
            allow_missing=False,
        )
        _assert_cache_parent_unchanged(source.parent, parent_status)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_size != len(payload)
            or len(payload) > MAX_CACHE_BYTES
            or target_after is None
            or target_after.st_dev != before.st_dev
            or target_after.st_ino != before.st_ino
            or target_after.st_size != before.st_size
        ):
            raise ValueError("cache target changed during read or is oversized")
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _parse_refiner_cache(payload: bytes) -> RefinerCache:
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("malformed refiner cache JSON") from error
    if not isinstance(document, dict):
        raise ValueError("refiner cache JSON must be an object")
    _require_document_keys(
        document,
        {
            "docaligner_model",
            "records",
            "schema_version",
            "seed",
            "smartdoc_archive_sha256",
            "smartdoc_version",
            "stride",
        },
        "refiner cache",
    )
    model_document = document["docaligner_model"]
    if not isinstance(model_document, dict):
        raise ValueError("malformed DocAligner model identity")
    _require_document_keys(
        model_document,
        {"adapter", "filename", "sha256", "size_bytes"},
        "DocAligner model identity",
    )
    records_document = document["records"]
    if not isinstance(records_document, list):
        raise ValueError("refiner cache records must be a list")
    if len(records_document) > MAX_CACHE_RECORDS:
        raise ValueError("refiner cache has too many records")
    records: list[RefinerCacheRecord] = []
    record_keys = {
        "augmentation_identity",
        "background",
        "cache_key",
        "coarse_corners",
        "exclusion_reason",
        "frame_index",
        "ground_truth_corners",
        "original_image_sha256",
        "original_image_size_bytes",
        "sequence",
        "split",
        "view",
    }
    for item in records_document:
        if not isinstance(item, dict):
            raise ValueError("malformed refiner cache record")
        _require_document_keys(item, record_keys, "refiner cache record")
        try:
            records.append(RefinerCacheRecord(**item))
        except TypeError as error:
            raise ValueError("malformed refiner cache record") from error
    try:
        model_identity = ModelArtifactIdentity(**model_document)
        return RefinerCache(
            smartdoc_archive_sha256=document["smartdoc_archive_sha256"],
            smartdoc_version=document["smartdoc_version"],
            stride=document["stride"],
            seed=document["seed"],
            model_identity=model_identity,
            records=tuple(records),
            schema_version=document["schema_version"],
        )
    except TypeError as error:
        raise ValueError("malformed refiner cache metadata") from error


def load_verified_refiner_cache(
    path: Path,
    records: Sequence[SmartDocRecord],
    *,
    smartdoc_archive_sha256: str,
    smartdoc_version: str,
    stride: int,
    seed: int = DEFAULT_SEED,
    model_identity: ModelArtifactIdentity,
    backgrounds: Sequence[str] | None = None,
    payload: bytes | None = None,
) -> RefinerCache:
    """Load a cache only after replaying and re-hashing selected verified views.

    The optional background filter is intentionally applied after full raw-cache
    schema/canonical-byte validation and before any view is regenerated.  This
    lets validation-only consumers bind the complete cache artifact without
    decoding or augmenting training images.
    """
    expected_archive_sha256 = _lower_hex(
        smartdoc_archive_sha256, "smartdoc_archive_sha256"
    )
    if smartdoc_version != "2.0.0":
        raise ValueError("unsupported SmartDoc version")
    expected_stride = _positive_int(stride, "stride")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(model_identity, ModelArtifactIdentity):
        raise TypeError("model_identity must be a ModelArtifactIdentity")
    if model_identity.adapter != "docaligner":
        raise ValueError("model identity must identify DocAligner")
    assert_protected_hashes_absent(
        (expected_archive_sha256, model_identity.sha256)
    )
    stored = tuple(records)
    if any(not isinstance(record, SmartDocRecord) for record in stored):
        raise TypeError("records must contain SmartDocRecord values")
    validate_smartdoc_record_paths(stored)
    raw_cache_canonical = False
    if backgrounds is None:
        selected_backgrounds: frozenset[str] | None = None
    else:
        selected_backgrounds = frozenset(backgrounds)
        if not selected_backgrounds or any(
            not isinstance(background, str) or not background
            for background in selected_backgrounds
        ):
            raise ValueError("backgrounds must contain non-empty strings")
        stored = tuple(
            record for record in stored if record.background in selected_backgrounds
        )
        if not stored:
            raise ValueError("background filter selected no SmartDoc records")
    if payload is None:
        payload = _read_cache_payload(Path(path))
    elif not isinstance(payload, bytes) or not payload or len(payload) > MAX_CACHE_BYTES:
        raise ValueError("supplied refiner cache payload is invalid")
    cache = _parse_refiner_cache(payload)
    if selected_backgrounds is not None:
        if payload != refiner_cache_bytes(cache):
            raise ValueError("refiner cache encoding is not canonical or changed")
        raw_cache_canonical = True
    expected_metadata = (
        expected_archive_sha256,
        smartdoc_version,
        expected_stride,
        seed,
        model_identity,
    )
    actual_metadata = (
        cache.smartdoc_archive_sha256,
        cache.smartdoc_version,
        cache.stride,
        cache.seed,
        cache.model_identity,
    )
    if actual_metadata != expected_metadata:
        raise ValueError("refiner cache metadata mismatch")

    if selected_backgrounds is not None:
        filtered_records = tuple(
            item
            for item in cache.records
            if item.background in selected_backgrounds
        )
        if not filtered_records:
            raise ValueError("background filter selected no refiner cache records")
        cache = RefinerCache(
            smartdoc_archive_sha256=cache.smartdoc_archive_sha256,
            smartdoc_version=cache.smartdoc_version,
            stride=cache.stride,
            seed=cache.seed,
            model_identity=cache.model_identity,
            records=filtered_records,
        )

    by_identity: dict[tuple[str, str, int], SmartDocRecord] = {}
    for record in stored:
        identity = _record_identity(record)
        if identity in by_identity:
            raise ValueError("duplicate SmartDoc record")
        by_identity[identity] = record
    cache_identities = {
        (item.background, item.sequence, item.frame_index) for item in cache.records
    }
    if cache_identities != set(by_identity):
        raise ValueError("refiner cache SmartDoc records mismatch")
    for item in cache.records:
        identity = (item.background, item.sequence, item.frame_index)
        record = by_identity[identity]
        if (
            record.image_sha256 != item.original_image_sha256
            or record.image_size_bytes != item.original_image_size_bytes
            or not np.array_equal(record.corners, item.ground_truth_corners)
            or _split_for_background(record.background) != item.split
        ):
            raise ValueError("SmartDoc record differs from refiner cache")
        image, generated_identity = generate_refiner_view(
            record, item.view, seed=cache.seed
        )
        if item.coarse_corners is not None:
            try:
                if (
                    docaligner.accepted_docaligner_corners(
                        item.coarse_corners,
                        image.shape,
                    )
                    is None
                ):
                    raise ValueError("production DocAligner rejected detection")
                _validate_ordered_quad(item.coarse_corners, image.shape)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "cached detection is invalid for regenerated source bounds"
                ) from error
        if generated_identity != item.augmentation_identity:
            raise ValueError("refiner augmentation identity mismatch")
        generated_key = refiner_cache_key(
            image,
            seed=cache.seed,
            augmentation_identity=generated_identity,
            view=item.view,
            model_identity=cache.model_identity,
        )
        if generated_key != item.cache_key:
            raise ValueError("refiner cache key mismatch")
    if not raw_cache_canonical and payload != refiner_cache_bytes(cache):
        raise ValueError("refiner cache encoding is not canonical or changed")
    return cache


def _immutable_image(value: np.ndarray) -> np.ndarray:
    return np.frombuffer(value.tobytes(order="C"), dtype=np.uint8).reshape(value.shape)


def _gaussian_target(center: np.ndarray, *, sigma: float = 1.5) -> np.ndarray:
    yy, xx = np.mgrid[:HEATMAP_SIZE, :HEATMAP_SIZE].astype(np.float32)
    squared_distance = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return np.exp(-0.5 * squared_distance / np.float32(sigma * sigma)).astype(
        np.float32
    )


def _rasterize_segment(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    target = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    start = tuple(int(value) for value in np.rint(first))
    end = tuple(int(value) for value in np.rint(second))
    visible, clipped_start, clipped_end = cv2.clipLine(
        (0, 0, HEATMAP_SIZE, HEATMAP_SIZE), start, end
    )
    if visible:
        cv2.line(
            target,
            clipped_start,
            clipped_end,
            1.0,
            thickness=2,
            lineType=cv2.LINE_8,
        )
    return target


def _refiner_targets(
    ground_truth: np.ndarray,
    transforms: tuple[
        PatchTransform, PatchTransform, PatchTransform, PatchTransform
    ],
) -> dict[str, torch.Tensor]:
    corner_xy = np.empty((4, 2), dtype=np.float32)
    corner_valid = np.zeros(4, dtype=np.bool_)
    heatmaps = np.zeros((4, 1, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    edge_maps = np.zeros((4, 2, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    heatmap_scale = np.float64((HEATMAP_SIZE - 1) / (PATCH_SIZE - 1))
    for corner_index, transform in enumerate(transforms):
        true_corner = transform.source_to_patch_points(
            ground_truth[corner_index : corner_index + 1]
        )[0]
        corner_xy[corner_index] = (true_corner / np.float64(PATCH_SIZE - 1)).astype(
            np.float32
        )
        valid = bool(
            np.all(true_corner >= 0.0)
            and np.all(true_corner <= float(PATCH_SIZE - 1))
        )
        corner_valid[corner_index] = valid
        if valid:
            heatmaps[corner_index, 0] = _gaussian_target(
                true_corner * heatmap_scale
            )

        next_segment = ground_truth[
            [corner_index, (corner_index + 1) % 4]
        ]
        previous_segment = ground_truth[
            [corner_index, (corner_index - 1) % 4]
        ]
        mapped_next = transform.source_to_patch_points(next_segment) * heatmap_scale
        mapped_previous = (
            transform.source_to_patch_points(previous_segment) * heatmap_scale
        )
        edge_maps[corner_index, 0] = _rasterize_segment(
            mapped_next[0], mapped_next[1]
        )
        edge_maps[corner_index, 1] = _rasterize_segment(
            mapped_previous[0], mapped_previous[1]
        )

    arrays = (corner_xy, heatmaps, edge_maps)
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("refiner target generation produced non-finite values")
    return {
        "corner_heatmap": torch.from_numpy(heatmaps),
        "edge_maps": torch.from_numpy(edge_maps),
        "corner_xy": torch.from_numpy(corner_xy),
        "corner_valid": torch.from_numpy(corner_valid),
    }


@dataclass(frozen=True, slots=True)
class RefinerExample:
    """One focused example with enough geometry for validation visualization."""

    source_bgr: np.ndarray = field(compare=False)
    ground_truth_corners: np.ndarray = field(compare=False)
    coarse_corners: np.ndarray = field(compare=False)
    transforms: tuple[
        PatchTransform, PatchTransform, PatchTransform, PatchTransform
    ] = field(compare=False)
    radius_ratio: float
    view: str
    augmentation_identity: str
    patches: torch.Tensor = field(compare=False)
    targets: Mapping[str, torch.Tensor] = field(compare=False)


@dataclass(frozen=True, slots=True)
class _MaterializedRefinerView:
    group_index: int
    source_bgr: np.ndarray = field(compare=False)
    augmentation_identity: str


class LocalCornerRefinerDataset(
    Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]
):
    """Verified views expanded over three exact local-patch radii.

    One immutable source view is retained while a grouped sampler visits its
    three radii. Moving to any other view drops it; the next materialization
    rereads provenance-bound source bytes and verifies the cache key.
    """

    def __init__(
        self,
        cache: RefinerCache,
        records: Sequence[SmartDocRecord],
        split: str,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if not isinstance(cache, RefinerCache):
            raise TypeError("cache must be a RefinerCache")
        if split not in _SPLITS:
            raise ValueError("split must be train or validation")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if cache.seed != seed:
            raise ValueError("dataset seed differs from refiner cache")
        stored = tuple(records)
        if any(not isinstance(record, SmartDocRecord) for record in stored):
            raise TypeError("records must contain SmartDocRecord values")
        by_identity: dict[tuple[str, str, int], SmartDocRecord] = {}
        for record in stored:
            identity = _record_identity(record)
            if identity in by_identity:
                raise ValueError("duplicate SmartDoc record")
            by_identity[identity] = record
        cache_identities = {
            (item.background, item.sequence, item.frame_index)
            for item in cache.records
        }
        if cache_identities != set(by_identity):
            raise ValueError("refiner cache SmartDoc records mismatch")
        for item in cache.records:
            record = by_identity[(item.background, item.sequence, item.frame_index)]
            if (
                record.image_sha256 != item.original_image_sha256
                or record.image_size_bytes != item.original_image_size_bytes
                or not np.array_equal(record.corners, item.ground_truth_corners)
                or _split_for_background(record.background) != item.split
            ):
                raise ValueError("SmartDoc record differs from refiner cache")
        self.cache = cache
        self.records = stored
        self.split = split
        self.seed = seed
        self._records_by_identity = by_identity
        self._cache_records = tuple(
            item
            for item in cache.records
            if item.split == split and item.coarse_corners is not None
        )
        self._materialized_view: _MaterializedRefinerView | None = None

    def __len__(self) -> int:
        return len(self._cache_records) * len(PATCH_RADIUS_RATIOS)

    def _index_components(
        self,
        index: int,
    ) -> tuple[int, RefinerCacheRecord, float]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("index must be an integer")
        numeric_index = int(index)
        if numeric_index < 0 or numeric_index >= len(self):
            raise IndexError("refiner dataset index out of range")
        group_index = numeric_index // len(PATCH_RADIUS_RATIOS)
        cache_record = self._cache_records[group_index]
        radius_ratio = PATCH_RADIUS_RATIOS[
            numeric_index % len(PATCH_RADIUS_RATIOS)
        ]
        return group_index, cache_record, radius_ratio

    def _materialize(
        self,
        group_index: int,
        cache_record: RefinerCacheRecord,
    ) -> _MaterializedRefinerView:
        materialized = self._materialized_view
        if materialized is not None and materialized.group_index == group_index:
            return materialized
        self._materialized_view = None
        record = self._records_by_identity[
            (
                cache_record.background,
                cache_record.sequence,
                cache_record.frame_index,
            )
        ]
        if (
            record.image_sha256 != cache_record.original_image_sha256
            or record.image_size_bytes != cache_record.original_image_size_bytes
            or not np.array_equal(record.corners, cache_record.ground_truth_corners)
        ):
            raise RuntimeError("SmartDoc record differs from refiner cache")
        source, identity = generate_refiner_view(
            record, cache_record.view, seed=self.seed
        )
        key = refiner_cache_key(
            source,
            seed=self.seed,
            augmentation_identity=identity,
            view=cache_record.view,
            model_identity=self.cache.model_identity,
        )
        if (
            identity != cache_record.augmentation_identity
            or key != cache_record.cache_key
        ):
            raise RuntimeError("refiner cache key mismatch")
        if (
            cache_record.coarse_corners is None
            or docaligner.accepted_docaligner_corners(
                cache_record.coarse_corners,
                source.shape,
            )
            is None
        ):
            raise RuntimeError("cached detection is rejected by production DocAligner")
        materialized = _MaterializedRefinerView(
            group_index=group_index,
            source_bgr=_immutable_image(source),
            augmentation_identity=identity,
        )
        self._materialized_view = materialized
        return materialized

    def _load_tensors(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        tuple[PatchTransform, PatchTransform, PatchTransform, PatchTransform],
        RefinerCacheRecord,
        float,
        _MaterializedRefinerView,
    ]:
        group_index, cache_record, radius_ratio = self._index_components(index)
        materialized = self._materialize(group_index, cache_record)
        coarse = cache_record.coarse_corners
        if coarse is None:
            raise RuntimeError("excluded cache record reached dataset indexing")
        raw_patches, transforms = extract_corner_patches(
            materialized.source_bgr,
            coarse,
            radius_ratio=radius_ratio,
        )
        rgb = raw_patches[..., ::-1].astype(np.float32) / np.float32(255.0)
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        patches = torch.from_numpy(
            np.transpose(normalized, (0, 3, 1, 2)).copy()
        )
        targets = _refiner_targets(cache_record.ground_truth_corners, transforms)
        return (
            patches,
            targets,
            transforms,
            cache_record,
            radius_ratio,
            materialized,
        )

    def load_example(self, index: int) -> RefinerExample:
        (
            patches,
            targets,
            transforms,
            cache_record,
            radius_ratio,
            materialized,
        ) = self._load_tensors(index)
        coarse = cache_record.coarse_corners
        if coarse is None:
            raise RuntimeError("excluded cache record reached dataset indexing")
        return RefinerExample(
            source_bgr=materialized.source_bgr,
            ground_truth_corners=cache_record.ground_truth_corners,
            coarse_corners=coarse,
            transforms=transforms,
            radius_ratio=radius_ratio,
            view=cache_record.view,
            augmentation_identity=materialized.augmentation_identity,
            patches=patches,
            targets=targets,
        )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patches, targets, _, _, _, _ = self._load_tensors(index)
        return patches, targets

    def grouped_sampler(self) -> RefinerViewGroupedSampler:
        """Return the canonical grouped sampler for this dataset split."""
        return RefinerViewGroupedSampler(self)


class RefinerViewGroupedSampler(Sampler[int]):
    """Yield all three radii for each view before advancing to another view."""

    def __init__(self, dataset: LocalCornerRefinerDataset) -> None:
        if not isinstance(dataset, LocalCornerRefinerDataset):
            raise TypeError("dataset must be a LocalCornerRefinerDataset")
        self._group_count = len(dataset._cache_records)
        self._split = dataset.split
        self._seed = dataset.seed
        self._epoch = 0

    def __len__(self) -> int:
        return self._group_count * len(PATCH_RADIUS_RATIOS)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch

    def __iter__(self):
        groups = np.arange(self._group_count, dtype=np.int64)
        if self._split == "train":
            identity = f"{self._seed}\0{self._epoch}\0{self._split}".encode()
            generator = np.random.default_rng(
                int.from_bytes(hashlib.sha256(identity).digest()[:16], "little")
            )
            generator.shuffle(groups)
        for group in groups:
            start = int(group) * len(PATCH_RADIUS_RATIOS)
            yield from range(start, start + len(PATCH_RADIUS_RATIOS))


def _argument_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _validate_cache_output_collisions(
    cache_output: Path,
    *,
    dataset_root: Path,
    archive_path: Path,
    model_dir: Path,
    model_identity: ModelArtifactIdentity,
) -> None:
    output = Path(os.path.abspath(cache_output)).resolve(strict=False)
    dataset = Path(os.path.abspath(dataset_root)).resolve(strict=False)
    archive = Path(os.path.abspath(archive_path)).resolve(strict=False)
    model = (
        Path(os.path.abspath(model_dir)) / model_identity.filename
    ).resolve(strict=False)
    if (
        output == archive
        or output == model
        or output == dataset
        or dataset in output.parents
    ):
        raise ValueError("cache output collides with verified input data")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path, dest="archive_path")
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stride", required=True, type=_argument_positive_int)
    parser.add_argument("--cpu-threads", required=True, type=_argument_positive_int)
    parser.add_argument("--cache-output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.archive_sha256 != SMARTDOC_ARCHIVE_SHA256:
        raise ValueError(
            "archive_sha256 must equal the pinned SmartDoc 2015 v2.0.0 hash"
        )
    validate_smartdoc_source_paths(args.dataset_root, args.archive_path)
    verify_smartdoc_markers(args.dataset_root)
    source = verify_smartdoc_source(args.archive_path, args.dataset_root)
    if source.archive_sha256 != args.archive_sha256:
        raise ValueError("verified SmartDoc archive hash differs from supplied hash")
    assert_protected_hashes_absent(
        (source.archive_sha256, *(item.sha256 for item in source.files.values()))
    )
    selected = load_smartdoc_records(
        args.dataset_root,
        stride=args.stride,
        verified_files=source.files,
    )
    train, validation = split_smartdoc_records(selected, stride=1)
    records = (*train, *validation)
    if not records:
        raise ValueError("stride-selected SmartDoc split is empty")
    session, model_identity = docaligner._load_docaligner_session(
        args.model_dir, args.cpu_threads
    )
    assert_protected_hashes_absent((model_identity.sha256,))
    _validate_cache_output_collisions(
        args.cache_output,
        dataset_root=args.dataset_root,
        archive_path=args.archive_path,
        model_dir=args.model_dir,
        model_identity=model_identity,
    )
    cache = build_refiner_cache(
        records,
        smartdoc_archive_sha256=source.archive_sha256,
        smartdoc_version=SMARTDOC_VERSION,
        stride=args.stride,
        seed=args.seed,
        session=session,
        model_identity=model_identity,
    )
    save_refiner_cache(cache, args.cache_output)
    return 0


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_SEED",
    "MAX_CACHE_BYTES",
    "MAX_CACHE_RECORDS",
    "PATCH_RADIUS_RATIOS",
    "VIEW_NAMES",
    "LocalCornerRefinerDataset",
    "RefinerCache",
    "RefinerCacheRecord",
    "RefinerExample",
    "RefinerViewGroupedSampler",
    "augmentation_identity",
    "build_refiner_cache",
    "generate_refiner_view",
    "load_verified_refiner_cache",
    "main",
    "parse_args",
    "refiner_cache_bytes",
    "refiner_cache_key",
    "save_refiner_cache",
]


if __name__ == "__main__":
    raise SystemExit(main())
