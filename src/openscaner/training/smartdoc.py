"""Isolated, deterministic SmartDoc 2015 corner-training data."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import re
import stat
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import BinaryIO

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from openscaner.geometry import order_quad

SMARTDOC_ARCHIVE_SHA256 = (
    "3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae"
)
PROTECTED_SHA256 = frozenset(
    {
        "ea7cc4e7255051730710ebaef345973eb39fb5e22df2cd363595dda8b66ae83b",
        "f077cecae296dd2b095735414af99f2b676f3b723c99bd63d6d56a35e5e1b02d",
        "c48c9514eb68b17eb35a0341d7c434376e95926670fc55e1fe3b86a736fc2dc1",
    }
)
SMARTDOC_METADATA_FIELDS = (
    "bg_name",
    "bg_id",
    "model_name",
    "model_id",
    "modeltype_name",
    "modeltype_id",
    "model_subid",
    "image_path",
    "frame_index",
    "model_width",
    "model_height",
    "tl_x",
    "tl_y",
    "bl_x",
    "bl_y",
    "br_x",
    "br_y",
    "tr_x",
    "tr_y",
)
SMARTDOC_TRAIN_BACKGROUNDS = frozenset(
    {"background01", "background02", "background03", "background04"}
)
SMARTDOC_VALIDATION_BACKGROUND = "background05"
SMARTDOC_STRIDE = 5
SMARTDOC_STRIDE_5_TRAIN_RECORDS = 4416
SMARTDOC_STRIDE_5_VALIDATION_RECORDS = 503
SMARTDOC_INPUT_SIZE = 384
SMARTDOC_TARGET_SIZE = 96
SMARTDOC_GLOBAL_SEED = 20260825
SMARTDOC_VERSION = "2.0.0"
SMARTDOC_MARKER_SHA256 = MappingProxyType(
    {
        "VERSION": "83032357fad1290b27c1ebc7f551ae0df9b0a61676865fb9b224e9ef2e12f17d",
        "LICENCE": "fae21effd8909451cf43888c859b67206882958f429320fb6a8559cf4e78ce6c",
    }
)
SMARTDOC_MAX_MARKER_BYTES = 64 * 1024
SMARTDOC_MAX_UNVERIFIED_FILE_BYTES = 256 * 1024 * 1024
SMARTDOC_PROTECTED_PATH_TOKENS = frozenset(
    {"evaluator", "groundtruth", "reference", "target"}
)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

_PATH_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_protected_hashes_absent(source_hashes: Iterable[str]) -> None:
    """Reject content matching any evaluator-only protected artifact."""
    overlap = PROTECTED_SHA256.intersection(
        str(source_hash).strip().lower() for source_hash in source_hashes
    )
    if overlap:
        raise ValueError(f"protected hash present in SmartDoc training data: {sorted(overlap)}")


def _normalized_path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in Path(os.path.abspath(path)).parts:
        normalized = re.sub(
            r"ground[^a-z0-9]*truth",
            "groundtruth",
            part.lower(),
        )
        tokens.update(
            token for token in _PATH_TOKEN_SPLIT.split(normalized) if token
        )
    return tokens


def _validate_smartdoc_path(path: Path, *, description: str) -> None:
    overlap = _normalized_path_tokens(Path(path)) & SMARTDOC_PROTECTED_PATH_TOKENS
    if overlap:
        raise ValueError(
            f"protected {description} path token is forbidden: {sorted(overlap)}"
        )


def validate_smartdoc_source_paths(
    dataset_root: Path,
    archive_path: Path,
) -> None:
    """Reject source paths containing protected exact normalized tokens."""
    _validate_smartdoc_path(Path(dataset_root), description="dataset")
    _validate_smartdoc_path(Path(archive_path), description="archive")


@dataclass(frozen=True)
class VerifiedSmartDocFile:
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class VerifiedSmartDocSource:
    archive_sha256: str
    files: Mapping[str, VerifiedSmartDocFile]


@dataclass(frozen=True)
class _ExtractionFile:
    path: Path
    device: int
    inode: int
    size_bytes: int
    link_count: int


@contextmanager
def _verified_archive_snapshot(
    archive_path: Path, expected_sha256: str | None
) -> Iterator[tuple[BinaryIO, str]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(Path(archive_path), flags)
    except OSError as error:
        raise ValueError("SmartDoc archive is missing or is not a regular file") from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("SmartDoc archive is missing or is not a regular file")
        source = os.fdopen(descriptor, "rb")
        descriptor = -1
        with source:
            with tempfile.TemporaryFile(mode="w+b") as snapshot:
                digest = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    snapshot.write(chunk)
                archive_sha256 = digest.hexdigest()
                assert_protected_hashes_absent((archive_sha256,))
                if expected_sha256 is not None and archive_sha256 != expected_sha256:
                    raise ValueError(
                        "SmartDoc archive SHA-256 mismatch: "
                        f"expected {expected_sha256}, got {archive_sha256}"
                    )
                snapshot.flush()
                snapshot.seek(0)
                yield snapshot, archive_sha256
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_smartdoc_archive(archive_path: Path) -> str:
    """Verify and return the digest of the pinned SmartDoc 2015 v2.0.0 archive."""
    with _verified_archive_snapshot(
        archive_path, SMARTDOC_ARCHIVE_SHA256
    ) as (_, archive_sha256):
        return archive_sha256


def _extraction_inventory(root: Path) -> dict[str, _ExtractionFile]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("SmartDoc extraction does not match verified archive")
    files: dict[str, _ExtractionFile] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError as error:
            raise ValueError(
                "SmartDoc extraction does not match verified archive"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError("SmartDoc extraction does not match verified archive")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    descriptor = os.open(path, flags)
                    try:
                        status = os.fstat(descriptor)
                    finally:
                        os.close(descriptor)
                except OSError as error:
                    raise ValueError(
                        "SmartDoc extraction does not match verified archive"
                    ) from error
                if not stat.S_ISREG(status.st_mode):
                    raise ValueError(
                        "SmartDoc extraction does not match verified archive"
                    )
                files[path.relative_to(root).as_posix()] = _ExtractionFile(
                    path=path,
                    device=status.st_dev,
                    inode=status.st_ino,
                    size_bytes=status.st_size,
                    link_count=status.st_nlink,
                )
            else:
                raise ValueError("SmartDoc extraction does not match verified archive")
    identities: dict[tuple[int, int], str] = {}
    for name, extracted in files.items():
        identity = (extracted.device, extracted.inode)
        if identity in identities:
            raise ValueError(
                "SmartDoc duplicate extraction inode: "
                f"{identities[identity]} and {name}"
            )
        identities[identity] = name
    if any(extracted.link_count != 1 for extracted in files.values()):
        raise ValueError("SmartDoc hard-linked extraction file is not allowed")
    return files


def _safe_archive_name(name: str) -> str:
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != name
    ):
        raise ValueError("SmartDoc extraction does not match verified archive")
    return relative.as_posix()


def _matches_archive_file(
    member: tarfile.TarInfo, extracted: _ExtractionFile, stream: BinaryIO
) -> VerifiedSmartDocFile | None:
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(extracted.path, flags)
        try:
            root_stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with root_stream:
            root_stat = os.fstat(root_stream.fileno())
            if (
                not stat.S_ISREG(root_stat.st_mode)
                or root_stat.st_nlink != 1
                or root_stat.st_size != member.size
                or (root_stat.st_dev, root_stat.st_ino)
                != (extracted.device, extracted.inode)
            ):
                return None
            digest = hashlib.sha256()
            while True:
                archive_chunk = stream.read(1024 * 1024)
                root_chunk = root_stream.read(1024 * 1024)
                if archive_chunk != root_chunk:
                    return None
                if not archive_chunk:
                    return VerifiedSmartDocFile(
                        size_bytes=member.size,
                        sha256=digest.hexdigest(),
                    )
                digest.update(archive_chunk)
    except OSError:
        return None


def _verify_snapshot_extraction(
    snapshot: BinaryIO, dataset_root: Path
) -> Mapping[str, VerifiedSmartDocFile]:
    """Verify the complete regular-file inventory, intentionally ignoring dirs."""
    root = Path(dataset_root)
    files = _extraction_inventory(root)
    seen: set[str] = set()
    verified: dict[str, VerifiedSmartDocFile] = {}
    try:
        snapshot.seek(0)
        with tarfile.open(fileobj=snapshot, mode="r:gz") as archive:
            for member in archive:
                name = _safe_archive_name(member.name)
                if not member.isfile() or name in seen or name not in files:
                    raise ValueError(
                        "SmartDoc extraction does not match verified archive"
                    )
                archived_file = archive.extractfile(member)
                matched = (
                    None
                    if archived_file is None
                    else _matches_archive_file(member, files[name], archived_file)
                )
                if matched is None:
                    raise ValueError(
                        "SmartDoc extraction does not match verified archive"
                    )
                verified[name] = matched
                seen.add(name)
    except (OSError, tarfile.TarError) as error:
        raise ValueError("SmartDoc extraction does not match verified archive") from error
    if seen != files.keys():
        raise ValueError("SmartDoc extraction does not match verified archive")
    return MappingProxyType(verified)


def verify_smartdoc_extraction(archive_path: Path, dataset_root: Path) -> None:
    """Compare the complete regular-file inventory from one archive snapshot."""
    with _verified_archive_snapshot(archive_path, None) as (snapshot, _):
        _verify_snapshot_extraction(snapshot, dataset_root)


def _verify_smartdoc_source(
    archive_path: Path,
    dataset_root: Path,
    *,
    expected_sha256: str,
    after_snapshot: Callable[[], None] | None = None,
) -> VerifiedSmartDocSource:
    with _verified_archive_snapshot(
        archive_path, expected_sha256
    ) as (snapshot, archive_sha256):
        if after_snapshot is not None:
            after_snapshot()
        files = _verify_snapshot_extraction(snapshot, dataset_root)
    return VerifiedSmartDocSource(archive_sha256=archive_sha256, files=files)


def verify_smartdoc_source(
    archive_path: Path, dataset_root: Path
) -> VerifiedSmartDocSource:
    """Bind the extraction to the exact bytes of the pinned archive snapshot."""
    return _verify_smartdoc_source(
        archive_path,
        dataset_root,
        expected_sha256=SMARTDOC_ARCHIVE_SHA256,
    )


@dataclass(frozen=True)
class SmartDocRecord:
    """One immutable SmartDoc frame and its complete ordered quadrilateral."""

    image_path: Path
    corners: np.ndarray = field(compare=False)
    background: str
    sequence: str
    frame_index: int
    image_size_bytes: int | None = field(default=None, compare=False)
    image_sha256: str | None = field(default=None, compare=False)
    dataset_root: Path | None = field(default=None, compare=False)
    image_relative_path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError("corners must have shape (4, 2)")
        corners = np.frombuffer(corners.tobytes(), dtype=np.float32).reshape(4, 2)
        if (self.image_size_bytes is None) != (self.image_sha256 is None):
            raise ValueError("image size and SHA-256 provenance must be provided together")
        if self.image_size_bytes is not None and self.image_size_bytes < 0:
            raise ValueError("image size provenance must be non-negative")
        if self.image_sha256 is not None and (
            len(self.image_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.image_sha256)
        ):
            raise ValueError("image SHA-256 provenance must be lowercase hexadecimal")
        image_path = Path(self.image_path)
        if (self.dataset_root is None) != (self.image_relative_path is None):
            raise ValueError(
                "dataset root and image-relative path must be provided together"
            )
        if self.dataset_root is not None and self.image_relative_path is not None:
            dataset_root = Path(self.dataset_root)
            image_relative_path = Path(self.image_relative_path)
            if (
                image_relative_path.is_absolute()
                or not image_relative_path.parts
                or ".." in image_relative_path.parts
                or image_path != dataset_root / image_relative_path
            ):
                raise ValueError("image path must be relative to the SmartDoc root")
            object.__setattr__(self, "dataset_root", dataset_root)
            object.__setattr__(self, "image_relative_path", image_relative_path)
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "corners", corners)


def validate_smartdoc_record_paths(records: Iterable[SmartDocRecord]) -> None:
    """Reject protected tokens in every record root and image path."""
    stored = tuple(records)
    if any(not isinstance(record, SmartDocRecord) for record in stored):
        raise TypeError("records must contain only SmartDocRecord instances")
    violations: set[str] = set()
    for record in stored:
        paths = [("image", record.image_path)]
        if record.dataset_root is not None:
            paths.append(("dataset", record.dataset_root))
        for description, path in paths:
            overlap = (
                _normalized_path_tokens(path) & SMARTDOC_PROTECTED_PATH_TOKENS
            )
            violations.update(
                f"{description}:{token}" for token in overlap
            )
    if violations:
        raise ValueError(
            "protected SmartDoc record path token is forbidden: "
            f"{sorted(violations)}"
        )


def _segments_cross(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
) -> bool:
    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        first_vector = b - a
        second_vector = c - a
        return float(
            first_vector[0] * second_vector[1]
            - first_vector[1] * second_vector[0]
        )

    first_side = orientation(first, second, third)
    second_side = orientation(first, second, fourth)
    third_side = orientation(third, fourth, first)
    fourth_side = orientation(third, fourth, second)
    return first_side * second_side < 0.0 and third_side * fourth_side < 0.0


def _validate_quad(corners: np.ndarray, *, width: int, height: int) -> None:
    if corners.shape != (4, 2):
        raise ValueError("corner coordinates must form a 4x2 quadrilateral")
    if not np.isfinite(corners).all():
        raise ValueError("corner coordinates must be finite")
    if (
        np.any(corners[:, 0] < 0.0)
        or np.any(corners[:, 0] > width - 1)
        or np.any(corners[:, 1] < 0.0)
        or np.any(corners[:, 1] > height - 1)
    ):
        raise ValueError("corner coordinates are outside image bounds")
    if len(np.unique(corners, axis=0)) != 4:
        raise ValueError("quadrilateral is degenerate")
    if _segments_cross(corners[0], corners[1], corners[2], corners[3]) or _segments_cross(
        corners[1], corners[2], corners[3], corners[0]
    ):
        raise ValueError("quadrilateral is self-crossing")
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if area <= np.finfo(np.float32).eps or not cv2.isContourConvex(
        corners.astype(np.float32)
    ):
        raise ValueError("quadrilateral is degenerate or non-convex")


def _absolute_root_components(root: Path) -> tuple[str, ...]:
    root_value = os.fspath(root)
    if not isinstance(root_value, str) or not root_value.startswith("/"):
        raise ValueError("SmartDoc dataset root must be an absolute path")
    if root_value == "/":
        return ()
    components = tuple(root_value[1:].split("/"))
    if any(
        not component or component in (".", "..") or "\x00" in component
        for component in components
    ):
        raise ValueError("SmartDoc dataset root contains ambiguous components")
    return components


def _open_absolute_root_descriptor(root: Path, directory_flags: int) -> int:
    components = _absolute_root_components(root)
    try:
        directory_descriptor = os.open("/", directory_flags)
    except OSError as error:
        raise ValueError("unable to open SmartDoc dataset root") from error
    try:
        for component in components:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise ValueError(
                    "unable to open SmartDoc dataset root without symlinks"
                ) from error
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return directory_descriptor
    except BaseException:
        os.close(directory_descriptor)
        raise


def _open_root_relative_descriptor(root: Path, relative_path: Path) -> int:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("SmartDoc path escapes the dataset root")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = _open_absolute_root_descriptor(root, directory_flags)
    try:
        for component in relative.parts[:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise ValueError(
                    f"unable to open SmartDoc directory without symlinks: {relative}"
                ) from error
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        try:
            return os.open(
                relative.parts[-1],
                file_flags,
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            raise ValueError(
                f"unable to open SmartDoc file without symlinks: {relative}"
            ) from error
    finally:
        os.close(directory_descriptor)


def _validate_root_relative_regular_file(root: Path, relative_path: Path) -> None:
    descriptor = _open_root_relative_descriptor(root, relative_path)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise ValueError(
                f"SmartDoc file must be a singly-linked regular file: {relative_path}"
            )
    finally:
        os.close(descriptor)


def _resolve_image_path(root: Path, value: str) -> tuple[Path, Path]:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("invalid image path in SmartDoc metadata")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image path escapes the SmartDoc root")
    candidate = root / relative
    try:
        _validate_root_relative_regular_file(root, relative)
    except ValueError as error:
        raise ValueError("missing image referenced by SmartDoc metadata") from error
    return candidate, relative


def _expected_file(
    root: Path,
    path: Path,
    verified_files: Mapping[str, VerifiedSmartDocFile] | None,
) -> VerifiedSmartDocFile | None:
    if verified_files is None:
        return None
    name = path.relative_to(root).as_posix()
    try:
        return verified_files[name]
    except KeyError as error:
        raise ValueError(f"verified SmartDoc inventory is missing {name}") from error


def _read_open_descriptor(
    descriptor: int,
    path: Path,
    expected: VerifiedSmartDocFile | None,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    effective_maximum = (
        SMARTDOC_MAX_UNVERIFIED_FILE_BYTES if max_bytes is None else max_bytes
    )
    if (
        isinstance(effective_maximum, bool)
        or not isinstance(effective_maximum, int)
        or effective_maximum < 1
    ):
        os.close(descriptor)
        raise ValueError("SmartDoc maximum read size must be a positive integer")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"SmartDoc file must be a singly-linked regular file: {path}")
        if expected is not None and before.st_size != expected.size_bytes:
            raise ValueError(f"verified SmartDoc file changed: {path}")
        read_limit = expected.size_bytes if expected is not None else effective_maximum
        if before.st_size > read_limit:
            raise ValueError(f"SmartDoc file exceeds maximum size: {path}")
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            raise
        descriptor = -1
        with stream:
            payload = stream.read(read_limit + 1)
            after = os.fstat(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    digest = hashlib.sha256(payload).hexdigest()
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != after.st_size
        or after.st_size != len(payload)
        or len(payload) > read_limit
        or (
            expected is not None
            and (expected.size_bytes != len(payload) or expected.sha256 != digest)
        )
    ):
        raise ValueError(f"verified SmartDoc file changed: {path}")
    return payload, digest


def _read_regular_file(
    path: Path,
    expected: VerifiedSmartDocFile | None,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"unable to read SmartDoc file: {path}") from error
    return _read_open_descriptor(
        descriptor,
        path,
        expected,
        max_bytes=max_bytes,
    )


def _read_root_relative_file(
    root: Path,
    relative_path: Path,
    expected: VerifiedSmartDocFile | None,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    descriptor = _open_root_relative_descriptor(root, relative_path)
    path = Path(root) / relative_path
    return _read_open_descriptor(
        descriptor,
        path,
        expected,
        max_bytes=max_bytes,
    )


def _decode_image(payload: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def read_verified_smartdoc_image(record: SmartDocRecord) -> np.ndarray:
    """Read one provenance-bound SmartDoc frame at its original BGR resolution."""
    if not isinstance(record, SmartDocRecord):
        raise TypeError("record must be a SmartDocRecord")
    if (
        record.dataset_root is None
        or record.image_relative_path is None
        or record.image_size_bytes is None
        or record.image_sha256 is None
    ):
        raise ValueError("SmartDoc record is missing verified image provenance")
    expected = VerifiedSmartDocFile(
        size_bytes=record.image_size_bytes,
        sha256=record.image_sha256,
    )
    payload, digest = _read_root_relative_file(
        record.dataset_root,
        record.image_relative_path,
        expected,
    )
    assert_protected_hashes_absent((digest,))
    image = _decode_image(payload)
    if (
        image is None
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError(f"unable to read SmartDoc image: {record.image_path}")
    return image.copy()


def verify_smartdoc_markers(dataset_root: Path) -> None:
    """Verify the version and licence files for SmartDoc 2015 v2.0.0."""
    root = Path(dataset_root)
    for name, expected_sha256 in SMARTDOC_MARKER_SHA256.items():
        _, digest = _read_root_relative_file(
            root,
            Path(name),
            None,
            max_bytes=SMARTDOC_MAX_MARKER_BYTES,
        )
        if digest != expected_sha256:
            raise ValueError(
                f"SmartDoc {name} does not identify version {SMARTDOC_VERSION}"
            )


def _parse_frame_index(value: str | None) -> int:
    try:
        frame_index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("frame_index must be an integer") from error
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    return frame_index


def _parse_csv_corners(row: dict[str, str | None]) -> np.ndarray:
    names = (
        "tl_x",
        "tl_y",
        "bl_x",
        "bl_y",
        "br_x",
        "br_y",
        "tr_x",
        "tr_y",
    )
    try:
        values = [float(row[name]) for name in names]  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("corner coordinates must be numeric") from error
    corners = np.asarray(values, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(corners).all():
        raise ValueError("corner coordinates must be finite")
    return corners


def _require_stride(stride: int) -> None:
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be an integer of at least 1")


def load_smartdoc_records(
    root: Path,
    *,
    stride: int | None = None,
    verified_files: Mapping[str, VerifiedSmartDocFile] | None = None,
    backgrounds: Sequence[str] | None = None,
) -> tuple[SmartDocRecord, ...]:
    """Load records, optionally deferring image work for skipped backgrounds."""
    if stride is not None:
        _require_stride(stride)
    if backgrounds is None:
        selected_backgrounds: frozenset[str] | None = None
    else:
        selected_backgrounds = frozenset(backgrounds)
        if not selected_backgrounds or any(
            not isinstance(background, str) or not background
            for background in selected_backgrounds
        ):
            raise ValueError("backgrounds must contain non-empty strings")
    selection_stride = 1 if stride is None else stride
    _absolute_root_components(root)
    dataset_root = Path(root)
    metadata_relative_path = Path("metadata.csv.gz")
    metadata_path = dataset_root / metadata_relative_path
    metadata_expected = _expected_file(dataset_root, metadata_path, verified_files)
    metadata_payload, _ = _read_root_relative_file(
        dataset_root,
        metadata_relative_path,
        metadata_expected,
    )

    selected_rows: list[
        tuple[
            int,
            Path,
            Path,
            VerifiedSmartDocFile | None,
            np.ndarray,
            str,
            str,
            int,
        ]
    ] = []
    image_paths: set[Path] = set()
    identities: set[tuple[str, str, int]] = set()
    row_count = 0
    try:
        with gzip.open(
            io.BytesIO(metadata_payload), "rt", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream, strict=True)
            if tuple(reader.fieldnames or ()) != SMARTDOC_METADATA_FIELDS:
                raise ValueError("invalid SmartDoc metadata schema")
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"malformed row at metadata row {row_number}")
                background = (row["bg_name"] or "").strip()
                sequence = (row["model_name"] or "").strip()
                if not background or not sequence:
                    raise ValueError(
                        f"background and sequence are required at metadata row {row_number}"
                    )
                frame_index = _parse_frame_index(row["frame_index"])
                identity = (background, sequence, frame_index)
                if identity in identities:
                    raise ValueError(f"duplicate record at metadata row {row_number}")

                csv_corners = _parse_csv_corners(row)
                if (
                    selected_backgrounds is not None
                    and background not in selected_backgrounds
                ):
                    continue

                image_path, image_relative_path = _resolve_image_path(
                    dataset_root, (row["image_path"] or "").strip()
                )
                image_expected = _expected_file(
                    dataset_root, image_path, verified_files
                )
                if image_path in image_paths:
                    raise ValueError(f"duplicate image at metadata row {row_number}")
                if frame_index % selection_stride == 0:
                    selected_rows.append(
                        (
                            row_number,
                            image_path,
                            image_relative_path,
                            image_expected,
                            csv_corners,
                            background,
                            sequence,
                            frame_index,
                        )
                    )
                image_paths.add(image_path)
                identities.add(identity)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError("invalid SmartDoc metadata.csv.gz") from error

    if row_count == 0:
        raise ValueError("SmartDoc metadata contains no records")

    records: list[SmartDocRecord] = []
    for (
        row_number,
        image_path,
        image_relative_path,
        image_expected,
        csv_corners,
        background,
        sequence,
        frame_index,
    ) in selected_rows:
        image_payload, image_sha256 = _read_root_relative_file(
            dataset_root,
            image_relative_path,
            image_expected,
        )
        assert_protected_hashes_absent((image_sha256,))
        image = _decode_image(image_payload)
        if image is None:
            raise ValueError(f"unreadable image at metadata row {row_number}")
        height, width = image.shape[:2]
        _validate_quad(csv_corners, width=width, height=height)
        records.append(
            SmartDocRecord(
                image_path=image_path,
                corners=order_quad(csv_corners),
                background=background,
                sequence=sequence,
                frame_index=frame_index,
                image_size_bytes=len(image_payload),
                image_sha256=image_sha256,
                dataset_root=dataset_root,
                image_relative_path=image_relative_path,
            )
        )
    return tuple(records)


def split_smartdoc_records(
    records: Sequence[SmartDocRecord], stride: int
) -> tuple[tuple[SmartDocRecord, ...], tuple[SmartDocRecord, ...]]:
    """Split by background and retain frames whose index is divisible by stride."""
    _require_stride(stride)
    known_backgrounds = SMARTDOC_TRAIN_BACKGROUNDS | {
        SMARTDOC_VALIDATION_BACKGROUND
    }
    unknown = sorted({record.background for record in records} - known_backgrounds)
    if unknown:
        raise ValueError(f"unknown SmartDoc background: {unknown}")

    selected = tuple(record for record in records if record.frame_index % stride == 0)
    train = tuple(
        record for record in selected if record.background in SMARTDOC_TRAIN_BACKGROUNDS
    )
    validation = tuple(
        record
        for record in selected
        if record.background == SMARTDOC_VALIDATION_BACKGROUND
    )
    return train, validation


def _rotate_image_and_corners(
    image: np.ndarray, corners: np.ndarray, degrees: int
) -> tuple[np.ndarray, np.ndarray]:
    turns = degrees // 90
    if degrees not in (0, 90, 180, 270):
        raise ValueError("rotation must be one of 0, 90, 180, or 270 degrees")
    if turns == 0:
        return image.copy(), corners.copy()

    height, width = image.shape[:2]
    rotated = np.rot90(image, turns).copy()
    transformed = corners.copy()
    if turns == 1:
        transformed = np.column_stack(
            (corners[:, 1], (width - 1.0) - corners[:, 0])
        )
        order = (1, 2, 3, 0)
    elif turns == 2:
        transformed = np.column_stack(
            ((width - 1.0) - corners[:, 0], (height - 1.0) - corners[:, 1])
        )
        order = (2, 3, 0, 1)
    else:
        transformed = np.column_stack(
            ((height - 1.0) - corners[:, 1], corners[:, 0])
        )
        order = (3, 0, 1, 2)
    return rotated, transformed[list(order)].astype(np.float32)


def _clutter_canvas(
    generator: np.random.Generator, *, height: int, width: int
) -> np.ndarray:
    base = generator.integers(20, 226, size=3, dtype=np.uint8)
    canvas = np.broadcast_to(base, (height, width, 3)).copy()
    noise = generator.normal(0.0, 12.0, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    for _ in range(10):
        color = tuple(int(value) for value in generator.integers(0, 256, size=3))
        first = tuple(int(value) for value in generator.integers(0, width, size=2))
        second = tuple(int(value) for value in generator.integers(0, height, size=2))
        cv2.rectangle(
            canvas,
            (first[0], second[0]),
            (first[1], second[1]),
            color,
            thickness=int(generator.integers(1, 8)),
            lineType=cv2.LINE_AA,
        )
    return canvas


def _zoom_out_on_clutter(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    scale: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scaled_width = max(2, int(round(width * scale)))
    scaled_height = max(2, int(round(height * scale)))
    reduced = cv2.resize(
        image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
    )
    offset_x = int(generator.integers(0, width - scaled_width + 1))
    offset_y = int(generator.integers(0, height - scaled_height + 1))
    canvas = _clutter_canvas(generator, height=height, width=width)
    canvas[
        offset_y : offset_y + scaled_height,
        offset_x : offset_x + scaled_width,
    ] = reduced
    coordinate_scale = np.asarray(
        (
            (scaled_width - 1.0) / (width - 1.0),
            (scaled_height - 1.0) / (height - 1.0),
        ),
        dtype=np.float32,
    )
    transformed = corners * coordinate_scale + np.asarray(
        (offset_x, offset_y), dtype=np.float32
    )
    return canvas, transformed.astype(np.float32)


def _apply_photometric_augmentation(
    image: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    gain = float(generator.uniform(0.78, 1.22))
    bias = float(generator.uniform(-24.0, 24.0))
    return np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def _apply_blur(
    image: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    sigma = float(generator.uniform(0.6, 2.0))
    return cv2.GaussianBlur(
        image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
    )


def _apply_spatial_shadow(
    image: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    center = (
        int(generator.uniform(0.2, 0.8) * (width - 1)),
        int(generator.uniform(0.2, 0.8) * (height - 1)),
    )
    axes = (
        max(4, int(generator.uniform(0.18, 0.35) * width)),
        max(4, int(generator.uniform(0.12, 0.28) * height)),
    )
    angle = float(generator.uniform(0.0, 180.0))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, thickness=-1)
    sigma = max(1.0, min(height, width) * 0.018)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask[mask < 1e-3] = 0.0
    strength = float(generator.uniform(0.22, 0.52))
    shadowed = image.astype(np.float32) * (1.0 - strength * mask[..., None])
    return np.rint(shadowed).astype(np.uint8)


def _occlude_corner(
    image: np.ndarray,
    corner: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    occluded = image.copy()
    height, width = image.shape[:2]
    radius = max(8, int(round(min(height, width) * generator.uniform(0.055, 0.10))))
    center = tuple(np.rint(corner).astype(int))
    color = tuple(int(value) for value in generator.integers(0, 256, size=3))
    before = occluded.copy()
    cv2.circle(occluded, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    if np.array_equal(before, occluded):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=-1)
        occluded[mask == 1] = 255 - occluded[mask == 1]
    return occluded


def _gaussian_heatmaps(corners: np.ndarray, size: int, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    centers = corners * np.float32(size - 1)
    heatmaps = np.empty((4, size, size), dtype=np.float32)
    for index, (center_x, center_y) in enumerate(centers):
        squared_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        shifted_distance = squared_distance.astype(np.float64)
        shifted_distance -= shifted_distance.min()
        with np.errstate(over="ignore", under="ignore"):
            scaled_distance = np.sqrt(shifted_distance) / sigma
            heatmaps[index] = np.exp(-0.5 * scaled_distance**2).astype(
                np.float32
            )
    return heatmaps


def _quadrilateral_mask(corners: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    polygon = np.rint(corners * np.float32(size - 1)).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, 1.0, lineType=cv2.LINE_8)
    return mask[None]


class SmartDocCornerDataset(
    Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]
):
    """SmartDoc frames with deterministic geometry and complete-corner targets."""

    def __init__(
        self,
        records: Sequence[SmartDocRecord],
        *,
        training: bool,
        global_seed: int = SMARTDOC_GLOBAL_SEED,
        rotation_choices: Sequence[int] = (0, 90, 180, 270),
        zoom_out_probability: float = 0.5,
        zoom_out_scale_range: tuple[float, float] = (0.45, 0.82),
        occlusion_probability: float = 0.5,
        occlusion_corner: int | None = None,
        photometric_probability: float = 1.0,
        blur_probability: float = 0.35,
        shadow_probability: float = 0.4,
        heatmap_sigma: float = 1.5,
    ) -> None:
        stored_records = tuple(records)
        if not stored_records:
            raise ValueError("records must be non-empty")
        if any(not isinstance(record, SmartDocRecord) for record in stored_records):
            raise TypeError("records must contain only SmartDocRecord instances")
        rotations = tuple(int(value) for value in rotation_choices)
        if not rotations or any(value not in (0, 90, 180, 270) for value in rotations):
            raise ValueError("rotation_choices must contain only 0, 90, 180, and 270")
        for name, probability in (
            ("zoom_out_probability", zoom_out_probability),
            ("occlusion_probability", occlusion_probability),
            ("photometric_probability", photometric_probability),
            ("blur_probability", blur_probability),
            ("shadow_probability", shadow_probability),
        ):
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        minimum_scale, maximum_scale = zoom_out_scale_range
        if (
            not np.isfinite((minimum_scale, maximum_scale)).all()
            or minimum_scale <= 0.0
            or minimum_scale > maximum_scale
            or maximum_scale > 1.0
        ):
            raise ValueError("zoom_out_scale_range must satisfy 0 < min <= max <= 1")
        if occlusion_corner is not None and occlusion_corner not in range(4):
            raise ValueError("occlusion_corner must be between 0 and 3")
        if not np.isfinite(heatmap_sigma) or heatmap_sigma <= 0.0:
            raise ValueError("heatmap_sigma must be positive")

        self.records = stored_records
        self.training = bool(training)
        self.global_seed = int(global_seed)
        self.input_size = SMARTDOC_INPUT_SIZE
        self.target_size = SMARTDOC_TARGET_SIZE
        self.rotation_choices = rotations
        self.zoom_out_probability = float(zoom_out_probability)
        self.zoom_out_scale_range = (float(minimum_scale), float(maximum_scale))
        self.occlusion_probability = float(occlusion_probability)
        self.occlusion_corner = occlusion_corner
        self.photometric_probability = float(photometric_probability)
        self.blur_probability = float(blur_probability)
        self.shadow_probability = float(shadow_probability)
        self.heatmap_sigma = float(heatmap_sigma)
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic augmentation stream for a training epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch.fill_(epoch)

    def _generator(self, record: SmartDocRecord) -> np.random.Generator:
        identity = "\0".join(
            (
                str(self.global_seed),
                str(int(self._epoch.item())),
                record.background,
                record.sequence,
                str(record.frame_index),
            )
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(identity).digest()[:16], "little")
        return np.random.default_rng(seed)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        expected = (
            None
            if record.image_sha256 is None or record.image_size_bytes is None
            else VerifiedSmartDocFile(
                size_bytes=record.image_size_bytes,
                sha256=record.image_sha256,
            )
        )
        try:
            if record.dataset_root is not None and record.image_relative_path is not None:
                image_payload, _ = _read_root_relative_file(
                    record.dataset_root,
                    record.image_relative_path,
                    expected,
                )
            else:
                image_payload, _ = _read_regular_file(record.image_path, expected)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        encoded = _decode_image(image_payload)
        if encoded is None:
            raise RuntimeError(f"unable to read SmartDoc image: {record.image_path}")
        image = cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        _validate_quad(record.corners, width=width, height=height)

        image = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        corners = record.corners.astype(np.float32).copy()
        corners *= np.asarray(
            (
                (self.input_size - 1.0) / (width - 1.0),
                (self.input_size - 1.0) / (height - 1.0),
            ),
            dtype=np.float32,
        )

        if self.training:
            generator = self._generator(record)
            rotation = int(generator.choice(self.rotation_choices))
            image, corners = _rotate_image_and_corners(image, corners, rotation)
            if generator.random() < self.zoom_out_probability:
                scale = float(generator.uniform(*self.zoom_out_scale_range))
                image, corners = _zoom_out_on_clutter(
                    image, corners, scale=scale, generator=generator
                )
            if generator.random() < self.photometric_probability:
                image = _apply_photometric_augmentation(image, generator)
            if generator.random() < self.blur_probability:
                image = _apply_blur(image, generator)
            if generator.random() < self.shadow_probability:
                image = _apply_spatial_shadow(image, generator)
            if generator.random() < self.occlusion_probability:
                corner_index = (
                    self.occlusion_corner
                    if self.occlusion_corner is not None
                    else int(generator.integers(0, 4))
                )
                image = _occlude_corner(image, corners[corner_index], generator)

        normalized_corners = corners / np.float32(self.input_size - 1)
        normalized_corners = normalized_corners.astype(np.float32)
        heatmaps = _gaussian_heatmaps(
            normalized_corners, self.target_size, self.heatmap_sigma
        )
        mask = _quadrilateral_mask(normalized_corners, self.target_size)

        normalized_image = image.astype(np.float32) / 255.0
        normalized_image = (normalized_image - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(
            np.transpose(normalized_image, (2, 0, 1)).copy()
        )
        targets = {
            "heatmaps": torch.from_numpy(heatmaps.copy()),
            "mask": torch.from_numpy(mask.copy()),
            "corners": torch.from_numpy(normalized_corners.copy()),
        }
        return image_tensor, targets
