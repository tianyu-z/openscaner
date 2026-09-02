"""Independent evaluation of immutable document-boundary benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity

from openscaner.fusion.artifacts import FUSION_DEPENDENCY_ARTIFACTS, FUSION_POLICY_ARTIFACT
from openscaner.prompted_sam.artifacts import (
    PROMPTED_SAM_CALIBRATION_REPORT_ARTIFACT,
    PROMPTED_SAM_DEPENDENCY_ARTIFACTS,
    PROMPTED_SAM_POLICY_ARTIFACT,
)


SCHEMA_VERSION = 1
CORNER_RMSE_THRESHOLD = 0.025
SSIM_THRESHOLD = 0.90
MAX_RESULT_BYTES = 1_000_000
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_MODEL_BYTES = 128 * 1024 * 1024
_RESULT_FIELDS = frozenset(
    {
        "name",
        "status",
        "corners",
        "confidence",
        "backend",
        "elapsed_ms",
        "peak_rss_mb",
        "error",
        "diagnostics",
    }
)
_SEMANTIC_RESULT_FIELDS = _RESULT_FIELDS - frozenset({"elapsed_ms", "peak_rss_mb"})
_CPU_BACKENDS = frozenset({"CPUExecutionProvider", "torch:cpu", "opencv:cpu"})
_SUCCESS_FILES = frozenset({"result.json", "rectified.jpg", "overlay.jpg"})
_FAILURE_FILES = frozenset({"result.json"})
REQUIRED_CANDIDATES = (
    "docaligner",
    "docaligner_pp_lcnet_fusion",
    "docaligner_prompted_mobile_sam",
    "fast_scnn",
    "lraspp_mobilenetv3_small",
    "mobile_sam",
    "mobilenetv3_small_corner",
    "oss_contour",
    "pp_lcnet_050_corner",
    "pp_liteseg_t",
    "yolo11n_seg",
)
_FUSION_ADAPTER_NAME = FUSION_POLICY_ARTIFACT.adapter
_PROMPTED_SAM_ADAPTER_NAME = PROMPTED_SAM_POLICY_ARTIFACT.adapter


class ArtifactValidationError(ValueError):
    """Raised when an evaluator input is malformed or unsafe."""


class EvaluationIntegrityError(RuntimeError):
    """Raised when candidate artifacts change during or after evaluation."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_kind(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _open_regular_descriptor(path: Path, label: str) -> tuple[int, os.stat_result]:
    kind = _path_kind(path)
    if kind == "missing":
        raise ArtifactValidationError(f"missing {label}")
    if kind == "symlink":
        raise ArtifactValidationError(f"{label} must not be a symlink")
    if kind != "file":
        raise ArtifactValidationError(f"{label} must be a regular file")
    metadata = path.stat(follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ArtifactValidationError(f"{label} must be a regular file")
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ArtifactValidationError(f"{label} changed while opening")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _read_opened_bytes(
    descriptor: int,
    opened: os.stat_result,
    maximum: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    if opened.st_size > maximum:
        raise ArtifactValidationError(f"{label} is too large")
    payload = bytearray()
    while chunk := os.read(descriptor, min(1024 * 1024, maximum + 1 - len(payload))):
        payload.extend(chunk)
        if len(payload) > maximum:
            raise ArtifactValidationError(f"{label} is too large")
    finished = os.fstat(descriptor)
    if (
        (finished.st_dev, finished.st_ino) != (opened.st_dev, opened.st_ino)
        or finished.st_size != opened.st_size
        or finished.st_mtime_ns != opened.st_mtime_ns
        or len(payload) != finished.st_size
    ):
        raise ArtifactValidationError(f"{label} changed while reading")
    return bytes(payload), finished


def _read_regular_file(
    path: Path,
    maximum: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    descriptor, opened = _open_regular_descriptor(path, label)
    try:
        return _read_opened_bytes(descriptor, opened, maximum, label)
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, maximum: int, label: str) -> bytes:
    payload, _ = _read_regular_file(path, maximum, label)
    return payload


def _read_snapshot_bound_bytes(
    path: Path,
    maximum: int,
    label: str,
    snapshot: Mapping[str, object],
) -> bytes:
    payload, _ = _read_regular_file(path, maximum, label)
    if (
        snapshot.get("size_bytes") != len(payload)
        or snapshot.get("sha256") != _sha256_bytes(payload)
    ):
        raise ArtifactValidationError(f"{label} does not match initial snapshot")
    return payload


def _decode_image_bytes(payload: bytes, label: str, flags: int) -> np.ndarray:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None or image.size == 0:
        raise ArtifactValidationError(f"{label} is not a valid image")
    if min(image.shape[:2]) < 7:
        raise ArtifactValidationError(f"{label} is too tiny to score safely")
    return image


def _read_image(
    path: Path,
    label: str,
    flags: int = cv2.IMREAD_COLOR,
    snapshot: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, bytes]:
    payload = (
        _read_regular_bytes(path, MAX_IMAGE_BYTES, label)
        if snapshot is None
        else _read_snapshot_bound_bytes(path, MAX_IMAGE_BYTES, label, snapshot)
    )
    return _decode_image_bytes(payload, label, flags), payload


def _grayscale(image: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in {2, 3} or array.size == 0:
        raise ArtifactValidationError(f"{label} has an invalid image shape")
    if min(array.shape[:2]) < 7:
        raise ArtifactValidationError(f"{label} is too tiny to score safely")
    if not np.isfinite(array).all():
        raise ArtifactValidationError(f"{label} must contain only finite values")
    if array.ndim == 3:
        if array.shape[2] == 3:
            array = cv2.cvtColor(array.astype(np.float32), cv2.COLOR_BGR2GRAY)
        elif array.shape[2] == 4:
            array = cv2.cvtColor(array.astype(np.float32), cv2.COLOR_BGRA2GRAY)
        elif array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            raise ArtifactValidationError(f"{label} has an invalid channel count")
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def compute_grayscale_ssim(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Score a candidate after deterministic resize to exact reference dimensions."""
    candidate_gray = _grayscale(candidate, "candidate image")
    reference_gray = _grayscale(reference, "reference image")
    height, width = reference_gray.shape
    resized = cv2.resize(candidate_gray, (width, height), interpolation=cv2.INTER_AREA)
    score = float(structural_similarity(reference_gray, resized, data_range=255))
    if not math.isfinite(score):
        raise ArtifactValidationError("SSIM produced a non-finite result")
    return score


def _validated_corners(
    corners: object,
    source_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    try:
        array = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(f"{label} must contain numeric corners") from error
    if array.shape != (4, 2):
        raise ArtifactValidationError(f"{label} must contain exactly four 2D corners")
    if not np.isfinite(array).all():
        raise ArtifactValidationError(f"{label} must contain only finite values")
    height, width = source_shape
    if height < 1 or width < 1:
        raise ArtifactValidationError("source dimensions must be positive")
    if (
        (array[:, 0] < 0).any()
        or (array[:, 0] > width - 1).any()
        or (array[:, 1] < 0).any()
        or (array[:, 1] > height - 1).any()
    ):
        raise ArtifactValidationError(f"{label} must lie within source bounds")
    shifted = np.roll(array, -1, axis=0)
    signed_area = 0.5 * float(
        np.sum(array[:, 0] * shifted[:, 1] - array[:, 1] * shifted[:, 0])
    )
    if abs(signed_area) < 1e-6:
        raise ArtifactValidationError(f"{label} must form a nondegenerate quadrilateral")
    return array


def normalized_corner_rmse(
    candidate_corners: object,
    reference_corners: object,
    source_shape: tuple[int, int],
) -> float:
    """Return cyclic-start-only corner RMSE normalized by source diagonal."""
    candidate = _validated_corners(candidate_corners, source_shape, "candidate corners")
    reference = _validated_corners(reference_corners, source_shape, "reference corners")
    errors = []
    for offset in range(4):
        aligned = np.roll(candidate, -offset, axis=0)
        squared_distances = np.sum((aligned - reference) ** 2, axis=1)
        errors.append(math.sqrt(float(np.mean(squared_distances))))
    diagonal = math.hypot(*source_shape)
    return min(errors) / diagonal


def _snapshot_file(path: Path) -> dict[str, object]:
    kind = _path_kind(path)
    if kind == "file":
        descriptor, metadata = _open_regular_descriptor(
            path,
            f"candidate artifact {path.name}",
        )
        try:
            payload = None
            if metadata.st_size <= MAX_IMAGE_BYTES:
                payload, metadata = _read_opened_bytes(
                    descriptor,
                    metadata,
                    MAX_IMAGE_BYTES,
                    f"candidate artifact {path.name}",
                )
        finally:
            os.close(descriptor)
    else:
        metadata = path.lstat()
        payload = None
    record: dict[str, object] = {
        "kind": kind,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "mode": stat.S_IMODE(metadata.st_mode),
    }
    if kind == "file":
        record["device"] = metadata.st_dev
        record["inode"] = metadata.st_ino
    if payload is not None:
        record["sha256"] = _sha256_bytes(payload)
    elif kind == "symlink":
        record["target"] = os.readlink(path)
    else:
        record["sha256"] = None
    return record


def snapshot_candidate_artifacts(
    artifacts_dir: Path,
    candidate_names: Sequence[str] = REQUIRED_CANDIDATES,
) -> dict[str, object]:
    """Snapshot candidate artifact metadata and hashes without following links."""
    root_kind = _path_kind(artifacts_dir)
    if root_kind == "symlink":
        raise ArtifactValidationError("candidate artifacts root must not be a symlink")
    if root_kind != "directory":
        raise ArtifactValidationError("candidate artifacts root must be a real directory")

    snapshot: dict[str, object] = {}
    for name in sorted(set(candidate_names)):
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ArtifactValidationError(f"invalid candidate name: {name!r}")
        candidate = artifacts_dir / name
        kind = _path_kind(candidate)
        entry: dict[str, object] = {"kind": kind, "files": {}}
        if kind == "directory":
            metadata = candidate.lstat()
            entry["mtime_ns"] = metadata.st_mtime_ns
            entry["mode"] = stat.S_IMODE(metadata.st_mode)
            files: dict[str, object] = {}
            for path in sorted(candidate.iterdir(), key=lambda item: item.name):
                files[path.name] = _snapshot_file(path)
            entry["files"] = files
        elif kind == "symlink":
            entry["target"] = os.readlink(candidate)
        snapshot[name] = entry
    return snapshot


def verify_candidate_artifacts_unchanged(
    artifacts_dir: Path,
    expected_snapshot: Mapping[str, object],
) -> None:
    actual = snapshot_candidate_artifacts(artifacts_dir, tuple(expected_snapshot))
    if actual != dict(expected_snapshot):
        raise EvaluationIntegrityError("candidate artifacts changed after inference")


def _reject_json_constant(value: str) -> None:
    raise ArtifactValidationError(f"non-finite JSON number: {value}")


def _read_json(
    path: Path,
    maximum: int,
    label: str,
    snapshot: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], bytes]:
    payload = (
        _read_regular_bytes(path, maximum, label)
        if snapshot is None
        else _read_snapshot_bound_bytes(path, maximum, label, snapshot)
    )
    try:
        decoded = json.loads(payload, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"{label} must contain valid JSON") from error
    if not isinstance(decoded, dict):
        raise ArtifactValidationError(f"{label} must contain a JSON object")
    return decoded, payload


def _finite_number(value: object, label: str, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        raise ArtifactValidationError(f"{label} is outside its valid range")
    return number


def _artifact_records(snapshot_entry: Mapping[str, object]) -> dict[str, object]:
    files = snapshot_entry.get("files", {})
    if not isinstance(files, dict):
        return {}
    return {
        name: {
            "sha256": metadata.get("sha256"),
            "size_bytes": metadata.get("size_bytes"),
            "dimensions": None,
        }
        for name, metadata in sorted(files.items())
        if isinstance(metadata, dict)
    }


def _candidate_digest(artifacts: Mapping[str, object]) -> str | None:
    if not artifacts:
        return None
    digest = hashlib.sha256()
    for name, metadata in sorted(artifacts.items()):
        if not isinstance(metadata, dict) or not isinstance(metadata.get("sha256"), str):
            return None
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(metadata["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _invalid_record(
    name: str,
    error: BaseException,
    snapshot_entry: Mapping[str, object],
) -> dict[str, object]:
    artifacts = _artifact_records(snapshot_entry)
    return {
        "name": name,
        "status": "invalid_artifact",
        "corners": None,
        "error": f"{type(error).__name__}: {error}",
        "normalized_corner_rmse": None,
        "ssim": None,
        "corner_pass": False,
        "ssim_pass": False,
        "accepted": False,
        "confidence": None,
        "backend": None,
        "elapsed_ms": None,
        "peak_rss_mb": None,
        "orientation": None,
        "artifacts": artifacts,
        "candidate_sha256": _candidate_digest(artifacts),
        "rank": None,
    }


def _validate_snapshot_entry(name: str, entry: Mapping[str, object]) -> None:
    kind = entry.get("kind")
    if kind == "missing":
        raise ArtifactValidationError(f"missing candidate directory: {name}")
    if kind == "symlink":
        raise ArtifactValidationError(f"candidate directory must not be a symlink: {name}")
    if kind != "directory":
        raise ArtifactValidationError(f"candidate path must be a real directory: {name}")
    files = entry.get("files")
    if not isinstance(files, dict):
        raise ArtifactValidationError("candidate snapshot is malformed")
    for filename, metadata in files.items():
        if filename not in _SUCCESS_FILES:
            raise ArtifactValidationError(f"unexpected candidate artifact: {filename}")
        if not isinstance(metadata, dict) or metadata.get("kind") != "file":
            kind = metadata.get("kind") if isinstance(metadata, dict) else "unknown"
            if kind == "symlink":
                raise ArtifactValidationError(f"candidate artifact must not be a symlink: {filename}")
            raise ArtifactValidationError(f"candidate artifact must be a regular file: {filename}")


def _parse_candidate(
    name: str,
    candidate_dir: Path,
    snapshot_entry: Mapping[str, object],
    source_shape: tuple[int, int],
    reference_corners: np.ndarray,
    reference_gray: np.ndarray,
) -> tuple[dict[str, object], np.ndarray | None]:
    _validate_snapshot_entry(name, snapshot_entry)
    snapshot_files = snapshot_entry["files"]
    result_snapshot = snapshot_files.get("result.json")
    result, _ = _read_json(
        candidate_dir / "result.json",
        MAX_RESULT_BYTES,
        "result.json",
        result_snapshot if isinstance(result_snapshot, Mapping) else None,
    )
    result_fields = set(result)
    if result_fields != _RESULT_FIELDS and result_fields != _SEMANTIC_RESULT_FIELDS:
        raise ArtifactValidationError("result.json has an invalid schema")
    if result["name"] != name:
        raise ArtifactValidationError("result.json candidate name does not match its directory")
    status_value = result["status"]
    if status_value not in {"ok", "not_detected", "unavailable", "error"}:
        raise ArtifactValidationError("result.json has an invalid status")
    status = str(status_value)
    confidence = _finite_number(result["confidence"], "confidence", 1.0)
    elapsed_ms = (
        _finite_number(result["elapsed_ms"], "elapsed_ms")
        if "elapsed_ms" in result
        else None
    )
    peak_rss_mb = (
        _finite_number(result["peak_rss_mb"], "peak_rss_mb")
        if "peak_rss_mb" in result
        else None
    )
    diagnostics = result["diagnostics"]
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ArtifactValidationError("diagnostics must be an object or null")
    files = set(snapshot_entry["files"])
    expected_files = _SUCCESS_FILES if status == "ok" else _FAILURE_FILES
    if files != expected_files:
        if status == "ok" and "rectified.jpg" not in files:
            raise ArtifactValidationError("missing rectified.jpg for successful candidate")
        raise ArtifactValidationError(f"candidate files are inconsistent with status {status}")

    artifacts = _artifact_records(snapshot_entry)
    base: dict[str, object] = {
        "name": name,
        "status": status,
        "corners": None,
        "error": result["error"],
        "normalized_corner_rmse": None,
        "ssim": None,
        "corner_pass": False,
        "ssim_pass": False,
        "accepted": False,
        "confidence": confidence,
        "backend": result["backend"],
        "elapsed_ms": elapsed_ms,
        "peak_rss_mb": peak_rss_mb,
        "orientation": diagnostics.get("orientation") if isinstance(diagnostics, dict) else None,
        "artifacts": artifacts,
        "candidate_sha256": _candidate_digest(artifacts),
        "rank": None,
    }
    if status != "ok":
        if result["corners"] is not None:
            raise ArtifactValidationError("failed candidate must not contain corners")
        if status == "not_detected":
            if result["backend"] not in _CPU_BACKENDS or result["error"] is not None:
                raise ArtifactValidationError("not_detected result has inconsistent fields")
        elif result["backend"] is not None or not isinstance(result["error"], str) or not result["error"]:
            raise ArtifactValidationError("failed candidate result has inconsistent fields")
        return base, None

    if result["backend"] not in _CPU_BACKENDS or result["error"] is not None:
        raise ArtifactValidationError("successful candidate result has inconsistent fields")
    corners = _validated_corners(result["corners"], source_shape, "candidate corners")
    image, image_payload = _read_image(
        candidate_dir / "rectified.jpg",
        "rectified.jpg",
        snapshot=snapshot_files["rectified.jpg"],
    )
    overlay, _ = _read_image(
        candidate_dir / "overlay.jpg",
        "overlay.jpg",
        snapshot=snapshot_files["overlay.jpg"],
    )
    image_gray = _decode_image_bytes(image_payload, "rectified.jpg", cv2.IMREAD_GRAYSCALE)
    rmse = normalized_corner_rmse(corners, reference_corners, source_shape)
    ssim = compute_grayscale_ssim(image_gray, reference_gray)
    corner_pass = rmse <= CORNER_RMSE_THRESHOLD
    ssim_pass = ssim >= SSIM_THRESHOLD
    base.update(
        {
            "corners": corners.tolist(),
            "normalized_corner_rmse": rmse,
            "ssim": ssim,
            "corner_pass": corner_pass,
            "ssim_pass": ssim_pass,
            "accepted": corner_pass and ssim_pass,
        }
    )
    artifacts["rectified.jpg"]["dimensions"] = {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }
    artifacts["overlay.jpg"]["dimensions"] = {
        "width": int(overlay.shape[1]),
        "height": int(overlay.shape[0]),
    }
    return base, image


def _rank(records: list[dict[str, object]]) -> list[str]:
    def key(record: Mapping[str, object]) -> tuple[object, ...]:
        scored = record["status"] == "ok" and record["normalized_corner_rmse"] is not None
        if not scored:
            return (1, 0, math.inf, 0.0, str(record["name"]))
        passes = int(bool(record["corner_pass"])) + int(bool(record["ssim_pass"]))
        return (
            0,
            -passes,
            float(record["normalized_corner_rmse"]),
            -float(record["ssim"]),
            str(record["name"]),
        )

    ranked = sorted(records, key=key)
    for index, record in enumerate(ranked, start=1):
        record["rank"] = index
    return [str(record["name"]) for record in ranked]


def _fit_panel_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _comparison_image(
    reference_image: np.ndarray,
    records: Sequence[Mapping[str, object]],
    successful_images: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    panel_width, panel_height, header_height = 420, 300, 84
    panels: list[np.ndarray] = []
    names = ["reference"]
    ordered = sorted(records, key=lambda record: int(record["rank"]))
    panel_specs: list[tuple[str, Mapping[str, object] | None, np.ndarray | None]] = [
        ("reference", None, reference_image)
    ]
    panel_specs.extend(
        (str(record["name"]), record, successful_images.get(str(record["name"])))
        for record in ordered
    )
    for name, record, image in panel_specs:
        panel = np.full((header_height + panel_height, panel_width, 3), 245, dtype=np.uint8)
        cv2.putText(panel, name, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        if record is None:
            detail = "frozen target reference"
        elif image is not None:
            detail = (
                f"RMSE {float(record['normalized_corner_rmse']):.6f}  "
                f"SSIM {float(record['ssim']):.6f}  "
                f"accepted {str(bool(record['accepted'])).lower()}"
            )
        else:
            detail = f"{record['status']}: {record['error'] or 'no detection'}"
        cv2.putText(
            panel,
            detail[:62],
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (35, 35, 180) if record is not None and image is None else (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        if image is not None:
            panel[header_height:] = _fit_panel_image(image, panel_width, panel_height)
        else:
            cv2.rectangle(
                panel,
                (12, header_height + 12),
                (panel_width - 12, header_height + panel_height - 12),
                (70, 70, 180),
                2,
            )
            cv2.putText(
                panel,
                "STRUCTURED FAILURE",
                (72, header_height + panel_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (70, 70, 180),
                2,
                cv2.LINE_AA,
            )
        panels.append(panel)
        if record is not None:
            names.append(name)

    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    canvas = np.full((rows * panels[0].shape[0], columns * panel_width, 3), 225, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        y, x = row * panel.shape[0], column * panel_width
        canvas[y : y + panel.shape[0], x : x + panel_width] = panel
    return canvas, names


def _stage_output(path: Path, payload: bytes) -> Path:
    destination_mode = (
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if _path_kind(path) == "file"
        else 0o644
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), destination_mode)
            stream.write(payload)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_output_pair(
    comparison_path: Path,
    comparison_bytes: bytes,
    summary_path: Path,
    summary_bytes: bytes,
) -> list[tuple[Path, Path]]:
    staged: list[tuple[Path, Path]] = []
    try:
        staged.append((_stage_output(comparison_path, comparison_bytes), comparison_path))
        staged.append((_stage_output(summary_path, summary_bytes), summary_path))
    except BaseException:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    return staged


def _stage_output_backups(
    destinations: Sequence[Path],
) -> dict[Path, Path | None]:
    backups: dict[Path, Path | None] = {}
    try:
        for destination in destinations:
            kind = _path_kind(destination)
            if kind == "missing":
                backups[destination] = None
                continue
            if kind != "file":
                raise ArtifactValidationError("existing evaluator output must be a regular file")
            payload = _read_regular_bytes(
                destination,
                MAX_IMAGE_BYTES,
                f"existing evaluator output {destination.name}",
            )
            backups[destination] = _stage_output(destination, payload)
    except BaseException:
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)
        raise
    return backups


def _publish_staged_output_pair(staged: Sequence[tuple[Path, Path]]) -> None:
    backups = _stage_output_backups(tuple(destination for _, destination in staged))
    published: list[Path] = []
    try:
        for temporary, destination in staged:
            os.replace(temporary, destination)
            published.append(destination)
    except BaseException:
        rollback_errors: list[BaseException] = []
        for destination in reversed(published):
            backup = backups[destination]
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except BaseException as error:
                rollback_errors.append(error)
        if rollback_errors:
            raise EvaluationIntegrityError("failed to roll back evaluator outputs") from rollback_errors[0]
        raise
    finally:
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)


def _image_dimensions(image: np.ndarray) -> dict[str, int]:
    return {"width": int(image.shape[1]), "height": int(image.shape[0])}


def _load_gold(
    path: Path,
    source_hash: str,
    source_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object], str]:
    payload, raw = _read_json(path, MAX_RESULT_BYTES, "reference corners")
    expected = {
        "schema_version",
        "source",
        "source_sha256",
        "coordinate_space",
        "order",
        "corners",
        "provenance",
    }
    if set(payload) != expected or payload["schema_version"] != 1:
        raise ArtifactValidationError("reference corners have an invalid schema")
    if payload["source_sha256"] != source_hash:
        raise ArtifactValidationError("reference corners source hash does not match source image")
    if payload["coordinate_space"] != "source_pixels" or payload["order"] != ["TL", "TR", "BR", "BL"]:
        raise ArtifactValidationError("reference corners have invalid coordinate metadata")
    if not isinstance(payload["provenance"], str) or not payload["provenance"].strip():
        raise ArtifactValidationError("reference corners must record provenance")
    corners = _validated_corners(payload["corners"], source_shape, "reference corners")
    return corners, payload, _sha256_bytes(raw)


def _manifest_entries(
    payload: Mapping[str, object],
    field: str,
    identity_field: str,
    expected_names: set[str],
    *,
    allowed_names: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    entries = payload.get(field)
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ArtifactValidationError(f"model manifest {field} must be a list of objects")
    indexed: dict[str, dict[str, object]] = {}
    for entry in entries:
        name = entry.get(identity_field)
        if not isinstance(name, str) or not name or name in indexed:
            raise ArtifactValidationError(f"model manifest {field} has an invalid identity")
        indexed[name] = entry
    actual_names = set(indexed)
    allowed = expected_names if allowed_names is None else allowed_names
    if not expected_names <= actual_names or not actual_names <= allowed:
        raise ArtifactValidationError("model manifest is missing required model entries")
    return indexed


def _validate_fusion_manifest_entry(
    entry: Mapping[str, object],
    candidates: Mapping[str, Mapping[str, object]],
) -> None:
    policy = entry.get("fusion_policy")
    expected_policy = FUSION_POLICY_ARTIFACT.as_document()
    if policy != expected_policy:
        raise ArtifactValidationError("fusion policy manifest identity is invalid")
    if (
        entry.get("local_filename") != FUSION_POLICY_ARTIFACT.filename
        or entry.get("sha256") != FUSION_POLICY_ARTIFACT.sha256
        or entry.get("checkpoint_size_bytes")
        != FUSION_POLICY_ARTIFACT.size_bytes
    ):
        raise ArtifactValidationError("fusion policy model identity is invalid")

    calibration = entry.get("calibration")
    dependencies = (
        calibration.get("models") if isinstance(calibration, Mapping) else None
    )
    expected_dependencies = {
        name: identity.as_document()
        for name, identity in FUSION_DEPENDENCY_ARTIFACTS.items()
    }
    if dependencies != expected_dependencies:
        raise ArtifactValidationError("fusion dependency model identities are invalid")

    for name, identity in FUSION_DEPENDENCY_ARTIFACTS.items():
        dependency = candidates.get(name)
        if dependency is None:
            raise ArtifactValidationError("fusion dependency model entry is missing")
        if (
            dependency.get("local_filename") != identity.filename
            or dependency.get("sha256") != identity.sha256
            or dependency.get("checkpoint_size_bytes") != identity.size_bytes
        ):
            raise ArtifactValidationError(
                f"fusion dependency model entry is invalid: {name}"
            )


def _validate_prompted_sam_manifest_entry(
    entry: Mapping[str, object],
    candidates: Mapping[str, Mapping[str, object]],
) -> None:
    if entry.get("calibration_report") != (
        PROMPTED_SAM_CALIBRATION_REPORT_ARTIFACT.as_document()
    ):
        raise ArtifactValidationError(
            "prompted MobileSAM calibration report manifest identity is invalid"
        )
    policy = entry.get("prompted_sam_policy")
    expected_policy = PROMPTED_SAM_POLICY_ARTIFACT.as_document()
    if policy != expected_policy:
        raise ArtifactValidationError("prompted MobileSAM policy manifest identity is invalid")
    if (
        entry.get("local_filename") != PROMPTED_SAM_POLICY_ARTIFACT.filename
        or entry.get("sha256") != PROMPTED_SAM_POLICY_ARTIFACT.sha256
        or entry.get("checkpoint_size_bytes")
        != PROMPTED_SAM_POLICY_ARTIFACT.size_bytes
    ):
        raise ArtifactValidationError("prompted MobileSAM policy model identity is invalid")

    calibration = entry.get("calibration")
    dependencies = (
        calibration.get("models") if isinstance(calibration, Mapping) else None
    )
    expected_dependencies = {
        name: identity.as_document()
        for name, identity in PROMPTED_SAM_DEPENDENCY_ARTIFACTS.items()
    }
    if dependencies != expected_dependencies:
        raise ArtifactValidationError(
            "prompted MobileSAM dependency model identities are invalid"
        )

    for name, identity in PROMPTED_SAM_DEPENDENCY_ARTIFACTS.items():
        dependency = candidates.get(name)
        if dependency is None:
            raise ArtifactValidationError("prompted MobileSAM dependency model entry is missing")
        size_field = "size_bytes" if name == "mobile_sam" else "checkpoint_size_bytes"
        if (
            dependency.get("local_filename") != identity.filename
            or dependency.get("sha256") != identity.sha256
            or dependency.get(size_field) != identity.size_bytes
        ):
            raise ArtifactValidationError(
                f"prompted MobileSAM dependency model entry is invalid: {name}"
            )


def _model_artifact_spec(
    model_dir: Path,
    entry: Mapping[str, object],
    label: str,
) -> tuple[Path, str, str] | None:
    filename = entry.get("local_filename")
    expected_hash = entry.get("sha256")
    if filename is None and expected_hash is None:
        return None
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ArtifactValidationError(f"{label} has an invalid model filename")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ArtifactValidationError(f"{label} has an invalid model SHA-256")
    return model_dir / filename, filename, expected_hash


def _verified_model_artifact(
    spec: tuple[Path, str, str] | None,
) -> tuple[
    dict[str, object] | None,
    tuple[Path, str, int, tuple[int, int]] | None,
]:
    if spec is None:
        return None, None
    model_path, filename, expected_hash = spec
    model_bytes, metadata = _read_regular_file(
        model_path,
        MAX_MODEL_BYTES,
        f"model artifact {filename}",
    )
    actual_hash = _sha256_bytes(model_bytes)
    if actual_hash != expected_hash:
        raise ArtifactValidationError(f"model artifact {filename} checksum mismatch")
    artifact = {
        "filename": filename,
        "path": str(model_path),
        "sha256": actual_hash,
        "size_bytes": len(model_bytes),
        "verified": True,
    }
    return artifact, (
        model_path,
        actual_hash,
        len(model_bytes),
        (metadata.st_dev, metadata.st_ino),
    )


def _model_record(
    entry: Mapping[str, object],
    identity_field: str,
    artifact: dict[str, object] | None,
) -> dict[str, object]:
    identity = entry.get(identity_field)
    family = entry.get("model_family")
    availability = entry.get("availability")
    if not isinstance(family, str) or not family:
        raise ArtifactValidationError(f"model manifest entry {identity!r} lacks model_family")
    if identity_field == "adapter" and identity == "oss_contour" and availability != "built_in":
        raise ArtifactValidationError("oss_contour availability must be built_in")
    if not isinstance(availability, str) or not availability:
        raise ArtifactValidationError(f"model manifest entry {identity!r} lacks availability")

    provenance: dict[str, str | None] = {}
    for field in ("source", "upstream"):
        value = entry.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ArtifactValidationError(
                f"model manifest entry {identity!r} has invalid provenance {field}"
            )
        provenance[field] = value
    if not any(provenance.values()):
        raise ArtifactValidationError(
            f"model manifest entry {identity!r} must define provenance source or upstream"
        )

    record = {
        identity_field: identity,
        "model_family": family,
        "availability": availability,
        "provenance": provenance,
        "artifact": artifact,
    }
    if identity_field == "name":
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            raise ArtifactValidationError(f"shared model {identity!r} lacks role")
        record["role"] = role
    return record


def _register_model_artifact_identity(
    identities: dict[tuple[int, int], Path],
    path: Path,
    identity: tuple[int, int],
) -> None:
    if identity in identities:
        raise ArtifactValidationError(
            "model artifacts must not reference the same filesystem object"
        )
    identities[identity] = path


def _load_model_summary(
    manifest_path: Path,
    output_paths: Sequence[Path],
) -> tuple[dict[str, object], dict[str, tuple[str, int]]]:
    if _path_kind(manifest_path.parent) == "symlink":
        raise ArtifactValidationError("model directory must not be a symlink")
    if _path_kind(manifest_path.parent) != "directory":
        raise ArtifactValidationError("model directory must be a real directory")
    manifest, manifest_bytes = _read_json(
        manifest_path,
        MAX_RESULT_BYTES,
        "model manifest",
    )
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version < 1:
        raise ArtifactValidationError("model manifest has an invalid schema version")
    candidates = _manifest_entries(
        manifest,
        "models",
        "adapter",
        set(REQUIRED_CANDIDATES),
    )
    shared = _manifest_entries(
        manifest,
        "shared_models",
        "name",
        {"document_orientation"},
    )
    candidate_specs = {
        name: _model_artifact_spec(manifest_path.parent, entry, name)
        for name, entry in candidates.items()
    }
    shared_specs = {
        name: _model_artifact_spec(manifest_path.parent, entry, name)
        for name, entry in shared.items()
    }
    _reject_output_input_overlaps(
        output_paths,
        tuple(
            spec[0]
            for spec in (*candidate_specs.values(), *shared_specs.values())
            if spec is not None
        ),
    )
    _validate_fusion_manifest_entry(candidates[_FUSION_ADAPTER_NAME], candidates)
    _validate_prompted_sam_manifest_entry(
        candidates[_PROMPTED_SAM_ADAPTER_NAME],
        candidates,
    )

    snapshots: dict[str, tuple[str, int]] = {
        str(manifest_path): (_sha256_bytes(manifest_bytes), MAX_RESULT_BYTES)
    }
    model_identities: dict[tuple[int, int], Path] = {}
    candidate_records = []
    for name in REQUIRED_CANDIDATES:
        if name not in candidates:
            continue
        entry = candidates[name]
        artifact, snapshot = _verified_model_artifact(candidate_specs[name])
        if name != "oss_contour" and artifact is None:
            raise ArtifactValidationError(f"required model artifact is missing for {name}")
        if name == "oss_contour" and artifact is not None:
            raise ArtifactValidationError("oss_contour must be recorded as a built-in model")
        if snapshot is not None:
            path, digest, size_bytes, identity = snapshot
            _register_model_artifact_identity(model_identities, path, identity)
            snapshots[str(path)] = (digest, MAX_MODEL_BYTES)
            if artifact["size_bytes"] != size_bytes:
                raise EvaluationIntegrityError("model artifact size changed while loading")
            expected_fusion_artifact = FUSION_DEPENDENCY_ARTIFACTS.get(name)
            if (
                expected_fusion_artifact is not None
                and artifact["size_bytes"] != expected_fusion_artifact.size_bytes
            ):
                raise ArtifactValidationError(
                    f"fusion dependency artifact size is invalid: {name}"
                )
        candidate_records.append(_model_record(entry, "adapter", artifact))

    shared_records = []
    for name in sorted(shared):
        entry = shared[name]
        artifact, snapshot = _verified_model_artifact(shared_specs[name])
        if artifact is None or snapshot is None:
            raise ArtifactValidationError(f"required model artifact is missing for {name}")
        path, digest, _, identity = snapshot
        _register_model_artifact_identity(model_identities, path, identity)
        snapshots[str(path)] = (digest, MAX_MODEL_BYTES)
        shared_records.append(_model_record(entry, "name", artifact))

    return (
        {
            "manifest": {
                "path": str(manifest_path),
                "schema_version": schema_version,
                "sha256": _sha256_bytes(manifest_bytes),
            },
            "candidates": candidate_records,
            "shared": shared_records,
        },
        snapshots,
    )


def _verify_model_inputs_unchanged(snapshots: Mapping[str, tuple[str, int]]) -> None:
    for path_text, (expected_hash, maximum) in snapshots.items():
        path = Path(path_text)
        try:
            payload = _read_regular_bytes(path, maximum, f"model input {path.name}")
        except ArtifactValidationError as error:
            raise EvaluationIntegrityError("model artifacts changed during evaluation") from error
        if _sha256_bytes(payload) != expected_hash:
            raise EvaluationIntegrityError("model artifacts changed during evaluation")


def _reject_output_input_overlaps(
    output_paths: Sequence[Path],
    input_paths: Sequence[Path],
) -> None:
    resolved_inputs = [(path, path.resolve(strict=False)) for path in input_paths]
    for output_path in output_paths:
        resolved_output = output_path.resolve(strict=False)
        for input_path, resolved_input in resolved_inputs:
            overlaps = resolved_output == resolved_input
            if not overlaps and _path_kind(output_path) != "missing":
                try:
                    overlaps = output_path.samefile(input_path)
                except OSError:
                    overlaps = False
            if overlaps:
                raise ArtifactValidationError("output path overlaps evaluator input")


def evaluate_saved_candidates(
    *,
    source_path: Path,
    reference_path: Path,
    reference_corners_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    model_manifest_path: Path,
    candidate_names: Sequence[str] = REQUIRED_CANDIDATES,
    benchmark_command: str,
    evaluator_command: str,
) -> dict[str, object]:
    """Evaluate saved outputs only and write deterministic summary/comparison files."""
    output_resolved = output_dir.resolve(strict=False)
    artifacts_resolved = artifacts_dir.resolve(strict=True)
    if output_resolved == artifacts_resolved or artifacts_resolved in output_resolved.parents:
        raise ArtifactValidationError("output directory must not be inside candidate artifacts")
    if _path_kind(output_dir) == "symlink":
        raise ArtifactValidationError("output directory must not be a symlink")
    comparison_path = output_dir / "comparison.jpg"
    summary_path = output_dir / "summary.json"
    output_paths = (comparison_path, summary_path)
    _reject_output_input_overlaps(
        output_paths,
        (source_path, reference_path, reference_corners_path, model_manifest_path),
    )
    model_summary, model_snapshots = _load_model_summary(
        model_manifest_path,
        output_paths,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_output_input_overlaps(
        (comparison_path, summary_path),
        tuple(Path(path) for path in model_snapshots),
    )

    source_image, source_raw = _read_image(source_path, "source image")
    reference_image, reference_raw = _read_image(reference_path, "reference image")
    reference_gray = _decode_image_bytes(reference_raw, "reference image", cv2.IMREAD_GRAYSCALE)
    source_hash = _sha256_bytes(source_raw)
    source_shape = source_image.shape[:2]
    reference_corners, gold_payload, gold_hash = _load_gold(
        reference_corners_path,
        source_hash,
        source_shape,
    )

    names = tuple(sorted(set(candidate_names)))
    initial_snapshot = snapshot_candidate_artifacts(artifacts_dir, names)
    records: list[dict[str, object]] = []
    successful_images: dict[str, np.ndarray] = {}
    for name in names:
        entry = initial_snapshot[name]
        try:
            if not isinstance(entry, dict):
                raise ArtifactValidationError("candidate snapshot is malformed")
            record, image = _parse_candidate(
                name,
                artifacts_dir / name,
                entry,
                source_shape,
                reference_corners,
                reference_gray,
            )
            if image is not None:
                successful_images[name] = image
        except (ArtifactValidationError, OSError, ValueError) as error:
            record = _invalid_record(name, error, entry if isinstance(entry, dict) else {})
        records.append(record)

    ranking = _rank(records)
    comparison_image, panel_names = _comparison_image(reference_image, records, successful_images)
    encoded_ok, encoded_comparison = cv2.imencode(
        ".jpg",
        comparison_image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not encoded_ok:
        raise RuntimeError("could not encode comparison image")
    comparison_bytes = encoded_comparison.tobytes()
    accepted = [record for record in records if record["accepted"]]
    scored = [record for record in records if record["status"] == "ok"]
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evaluator": {
            "module": "openscaner.evaluate",
            "schema_version": SCHEMA_VERSION,
            "ssim_implementation": "skimage.metrics.structural_similarity(data_range=255)",
            "resize_interpolation": "cv2.INTER_AREA",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__,
        },
        "thresholds": {
            "normalized_corner_rmse_max": CORNER_RMSE_THRESHOLD,
            "grayscale_ssim_min": SSIM_THRESHOLD,
        },
        "commands": {"benchmark": benchmark_command, "evaluator": evaluator_command},
        "models": model_summary,
        "inputs": {
            "source": {
                "path": str(source_path),
                "sha256": source_hash,
                "dimensions": _image_dimensions(source_image),
            },
            "reference": {
                "path": str(reference_path),
                "sha256": _sha256_bytes(reference_raw),
                "dimensions": _image_dimensions(reference_image),
            },
            "reference_corners": {
                "path": str(reference_corners_path),
                "sha256": gold_hash,
                "corners": reference_corners.tolist(),
                "order": gold_payload["order"],
                "coordinate_space": gold_payload["coordinate_space"],
                "provenance": gold_payload["provenance"],
            },
        },
        "required_candidates": list(REQUIRED_CANDIDATES),
        "candidates": sorted(records, key=lambda record: str(record["name"])),
        "ranking": ranking,
        "winner": (
            str(min(accepted, key=lambda item: int(item["rank"]))["name"])
            if accepted
            else None
        ),
        "best_candidate": str(min(scored, key=lambda item: int(item["rank"]))["name"]) if scored else None,
        "immutability_verified": True,
        "comparison": {
            "path": str(comparison_path),
            "sha256": _sha256_bytes(comparison_bytes),
            "dimensions": _image_dimensions(comparison_image),
            "panels": panel_names,
        },
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    staged = _stage_output_pair(
        comparison_path,
        comparison_bytes,
        summary_path,
        summary_bytes,
    )
    try:
        verify_candidate_artifacts_unchanged(artifacts_dir, initial_snapshot)
        _verify_model_inputs_unchanged(model_snapshots)
        _publish_staged_output_pair(staged)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate immutable document benchmark outputs")
    parser.add_argument("--source", dest="source_path", type=Path, required=True)
    parser.add_argument("--reference", dest="reference_path", type=Path, required=True)
    parser.add_argument(
        "--reference-corners",
        dest="reference_corners_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--artifacts", dest="artifacts_dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", dest="model_manifest_path", type=Path, required=True)
    parser.add_argument("--benchmark-command", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = _parser().parse_args(arguments)
    evaluator_command = shlex.join(
        ["uv", "run", "python", "-m", "openscaner.evaluate", *arguments]
    )
    summary = evaluate_saved_candidates(
        source_path=options.source_path,
        reference_path=options.reference_path,
        reference_corners_path=options.reference_corners_path,
        artifacts_dir=options.artifacts_dir,
        output_dir=options.output_dir,
        model_manifest_path=options.model_manifest_path,
        candidate_names=REQUIRED_CANDIDATES,
        benchmark_command=options.benchmark_command,
        evaluator_command=evaluator_command,
    )
    print(json.dumps({"winner": summary["winner"], "best_candidate": summary["best_candidate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
