"""Deterministic SmartDoc calibration for the local corner-refiner policy."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from itertools import product
import hashlib
import json
import math
import os
import re
import shlex
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch

from openscaner.refiner.policy import (
    POLICY_SCHEMA_VERSION,
    RefinerPolicy,
    apply_refiner_outputs,
)
from openscaner.training.refiner_data import (
    DEFAULT_SEED,
    RefinerCache,
    RefinerExample,
    LocalCornerRefinerDataset,
    _parse_refiner_cache,
    _read_cache_payload,
    load_verified_refiner_cache,
)
from openscaner.training.smartdoc import (
    SMARTDOC_ARCHIVE_SHA256,
    SMARTDOC_TRAIN_BACKGROUNDS,
    SMARTDOC_VALIDATION_BACKGROUND,
    SMARTDOC_VERSION,
    assert_protected_hashes_absent,
    load_smartdoc_records,
    validate_smartdoc_record_paths,
    validate_smartdoc_source_paths,
    verify_smartdoc_markers,
    verify_smartdoc_source,
)
from openscaner.training.train_corner_refiner import (
    ADAPTER_NAME as REFINER_MODEL_ADAPTER_NAME,
    MODEL_FILENAME,
    REPORT_FILENAME as TRAINING_REPORT_FILENAME,
    _MetricAccumulator,
    source_commit,
    validate_refiner_onnx_bytes,
)
from openscaner.training.train_corners import (
    OutputPaths,
    _json_bytes,
    publish_training_outputs,
)


RADIUS_RATIOS = (0.18, 0.24, 0.30)
MINIMUM_CONFIDENCES = (0.40, 0.55, 0.70)
MAXIMUM_RESIDUAL_RATIOS = (0.12, 0.20, 0.28)
ADAPTER_NAME = "docaligner_local_corner_refiner_calibration"
POLICY_FILENAME = f"{REFINER_MODEL_ADAPTER_NAME}_policy.json"
REPORT_FILENAME = f"{REFINER_MODEL_ADAPTER_NAME}.json"
_PROTECTED_PATH_TOKENS = frozenset(
    {"candidate", "evaluator", "groundtruth", "reference", "target"}
)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_TRAINING_REPORT_BYTES = 4 * 1024 * 1024
_MAX_REJECTION_REPORT_BYTES = 4 * 1024 * 1024
_CALIBRATION_MANIFEST_FIELDS = frozenset(
    {"local_corner_refiner_policy", "calibration_report", "calibration"}
)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    cache: Path
    dataset_root: Path
    archive: Path
    archive_sha256: str
    model_dir: Path
    policy_output: Path
    report_output: Path
    manifest: Path
    rejection_report: Path = Path(
        "artifacts/local-corner-refiner-calibration/"
        f"{REFINER_MODEL_ADAPTER_NAME}.rejected.json"
    )
    seed: int = DEFAULT_SEED
    cpu_threads: int = 1


@dataclass(frozen=True, slots=True)
class TrainingReportBinding:
    """One verified pre-calibration manifest entry and report payload."""

    manifest_payload: bytes
    manifest_entry: dict[str, object]
    training_report_path: Path
    training_report_payload: bytes
    manifest_identity: _ManifestSnapshotIdentity | None = None


@dataclass(frozen=True, slots=True)
class _ManifestSnapshotIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    nlink: int

    @classmethod
    def from_status(cls, status: os.stat_result) -> _ManifestSnapshotIdentity:
        return cls(
            device=status.st_dev,
            inode=status.st_ino,
            size=status.st_size,
            mtime_ns=status.st_mtime_ns,
            ctime_ns=status.st_ctime_ns,
            nlink=status.st_nlink,
        )


@dataclass(frozen=True, slots=True)
class _ManifestTrainingReportReference:
    manifest_payload: bytes
    manifest_entry: dict[str, object]
    training_report_path: Path
    manifest_identity: _ManifestSnapshotIdentity


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be at least 1") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> CalibrationConfig:
    """Parse the public CPU-only calibration command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--policy-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--rejection-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cpu-threads", type=_positive_int, default=1)
    return CalibrationConfig(**vars(parser.parse_args(argv)))


@dataclass(frozen=True, slots=True)
class CachedRefinerOutput:
    """One verified validation example and its single static ONNX invocation."""

    example: RefinerExample
    corner_logits: object
    edge_logits: object
    model_confidence: object


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in Path(os.path.abspath(path)).parts:
        normalized = part.lower().replace("ground-truth", "groundtruth")
        tokens.update(token for token in _TOKEN_SPLIT.split(normalized) if token)
    return tokens


def validate_calibration_paths(
    dataset_root: Path,
    archive_path: Path,
    cache_path: Path,
    model_dir: Path,
) -> None:
    """Fail closed on protected path concepts before opening any input."""
    _validate_protected_paths(
        (
            ("dataset", Path(dataset_root)),
            ("archive", Path(archive_path)),
            ("cache", Path(cache_path)),
            ("model", Path(model_dir)),
        )
    )


def _validate_protected_paths(
    paths: Sequence[tuple[str, Path]],
) -> None:
    for description, path in paths:
        overlap = _path_tokens(path) & _PROTECTED_PATH_TOKENS
        if overlap:
            raise ValueError(
                f"protected {description} path token is forbidden: {sorted(overlap)}"
            )


def _paths_collide(first: Path, second: Path) -> bool:
    first_absolute = Path(os.path.abspath(first))
    second_absolute = Path(os.path.abspath(second))
    if (
        first_absolute == second_absolute
        or first_absolute in second_absolute.parents
        or second_absolute in first_absolute.parents
    ):
        return True
    if first_absolute.exists() and second_absolute.exists():
        try:
            return first_absolute.samefile(second_absolute)
        except OSError as error:
            raise ValueError("unable to compare calibration input and output paths") from error
    return False


def _manifest_report_path(value: object, *, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest training report path is invalid")
    raw = Path(value)
    report_path = Path(
        os.path.abspath(raw if raw.is_absolute() else manifest_path.parent / raw)
    )
    if report_path.name != TRAINING_REPORT_FILENAME:
        raise ValueError("manifest training report filename is invalid")
    _validate_protected_paths((("training report", report_path),))
    return report_path


def _precalibration_manifest_entry(
    manifest_payload: bytes, *, manifest_path: Path
) -> tuple[dict[str, object], Path]:
    manifest = _strict_json_object(
        manifest_payload, description="pre-calibration manifest"
    )
    if (
        manifest.get("schema_version") != 3
        or not isinstance(manifest.get("models"), list)
    ):
        raise ValueError("pre-calibration manifest schema is invalid")
    models = manifest["models"]
    if not all(isinstance(item, dict) for item in models):
        raise ValueError("pre-calibration manifest models are invalid")
    if any(
        item.get("adapter") == f"{REFINER_MODEL_ADAPTER_NAME}_calibration"
        for item in models
    ):
        raise ValueError("pre-calibration manifest contains legacy calibration entry")
    matches = [
        item for item in models if item.get("adapter") == REFINER_MODEL_ADAPTER_NAME
    ]
    if len(matches) != 1:
        raise ValueError(
            "pre-calibration manifest must contain one refiner model entry"
        )
    entry = matches[0]
    if (
        entry.get("local_filename") != MODEL_FILENAME
        or type(entry.get("checkpoint_size_bytes")) is not int
        or entry["checkpoint_size_bytes"] < 1
        or not isinstance(entry.get("sha256"), str)
        or len(entry["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in entry["sha256"])
    ):
        raise ValueError("pre-calibration manifest refiner model identity is invalid")
    training_report = entry.get("training_report")
    if not isinstance(training_report, Mapping) or set(training_report) != {
        "filename",
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("pre-calibration manifest training report binding is invalid")
    if training_report["filename"] != TRAINING_REPORT_FILENAME:
        raise ValueError("manifest training report filename is invalid")
    if (
        type(training_report["size_bytes"]) is not int
        or training_report["size_bytes"] < 1
        or not isinstance(training_report["sha256"], str)
        or len(training_report["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in training_report["sha256"]
        )
    ):
        raise ValueError("manifest training report identity is invalid")
    report_path = _manifest_report_path(
        training_report["path"], manifest_path=manifest_path
    )
    return entry, report_path


def _manifest_training_report_reference(
    config: CalibrationConfig,
) -> _ManifestTrainingReportReference:
    manifest_path = Path(os.path.abspath(config.manifest))
    manifest_snapshot = _input_regular_manifest_snapshot(
        manifest_path,
        description="pre-calibration manifest",
        maximum=_MAX_MANIFEST_BYTES,
    )
    entry, report_path = _precalibration_manifest_entry(
        manifest_snapshot.payload, manifest_path=manifest_path
    )
    return _ManifestTrainingReportReference(
        manifest_payload=manifest_snapshot.payload,
        manifest_entry=entry,
        training_report_path=report_path,
        manifest_identity=manifest_snapshot.identity,
    )


def _read_manifest_training_report(
    reference: _ManifestTrainingReportReference,
) -> TrainingReportBinding:
    training_report = reference.manifest_entry["training_report"]
    if not isinstance(training_report, Mapping):
        raise ValueError("pre-calibration manifest training report binding is invalid")
    report_payload = _input_regular_bytes(
        reference.training_report_path,
        description="training report",
        maximum=_MAX_TRAINING_REPORT_BYTES,
    )
    actual_sha256 = hashlib.sha256(report_payload).hexdigest()
    if (
        len(report_payload) != training_report["size_bytes"]
        or actual_sha256 != training_report["sha256"]
    ):
        raise ValueError("manifest training report identity differs from report bytes")
    assert_protected_hashes_absent(
        (reference.manifest_entry["sha256"], actual_sha256)
    )
    return TrainingReportBinding(
        manifest_payload=reference.manifest_payload,
        manifest_entry=reference.manifest_entry,
        training_report_path=reference.training_report_path,
        training_report_payload=report_payload,
        manifest_identity=reference.manifest_identity,
    )


def validate_calibration_preflight(config: CalibrationConfig) -> TrainingReportBinding:
    """Reject unsafe outputs before opening any calibration input artifact."""
    _validate_config(config)
    validate_calibration_paths(
        config.dataset_root,
        config.archive,
        config.cache,
        config.model_dir,
    )
    _validate_protected_paths(
        (
            ("policy output", config.policy_output),
            ("report output", config.report_output),
            ("rejection report", config.rejection_report),
            ("manifest output", config.manifest),
        )
    )
    publication_paths = _resolve_publication_paths(
        config.policy_output,
        config.report_output,
        config.manifest,
    )
    _resolve_rejection_report_path(config, publication_paths)
    concrete_inputs = (
        ("dataset root", config.dataset_root),
        ("archive", config.archive),
        ("cache", config.cache),
        ("refiner ONNX", config.model_dir / MODEL_FILENAME),
    )
    for output_description, output in (
        ("policy output", config.policy_output),
        ("report output", config.report_output),
        ("manifest output", config.manifest),
    ):
        for input_description, input_path in concrete_inputs:
            if _paths_collide(output, input_path):
                raise ValueError(
                    f"{output_description} collides with calibration {input_description}"
                )
    reference = _manifest_training_report_reference(config)
    _resolve_rejection_report_path(
        config,
        publication_paths,
        training_report_path=reference.training_report_path,
    )
    for output_description, output in (
        ("policy output", config.policy_output),
        ("report output", config.report_output),
        ("manifest output", config.manifest),
    ):
        if _paths_collide(output, reference.training_report_path):
            raise ValueError(
                f"{output_description} collides with calibration training report"
            )
    return _read_manifest_training_report(reference)


def validate_validation_records(records: Sequence[object]) -> tuple[object, ...]:
    """Accept only a non-empty SmartDoc background05 validation sequence."""
    selected = tuple(records)
    if not selected:
        raise ValueError("SmartDoc background05 validation records are required")
    if any(
        getattr(record, "background", None) != SMARTDOC_VALIDATION_BACKGROUND
        for record in selected
    ):
        raise ValueError(
            "calibration accepts only SmartDoc background05 validation records; "
            "training backgrounds01-04 are forbidden"
        )
    return selected


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_object(payload: bytes, *, description: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be finite valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def bind_model_and_training_report(
    model_path: Path,
    model_payload: bytes,
    training_report_path: Path,
    training_report_payload: bytes,
) -> dict[str, dict[str, object]]:
    """Bind an exact refiner ONNX payload to its exact training-report bytes."""
    if not isinstance(model_payload, bytes) or not model_payload:
        raise ValueError("ONNX model payload must be non-empty bytes")
    if not isinstance(training_report_payload, bytes) or not training_report_payload:
        raise ValueError("training report payload must be non-empty bytes")
    document = _strict_json_object(
        training_report_payload, description="training report"
    )
    if not isinstance(document.get("onnx"), dict):
        raise ValueError("training report must contain an ONNX identity")
    onnx = document["onnx"]
    required = {"filename", "size_bytes", "sha256"}
    if not required <= set(onnx):
        raise ValueError("training report ONNX identity is incomplete")
    filename = onnx["filename"]
    expected_size = onnx["size_bytes"]
    expected_sha = onnx["sha256"]
    if (
        not isinstance(filename, str)
        or filename != Path(model_path).name
        or filename != MODEL_FILENAME
        or type(expected_size) is not int
        or expected_size < 1
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError("training report ONNX identity is invalid")
    actual_sha = hashlib.sha256(model_payload).hexdigest()
    report_sha = hashlib.sha256(training_report_payload).hexdigest()
    assert_protected_hashes_absent((expected_sha, actual_sha, report_sha))
    if expected_size != len(model_payload) or expected_sha != actual_sha:
        raise ValueError("training report ONNX identity differs from model bytes")
    return {
        "model": {
            "adapter": REFINER_MODEL_ADAPTER_NAME,
            "filename": filename,
            "size_bytes": len(model_payload),
            "sha256": actual_sha,
        },
        "training_report": {
            "filename": Path(training_report_path).name,
            "size_bytes": len(training_report_payload),
            "sha256": report_sha,
        },
    }


def _validate_manifest_model_binding(
    binding: TrainingReportBinding,
    model_binding: Mapping[str, object],
) -> None:
    model = model_binding.get("model")
    training_report = model_binding.get("training_report")
    if not isinstance(model, Mapping) or not isinstance(training_report, Mapping):
        raise ValueError("bound model and training report identities are invalid")
    entry = binding.manifest_entry
    if (
        entry.get("local_filename") != model.get("filename")
        or entry.get("checkpoint_size_bytes") != model.get("size_bytes")
        or entry.get("sha256") != model.get("sha256")
    ):
        raise ValueError("manifest refiner model identity differs from bound bytes")
    expected_report = entry["training_report"]
    if not isinstance(expected_report, Mapping) or (
        expected_report.get("filename") != training_report.get("filename")
        or expected_report.get("size_bytes") != training_report.get("size_bytes")
        or expected_report.get("sha256") != training_report.get("sha256")
    ):
        raise ValueError("manifest training report identity differs from bound bytes")


def cache_onnx_outputs(
    dataset: object,
    session: object,
) -> tuple[CachedRefinerOutput, ...]:
    """Run exactly one static batch-four ONNX inference for every dataset item."""
    if not callable(getattr(dataset, "load_example", None)):
        raise TypeError("validation dataset must provide load_example()")
    if not callable(getattr(session, "run", None)):
        raise TypeError("ONNX session must provide run()")
    input_name = getattr(session, "input_name", "patches")
    if not isinstance(input_name, str) or not input_name:
        raise ValueError("ONNX session input name is invalid")
    try:
        length = len(dataset)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("validation dataset must define a length") from error
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("validation dataset must be non-empty")
    cached: list[CachedRefinerOutput] = []
    for index in range(length):
        example = dataset.load_example(index)
        outputs = _run_onnx_example(example, session, input_name=input_name)
        cached.append(CachedRefinerOutput(example, *outputs))
    return tuple(cached)


def _run_onnx_example(
    example: object,
    session: object,
    *,
    input_name: str,
) -> tuple[object, object, object]:
    if not isinstance(example, RefinerExample):
        raise TypeError("validation dataset must yield RefinerExample values")
    patches = example.patches
    if (
        not isinstance(patches, torch.Tensor)
        or tuple(patches.shape) != (4, 3, 256, 256)
        or patches.dtype != torch.float32
        or not torch.isfinite(patches).all()
    ):
        raise ValueError("validation patches must be finite static 4x3x256x256 float32")
    array = patches.detach().to(device="cpu", dtype=torch.float32).numpy()
    outputs = session.run(
        ["corner_logits", "edge_logits", "confidence"],
        {input_name: array},
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("ONNX session returned an invalid output tuple")
    return outputs[0], outputs[1], outputs[2]


def enumerate_policy_grid() -> tuple[RefinerPolicy, ...]:
    """Return the complete frozen policy grid in canonical tuple order."""
    return tuple(
        RefinerPolicy(
            schema_version=POLICY_SCHEMA_VERSION,
            radius_ratio=radius_ratio,
            minimum_confidence=minimum_confidence,
            maximum_residual_ratio=maximum_residual_ratio,
            fallback="docaligner",
        )
        for radius_ratio, minimum_confidence, maximum_residual_ratio in product(
            RADIUS_RATIOS,
            MINIMUM_CONFIDENCES,
            MAXIMUM_RESIDUAL_RATIOS,
        )
    )


def policy_document(policy: RefinerPolicy) -> dict[str, object]:
    if not isinstance(policy, RefinerPolicy):
        raise TypeError("policy must be a RefinerPolicy")
    return {
        "schema_version": policy.schema_version,
        "radius_ratio": policy.radius_ratio,
        "minimum_confidence": policy.minimum_confidence,
        "maximum_residual_ratio": policy.maximum_residual_ratio,
        "fallback": policy.fallback,
    }


def policy_key(policy: RefinerPolicy) -> str:
    return json.dumps(
        policy_document(policy),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _aggregate_values(result: Mapping[str, object]) -> tuple[float, float, float, float]:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("calibration result must contain metrics")
    aggregate = metrics.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("calibration result must contain aggregate metrics")
    try:
        ssim = float(aggregate["warp_ssim"])
        rmse = float(aggregate["normalized_corner_rmse"])
        fallback_rate = float(aggregate["fallback_rate"])
        coverage = float(aggregate["coverage"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("calibration aggregate metrics are malformed") from error
    if not all(math.isfinite(value) for value in (ssim, rmse, fallback_rate, coverage)):
        raise ValueError("calibration aggregate metrics must be finite")
    if not (-1.0 <= ssim <= 1.0 and rmse >= 0.0):
        raise ValueError("calibration aggregate quality metrics are out of range")
    if not (0.0 <= fallback_rate <= 1.0 and 0.0 <= coverage <= 1.0):
        raise ValueError("calibration aggregate rates are out of range")
    return ssim, rmse, fallback_rate, coverage


def _policy_tuple(policy: RefinerPolicy) -> tuple[float, float, float]:
    return (
        policy.radius_ratio,
        policy.minimum_confidence,
        policy.maximum_residual_ratio,
    )


def select_best_result(
    results: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Rank finite, full-coverage candidates by the frozen published key."""
    candidates: list[Mapping[str, object]] = []
    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError("calibration results must be mappings")
        policy = result.get("policy")
        if not isinstance(policy, RefinerPolicy):
            raise TypeError("calibration result policy must be a RefinerPolicy")
        _, _, _, coverage = _aggregate_values(result)
        if coverage == 1.0:
            candidates.append(result)
    if not candidates:
        raise RuntimeError("calibration grid produced no finite full-coverage policy")
    return min(
        candidates,
        key=lambda result: (
            -_aggregate_values(result)[0],
            _aggregate_values(result)[1],
            _aggregate_values(result)[2],
            _policy_tuple(result["policy"]),  # type: ignore[arg-type]
        ),
    )


def smartdoc_gate(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
) -> bool:
    """Require a strictly better aggregate SSIM without an RMSE regression."""
    candidate_ssim, candidate_rmse, _, candidate_coverage = _aggregate_values(candidate)
    baseline_ssim, baseline_rmse, _, baseline_coverage = _aggregate_values(baseline)
    return bool(
        candidate_coverage == 1.0
        and baseline_coverage == 1.0
        and candidate_ssim > baseline_ssim
        and candidate_rmse <= baseline_rmse
    )


def _slice_document(value: object) -> dict[str, object]:
    complete = getattr(value, "complete", False)
    if not complete:
        raise RuntimeError("calibration did not produce finite full coverage")
    example_count = getattr(value, "example_count")
    fallback_count = getattr(value, "fallback_count")
    refined_count = getattr(value, "refined_count")
    if example_count < 1:
        raise RuntimeError("calibration slice is empty")
    return {
        "normalized_corner_rmse": float(getattr(value, "normalized_corner_rmse")),
        "warp_ssim": float(getattr(value, "warp_ssim")),
        "example_count": int(example_count),
        "refined_count": int(refined_count),
        "fallback_count": int(fallback_count),
        "coverage": 1.0,
        "fallback_rate": float(fallback_count / example_count),
    }


def _metric_result(
    views: Sequence[CachedRefinerOutput],
    *,
    policy: RefinerPolicy | None,
    radius_ratio: float,
) -> dict[str, object]:
    selected = tuple(
        item
        for item in views
        if math.isclose(
            item.example.radius_ratio,
            radius_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if not selected:
        raise ValueError("calibration has no views for the requested radius_ratio")
    accumulators = {
        "clean": _MetricAccumulator(),
        "augmented": _MetricAccumulator(),
        "aggregate": _MetricAccumulator(),
        f"radius:{radius_ratio:.2f}": _MetricAccumulator(),
    }
    for item in selected:
        if not isinstance(item, CachedRefinerOutput):
            raise TypeError("calibration views must be CachedRefinerOutput values")
        example = item.example
        if example.view not in {"clean", "augmented"}:
            raise ValueError("calibration view must be clean or augmented")
        if policy is None:
            prediction = np.array(example.coarse_corners, copy=True)
            fallback_used = True
        else:
            selected_corners = apply_refiner_outputs(
                example.source_bgr.shape,
                example.coarse_corners,
                example.transforms,
                item.corner_logits,
                item.edge_logits,
                item.model_confidence,
                policy,
            )
            prediction = selected_corners.corners
            fallback_used = selected_corners.fallback_used
        for key in ("aggregate", example.view, f"radius:{radius_ratio:.2f}"):
            accumulators[key].add(
                prediction,
                example,
                used_fallback=fallback_used,
            )
    metrics = {
        "clean": _slice_document(accumulators["clean"].finish()),
        "augmented": _slice_document(accumulators["augmented"].finish()),
        "aggregate": _slice_document(accumulators["aggregate"].finish()),
        "radii": {
            f"{radius_ratio:.2f}": _slice_document(
                accumulators[f"radius:{radius_ratio:.2f}"].finish()
            )
        },
    }
    aggregate = metrics["aggregate"]
    clean = metrics["clean"]
    augmented = metrics["augmented"]
    if (
        aggregate["example_count"]
        != clean["example_count"] + augmented["example_count"]
    ):
        raise RuntimeError("calibration aggregate views are incomplete")
    return metrics


def _radius_diagnostics(
    views: Sequence[CachedRefinerOutput],
    policy: RefinerPolicy | None,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    for radius_ratio in RADIUS_RATIOS:
        diagnostic_policy = (
            None
            if policy is None
            else RefinerPolicy(
                schema_version=policy.schema_version,
                radius_ratio=radius_ratio,
                minimum_confidence=policy.minimum_confidence,
                maximum_residual_ratio=policy.maximum_residual_ratio,
                fallback=policy.fallback,
            )
        )
        diagnostic = _metric_result(
            views,
            policy=diagnostic_policy,
            radius_ratio=radius_ratio,
        )
        diagnostics[f"{radius_ratio:.2f}"] = diagnostic["radii"][
            f"{radius_ratio:.2f}"
        ]
    return diagnostics


def evaluate_policy(
    policy: RefinerPolicy,
    views: Sequence[CachedRefinerOutput],
) -> dict[str, object]:
    """Score one frozen policy using cached model outputs only once."""
    if not isinstance(policy, RefinerPolicy):
        raise TypeError("policy must be a RefinerPolicy")
    metrics = _metric_result(views, policy=policy, radius_ratio=policy.radius_ratio)
    metrics["radii"] = _radius_diagnostics(views, policy)
    return {"policy": policy, "metrics": metrics}


def evaluate_coarse_baseline(
    views: Sequence[CachedRefinerOutput],
    *,
    radius_ratio: float,
) -> dict[str, object]:
    """Score an unchanged coarse DocAligner quadrilateral with full coverage."""
    if not isinstance(radius_ratio, (int, float)) or isinstance(radius_ratio, bool):
        raise TypeError("radius_ratio must be a real number")
    if not math.isfinite(float(radius_ratio)) or not 0.0 < float(radius_ratio) <= 1.0:
        raise ValueError("radius_ratio must be finite and in (0, 1]")
    metrics = _metric_result(
        views,
        policy=None,
        radius_ratio=float(radius_ratio),
    )
    metrics["radii"] = _radius_diagnostics(views, None)
    return {"policy": None, "metrics": metrics}


def _new_metric_accumulators() -> dict[str, _MetricAccumulator]:
    return {
        "clean": _MetricAccumulator(),
        "augmented": _MetricAccumulator(),
        "aggregate": _MetricAccumulator(),
    }


def _add_stream_metric(
    accumulators: Mapping[str, _MetricAccumulator],
    prediction: np.ndarray,
    example: RefinerExample,
    *,
    fallback_used: bool,
) -> None:
    for key in ("aggregate", example.view):
        accumulators[key].add(
            prediction,
            example,
            used_fallback=fallback_used,
        )


def _finish_stream_metrics(
    accumulators: Mapping[str, _MetricAccumulator],
) -> dict[str, object]:
    clean = _slice_document(accumulators["clean"].finish())
    augmented = _slice_document(accumulators["augmented"].finish())
    aggregate = _slice_document(accumulators["aggregate"].finish())
    if (
        aggregate["example_count"]
        != clean["example_count"] + augmented["example_count"]
        or aggregate["refined_count"]
        != clean["refined_count"] + augmented["refined_count"]
        or aggregate["fallback_count"]
        != clean["fallback_count"] + augmented["fallback_count"]
    ):
        raise RuntimeError("streamed calibration aggregate views are incomplete")
    return {"clean": clean, "augmented": augmented, "aggregate": aggregate}


def stream_calibration_results(
    dataset: object,
    session: object,
) -> tuple[tuple[dict[str, object], ...], dict[str, dict[str, object]]]:
    """Evaluate the frozen grid without retaining any full validation examples.

    Each view/radius is materialized once, invokes ONNX once, and immediately
    contributes to compact scalar metric accumulators for every relevant policy
    and diagnostic radius before its source image, patches, and outputs are
    released.
    """
    if not callable(getattr(dataset, "load_example", None)):
        raise TypeError("validation dataset must provide load_example()")
    if not callable(getattr(session, "run", None)):
        raise TypeError("ONNX session must provide run()")
    input_name = getattr(session, "input_name", "patches")
    if not isinstance(input_name, str) or not input_name:
        raise ValueError("ONNX session input name is invalid")
    try:
        length = len(dataset)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("validation dataset must define a length") from error
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("validation dataset must be non-empty")
    policies = enumerate_policy_grid()
    threshold_pairs = tuple(
        (confidence, residual)
        for confidence in MINIMUM_CONFIDENCES
        for residual in MAXIMUM_RESIDUAL_RATIOS
    )
    policy_by_key = {
        _policy_tuple(policy): policy for policy in policies
    }
    top = {policy: _new_metric_accumulators() for policy in policies}
    diagnostics = {
        policy: {radius: _new_metric_accumulators() for radius in RADIUS_RATIOS}
        for policy in policies
    }
    baselines = {radius: _new_metric_accumulators() for radius in RADIUS_RATIOS}
    for index in range(length):
        example = dataset.load_example(index)
        if not isinstance(example, RefinerExample):
            raise TypeError("validation dataset must yield RefinerExample values")
        if not any(
            math.isclose(
                example.radius_ratio,
                radius,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for radius in RADIUS_RATIOS
        ):
            raise ValueError("validation example radius is outside the frozen grid")
        radius = next(
            candidate
            for candidate in RADIUS_RATIOS
            if math.isclose(
                example.radius_ratio,
                candidate,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        corner_logits, edge_logits, model_confidence = _run_onnx_example(
            example,
            session,
            input_name=input_name,
        )
        _add_stream_metric(
            baselines[radius],
            np.array(example.coarse_corners, copy=True),
            example,
            fallback_used=True,
        )
        for minimum_confidence, maximum_residual_ratio in threshold_pairs:
            runtime_policy = policy_by_key[
                (radius, minimum_confidence, maximum_residual_ratio)
            ]
            selection = apply_refiner_outputs(
                example.source_bgr.shape,
                example.coarse_corners,
                example.transforms,
                corner_logits,
                edge_logits,
                model_confidence,
                runtime_policy,
            )
            for policy in policies:
                if (
                    policy.minimum_confidence == minimum_confidence
                    and policy.maximum_residual_ratio == maximum_residual_ratio
                ):
                    _add_stream_metric(
                        diagnostics[policy][radius],
                        selection.corners,
                        example,
                        fallback_used=selection.fallback_used,
                    )
                    if policy.radius_ratio == radius:
                        _add_stream_metric(
                            top[policy],
                            selection.corners,
                            example,
                            fallback_used=selection.fallback_used,
                        )
        del corner_logits, edge_logits, model_confidence, example
    grid: list[dict[str, object]] = []
    for policy in policies:
        metrics = _finish_stream_metrics(top[policy])
        metrics["radii"] = {
            f"{radius:.2f}": _finish_stream_metrics(diagnostics[policy][radius])["aggregate"]
            for radius in RADIUS_RATIOS
        }
        grid.append({"policy": policy, "metrics": metrics})
    baseline_results: dict[str, dict[str, object]] = {}
    baseline_radii = {
        f"{radius:.2f}": _finish_stream_metrics(baselines[radius])["aggregate"]
        for radius in RADIUS_RATIOS
    }
    for radius in RADIUS_RATIOS:
        metrics = _finish_stream_metrics(baselines[radius])
        metrics["radii"] = dict(baseline_radii)
        baseline_results[f"{radius:.2f}"] = {"policy": None, "metrics": metrics}
    return tuple(grid), baseline_results


def _serialized_result(result: Mapping[str, object]) -> dict[str, object]:
    policy = result.get("policy")
    if not isinstance(policy, RefinerPolicy):
        raise TypeError("calibration result policy must be a RefinerPolicy")
    _aggregate_values(result)
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("calibration result metrics are invalid")
    serialized = {
        "policy": policy_document(policy),
        "policy_key": policy_key(policy),
        "metrics": json.loads(json.dumps(metrics, allow_nan=False)),
    }
    _finite_json(serialized)
    return serialized


def build_calibration_report(
    *,
    selected: Mapping[str, object],
    grid: Sequence[Mapping[str, object]],
    baselines: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Create the finite, complete calibration record before publication."""
    expected_grid = enumerate_policy_grid()
    supplied = tuple(grid)
    if len(supplied) != len(expected_grid):
        raise ValueError("calibration report must contain all 27 grid policies")
    policies = tuple(item.get("policy") for item in supplied)
    if policies != expected_grid:
        raise ValueError("calibration report grid is incomplete or non-canonical")
    selected_policy = selected.get("policy")
    if not isinstance(selected_policy, RefinerPolicy) or selected not in supplied:
        raise ValueError("selected calibration result must be a grid entry")
    expected_baselines = {f"{value:.2f}" for value in RADIUS_RATIOS}
    if set(baselines) != expected_baselines:
        raise ValueError("calibration report must contain every radius baseline")
    baseline = baselines[f"{selected_policy.radius_ratio:.2f}"]
    gate_passed = smartdoc_gate(selected, baseline)
    serialized_baselines: dict[str, object] = {}
    for radius, result in sorted(baselines.items()):
        _aggregate_values(result)
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("coarse baseline metrics are invalid")
        serialized_baselines[radius] = json.loads(
            json.dumps(metrics, allow_nan=False)
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "adapter": ADAPTER_NAME,
        "model": provenance.get("model"),
        "training_report": provenance.get("training_report"),
        "cache": provenance.get("cache"),
        "smartdoc": provenance.get("smartdoc"),
        "docaligner": provenance.get("docaligner"),
        "seed": provenance.get("seed"),
        "source_commit": provenance.get("source_commit"),
        "reproducible_command": provenance.get("reproducible_command"),
        "declared_grid": {
            "radius_ratios": list(RADIUS_RATIOS),
            "minimum_confidences": list(MINIMUM_CONFIDENCES),
            "maximum_residual_ratios": list(MAXIMUM_RESIDUAL_RATIOS),
            "fallback": "docaligner",
            "policy_count": len(expected_grid),
        },
        "coarse_baselines": serialized_baselines,
        "grid": [_serialized_result(item) for item in supplied],
        "selected_policy": policy_document(selected_policy),
        "selected_metrics": json.loads(
            json.dumps(selected["metrics"], allow_nan=False)
        ),
        "smartdoc_gate": {
            "passed": gate_passed,
            "candidate_aggregate_warp_ssim": _aggregate_values(selected)[0],
            "baseline_aggregate_warp_ssim": _aggregate_values(baseline)[0],
            "candidate_aggregate_normalized_corner_rmse": _aggregate_values(
                selected
            )[1],
            "baseline_aggregate_normalized_corner_rmse": _aggregate_values(
                baseline
            )[1],
            "rule": (
                "candidate aggregate grayscale warp SSIM must be strictly greater "
                "than unchanged-coarse baseline and candidate aggregate normalized "
                "corner RMSE must be less than or equal to baseline"
            ),
        },
    }
    _finite_json(report)
    return report


def _finite_json(value: object) -> None:
    json.dumps(value, allow_nan=False)


_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "adapter",
        "model",
        "training_report",
        "cache",
        "smartdoc",
        "docaligner",
        "seed",
        "source_commit",
        "reproducible_command",
        "declared_grid",
        "coarse_baselines",
        "grid",
        "selected_policy",
        "selected_metrics",
        "smartdoc_gate",
    }
)
_METRIC_FIELDS = frozenset({"clean", "augmented", "aggregate", "radii"})
_SLICE_FIELDS = frozenset(
    {
        "normalized_corner_rmse",
        "warp_ssim",
        "example_count",
        "refined_count",
        "fallback_count",
        "coverage",
        "fallback_rate",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "radius_ratio",
        "minimum_confidence",
        "maximum_residual_ratio",
        "fallback",
    }
)
_GATE_RULE = (
    "candidate aggregate grayscale warp SSIM must be strictly greater than "
    "unchanged-coarse baseline and candidate aggregate normalized corner RMSE "
    "must be less than or equal to baseline"
)


def _exact_mapping(
    value: object,
    fields: frozenset[str],
    description: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{description} schema is invalid")
    return value


def _sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _positive_size(value: object, description: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _finite_metric(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{description} must be a finite number")
    return converted


def _policy_from_document(value: object, description: str) -> RefinerPolicy:
    document = _exact_mapping(value, _POLICY_FIELDS, description)
    try:
        policy = RefinerPolicy(**dict(document))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} is invalid") from error
    if policy_document(policy) != dict(document):
        raise ValueError(f"{description} is not canonical")
    return policy


def _validate_metric_slice(value: object, description: str) -> Mapping[str, object]:
    document = _exact_mapping(value, _SLICE_FIELDS, description)
    rmse = _finite_metric(document["normalized_corner_rmse"], f"{description} RMSE")
    ssim = _finite_metric(document["warp_ssim"], f"{description} SSIM")
    coverage = _finite_metric(document["coverage"], f"{description} coverage")
    fallback_rate = _finite_metric(
        document["fallback_rate"], f"{description} fallback rate"
    )
    counts = tuple(
        _positive_size(document[name], f"{description} {name}")
        if name == "example_count"
        else document[name]
        for name in ("example_count", "refined_count", "fallback_count")
    )
    example_count, refined_count, fallback_count = counts
    if (
        type(refined_count) is not int
        or type(fallback_count) is not int
        or refined_count < 0
        or fallback_count < 0
        or refined_count + fallback_count != example_count
        or rmse < 0.0
        or not -1.0 <= ssim <= 1.0
        or coverage != 1.0
        or not 0.0 <= fallback_rate <= 1.0
        or not math.isclose(
            fallback_rate,
            fallback_count / example_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{description} values are inconsistent")
    return document


def _validate_metrics(
    value: object,
    *,
    description: str,
    own_radius: float,
) -> Mapping[str, object]:
    metrics = _exact_mapping(value, _METRIC_FIELDS, description)
    clean = _validate_metric_slice(metrics["clean"], f"{description} clean")
    augmented = _validate_metric_slice(
        metrics["augmented"], f"{description} augmented"
    )
    aggregate = _validate_metric_slice(
        metrics["aggregate"], f"{description} aggregate"
    )
    radii = metrics["radii"]
    expected_radii = {f"{radius:.2f}" for radius in RADIUS_RATIOS}
    if not isinstance(radii, Mapping) or set(radii) != expected_radii:
        raise ValueError(f"{description} must contain all radius diagnostics")
    for radius in sorted(expected_radii):
        _validate_metric_slice(radii[radius], f"{description} radius {radius}")
    if (
        aggregate["example_count"]
        != clean["example_count"] + augmented["example_count"]
        or aggregate["refined_count"]
        != clean["refined_count"] + augmented["refined_count"]
        or aggregate["fallback_count"]
        != clean["fallback_count"] + augmented["fallback_count"]
    ):
        raise ValueError(f"{description} aggregate counts are inconsistent")
    aggregate_count = aggregate["example_count"]
    expected_ssim = (
        clean["warp_ssim"] * clean["example_count"]
        + augmented["warp_ssim"] * augmented["example_count"]
    ) / aggregate_count
    expected_rmse = math.sqrt(
        (
            clean["normalized_corner_rmse"] ** 2 * clean["example_count"]
            + augmented["normalized_corner_rmse"] ** 2 * augmented["example_count"]
        )
        / aggregate_count
    )
    if not (
        math.isclose(
            aggregate["warp_ssim"],
            expected_ssim,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            aggregate["normalized_corner_rmse"],
            expected_rmse,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{description} aggregate metrics are inconsistent")
    own_slice = radii[f"{own_radius:.2f}"]
    for field in _SLICE_FIELDS:
        if field in {"normalized_corner_rmse", "warp_ssim", "coverage", "fallback_rate"}:
            if not math.isclose(
                own_slice[field],
                aggregate[field],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{description} own-radius diagnostic differs from aggregate"
                )
        elif own_slice[field] != aggregate[field]:
            raise ValueError(
                f"{description} own-radius diagnostic differs from aggregate"
            )
    return metrics


def _validate_identity(
    value: object,
    *,
    description: str,
    fields: frozenset[str],
    expected_adapter: str | None = None,
    expected_filename: str | None = None,
) -> Mapping[str, object]:
    document = _exact_mapping(value, fields, description)
    if expected_adapter is not None and document.get("adapter") != expected_adapter:
        raise ValueError(f"{description} adapter is invalid")
    if expected_filename is not None and document.get("filename") != expected_filename:
        raise ValueError(f"{description} filename is invalid")
    if "filename" in document and (
        not isinstance(document["filename"], str) or not document["filename"]
    ):
        raise ValueError(f"{description} filename is invalid")
    _positive_size(document["size_bytes"], f"{description} size")
    _sha256(document["sha256"], f"{description} SHA-256")
    return document


def validate_calibration_report(
    report: object,
    policy: RefinerPolicy,
) -> None:
    """Require a complete, self-consistent calibration report before publish."""
    if not isinstance(policy, RefinerPolicy):
        raise TypeError("policy must be a RefinerPolicy")
    document = _exact_mapping(report, _REPORT_FIELDS, "calibration report")
    if document["schema_version"] != 1 or document["adapter"] != ADAPTER_NAME:
        raise ValueError("calibration report schema or adapter is invalid")
    model = _validate_identity(
        document["model"],
        description="calibration model",
        fields=frozenset({"adapter", "filename", "size_bytes", "sha256"}),
        expected_adapter=REFINER_MODEL_ADAPTER_NAME,
        expected_filename=MODEL_FILENAME,
    )
    training_report = _validate_identity(
        document["training_report"],
        description="training report",
        fields=frozenset({"filename", "size_bytes", "sha256"}),
        expected_filename=TRAINING_REPORT_FILENAME,
    )
    cache = _validate_identity(
        document["cache"],
        description="refiner cache",
        fields=frozenset({"filename", "path", "size_bytes", "sha256"}),
    )
    if not isinstance(cache["path"], str) or not cache["path"]:
        raise ValueError("refiner cache path is invalid")
    docaligner = _validate_identity(
        document["docaligner"],
        description="DocAligner",
        fields=frozenset({"adapter", "filename", "size_bytes", "sha256"}),
        expected_adapter="docaligner",
    )
    smartdoc = _exact_mapping(
        document["smartdoc"],
        frozenset(
            {
                "name",
                "version",
                "archive_sha256",
                "archive_path",
                "splits",
                "record_counts",
                "records",
            }
        ),
        "SmartDoc provenance",
    )
    if smartdoc["name"] != "SmartDoc 2015" or smartdoc["version"] != SMARTDOC_VERSION:
        raise ValueError("SmartDoc provenance name or version is invalid")
    _sha256(smartdoc["archive_sha256"], "SmartDoc archive SHA-256")
    if not isinstance(smartdoc["archive_path"], str) or not smartdoc["archive_path"]:
        raise ValueError("SmartDoc archive path is invalid")
    splits = _exact_mapping(
        smartdoc["splits"],
        frozenset({"validation", "training_rejected"}),
        "SmartDoc splits",
    )
    if (
        splits["validation"] != [SMARTDOC_VALIDATION_BACKGROUND]
        or splits["training_rejected"] != sorted(SMARTDOC_TRAIN_BACKGROUNDS)
    ):
        raise ValueError("SmartDoc validation-only split provenance is invalid")
    counts = _exact_mapping(
        smartdoc["record_counts"],
        frozenset({"all", "validation", "validation_refinement_views"}),
        "SmartDoc record counts",
    )
    for name, value in counts.items():
        _positive_size(value, f"SmartDoc record count {name}")
    records = smartdoc["records"]
    if (
        not isinstance(records, list)
        or len(records) != counts["validation"]
        or not all(
            isinstance(item, str) and item.startswith(f"{SMARTDOC_VALIDATION_BACKGROUND}/")
            for item in records
        )
        or len(set(records)) != len(records)
    ):
        raise ValueError("SmartDoc validation record provenance is invalid")
    if type(document["seed"]) is not int:
        raise ValueError("calibration seed is invalid")
    commit = document["source_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("calibration source commit is invalid")
    if not isinstance(document["reproducible_command"], str) or not document[
        "reproducible_command"
    ]:
        raise ValueError("calibration reproducible command is invalid")
    grid_document = _exact_mapping(
        document["declared_grid"],
        frozenset(
            {
                "radius_ratios",
                "minimum_confidences",
                "maximum_residual_ratios",
                "fallback",
                "policy_count",
            }
        ),
        "declared calibration grid",
    )
    if (
        grid_document["radius_ratios"] != list(RADIUS_RATIOS)
        or grid_document["minimum_confidences"] != list(MINIMUM_CONFIDENCES)
        or grid_document["maximum_residual_ratios"] != list(MAXIMUM_RESIDUAL_RATIOS)
        or grid_document["fallback"] != "docaligner"
        or grid_document["policy_count"] != 27
    ):
        raise ValueError("declared calibration grid is invalid")
    grid = document["grid"]
    expected_policies = enumerate_policy_grid()
    if not isinstance(grid, list) or len(grid) != len(expected_policies):
        raise ValueError("calibration report grid is incomplete")
    grid_metrics: dict[RefinerPolicy, Mapping[str, object]] = {}
    for index, (entry, expected_policy) in enumerate(zip(grid, expected_policies, strict=True)):
        item = _exact_mapping(
            entry,
            frozenset({"policy", "policy_key", "metrics"}),
            f"calibration grid entry {index}",
        )
        entry_policy = _policy_from_document(
            item["policy"], f"calibration grid policy {index}"
        )
        if entry_policy != expected_policy or item["policy_key"] != policy_key(entry_policy):
            raise ValueError("calibration report grid is not canonical")
        grid_metrics[entry_policy] = _validate_metrics(
            item["metrics"],
            description=f"calibration grid metrics {index}",
            own_radius=entry_policy.radius_ratio,
        )
    selected_policy = _policy_from_document(
        document["selected_policy"], "selected calibration policy"
    )
    if selected_policy != policy or selected_policy not in grid_metrics:
        raise ValueError("selected calibration policy differs from policy payload")
    optimal_policy = min(
        grid_metrics,
        key=lambda candidate_policy: (
            -grid_metrics[candidate_policy]["aggregate"]["warp_ssim"],
            grid_metrics[candidate_policy]["aggregate"]["normalized_corner_rmse"],
            grid_metrics[candidate_policy]["aggregate"]["fallback_rate"],
            _policy_tuple(candidate_policy),
        ),
    )
    if selected_policy != optimal_policy:
        raise ValueError("selected calibration policy is not the optimal grid result")
    selected_metrics = _validate_metrics(
        document["selected_metrics"],
        description="selected calibration metrics",
        own_radius=selected_policy.radius_ratio,
    )
    if dict(selected_metrics) != dict(grid_metrics[selected_policy]):
        raise ValueError("selected calibration metrics differ from selected grid entry")
    baselines = document["coarse_baselines"]
    expected_radii = {f"{radius:.2f}" for radius in RADIUS_RATIOS}
    if not isinstance(baselines, Mapping) or set(baselines) != expected_radii:
        raise ValueError("coarse baselines are incomplete")
    baseline_metrics = {
        radius: _validate_metrics(
            baselines[radius],
            description=f"coarse baseline {radius}",
            own_radius=float(radius),
        )
        for radius in sorted(expected_radii)
    }
    gate = _exact_mapping(
        document["smartdoc_gate"],
        frozenset(
            {
                "passed",
                "candidate_aggregate_warp_ssim",
                "baseline_aggregate_warp_ssim",
                "candidate_aggregate_normalized_corner_rmse",
                "baseline_aggregate_normalized_corner_rmse",
                "rule",
            }
        ),
        "SmartDoc gate",
    )
    baseline = baseline_metrics[f"{selected_policy.radius_ratio:.2f}"]
    candidate = selected_metrics["aggregate"]
    baseline_aggregate = baseline["aggregate"]
    expected_gate = bool(
        candidate["warp_ssim"] > baseline_aggregate["warp_ssim"]
        and candidate["normalized_corner_rmse"]
        <= baseline_aggregate["normalized_corner_rmse"]
    )
    if (
        gate["passed"] is not True
        or not expected_gate
        or gate["candidate_aggregate_warp_ssim"] != candidate["warp_ssim"]
        or gate["baseline_aggregate_warp_ssim"] != baseline_aggregate["warp_ssim"]
        or gate["candidate_aggregate_normalized_corner_rmse"]
        != candidate["normalized_corner_rmse"]
        or gate["baseline_aggregate_normalized_corner_rmse"]
        != baseline_aggregate["normalized_corner_rmse"]
        or gate["rule"] != _GATE_RULE
    ):
        raise ValueError("SmartDoc gate is absent, failed, or inconsistent")
    assert_protected_hashes_absent(
        (
            model["sha256"],
            training_report["sha256"],
            cache["sha256"],
            smartdoc["archive_sha256"],
            docaligner["sha256"],
        )
    )


def _manifest_record(
    policy_payload: bytes,
    report_payload: bytes,
    report: Mapping[str, object],
    *,
    training_manifest_entry: Mapping[str, object],
    policy_filename: str,
    report_filename: str,
) -> dict[str, object]:
    policy_sha256 = hashlib.sha256(policy_payload).hexdigest()
    entry = copy.deepcopy(dict(training_manifest_entry))
    entry["local_corner_refiner_policy"] = {
        "filename": policy_filename,
        "schema_version": POLICY_SCHEMA_VERSION,
        "sha256": policy_sha256,
        "size_bytes": len(policy_payload),
    }
    entry["calibration_report"] = {
        "filename": report_filename,
        "sha256": hashlib.sha256(report_payload).hexdigest(),
        "size_bytes": len(report_payload),
    }
    entry["calibration"] = {
        "model": report.get("model"),
        "training_report": report.get("training_report"),
        "cache": report.get("cache"),
        "smartdoc": report.get("smartdoc"),
        "docaligner": report.get("docaligner"),
        "source_commit": report.get("source_commit"),
        "reproducible_command": report.get("reproducible_command"),
        "selected_policy": report.get("selected_policy"),
        "smartdoc_gate": report.get("smartdoc_gate"),
    }
    return entry


def _bind_model_identity(
    manifest: dict[str, object],
    report: Mapping[str, object],
    *,
    expected_manifest_entry: Mapping[str, object] | None = None,
) -> dict[str, object]:
    identity = report.get("model")
    if identity is None:
        return manifest
    if not isinstance(identity, Mapping):
        raise ValueError("calibration model identity must be an object")
    required = {"adapter", "filename", "sha256", "size_bytes"}
    if set(identity) != required:
        raise ValueError("calibration model identity has an invalid schema")
    adapter = identity["adapter"]
    if not isinstance(adapter, str):
        raise ValueError("calibration model adapter is invalid")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("manifest must contain a models list")
    matches = [
        entry
        for entry in models
        if isinstance(entry, dict) and entry.get("adapter") == adapter
    ]
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one calibrated model")
    entry = matches[0]
    if expected_manifest_entry is not None and entry != dict(expected_manifest_entry):
        raise ValueError("pre-calibration manifest entry changed before publication")
    if (
        entry.get("local_filename") != identity["filename"]
        or entry.get("sha256") != identity["sha256"]
        or entry.get("checkpoint_size_bytes", entry.get("size_bytes"))
        != identity["size_bytes"]
    ):
        raise ValueError("manifest model identity differs from calibrated bytes")
    return manifest


def _validate_policy_payload(payload: bytes, expected: bytes) -> RefinerPolicy:
    if payload != expected:
        raise RuntimeError("staged policy bytes changed")
    try:
        document = _strict_json_object(payload, description="policy document")
        if set(document) != {
            "schema_version",
            "radius_ratio",
            "minimum_confidence",
            "maximum_residual_ratio",
            "fallback",
        }:
            raise ValueError("policy document schema is invalid")
        policy = RefinerPolicy(**document)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("staged policy JSON failed validation") from error
    if policy_document(policy) != document:
        raise RuntimeError("staged policy JSON is not canonical")
    return policy


def _validate_committed_calibration(
    payloads: Mapping[Path, bytes],
    *,
    paths: OutputPaths,
    policy_payload: bytes,
    report_payload: bytes,
    manifest_record: Mapping[str, object],
    training_manifest_entry: Mapping[str, object],
) -> None:
    if payloads.get(paths.checkpoint) != policy_payload:
        raise RuntimeError("published policy bytes changed")
    if payloads.get(paths.report) != report_payload:
        raise RuntimeError("published calibration report bytes changed")
    published_policy = _validate_policy_payload(policy_payload, policy_payload)
    try:
        report = _strict_json_object(
            report_payload, description="calibration report"
        )
        manifest = _strict_json_object(
            payloads[paths.manifest], description="calibration manifest"
        )
    except ValueError as error:
        raise RuntimeError("published calibration JSON failed validation") from error
    try:
        validate_calibration_report(report, published_policy)
    except (TypeError, ValueError) as error:
        raise RuntimeError("published calibration report failed validation") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 3
        or not isinstance(manifest.get("models"), list)
        or not all(isinstance(item, dict) for item in manifest["models"])
    ):
        raise RuntimeError("published calibration manifest schema is invalid")
    matches = [
        item
        for item in manifest["models"]
        if item.get("adapter") == REFINER_MODEL_ADAPTER_NAME
    ]
    rebuilt_record = _manifest_record(
        policy_payload,
        report_payload,
        report,
        training_manifest_entry=training_manifest_entry,
        policy_filename=paths.checkpoint.name,
        report_filename=paths.report.name,
    )
    if (
        manifest_record != rebuilt_record
        or len(matches) != 1
        or matches[0] != rebuilt_record
    ):
        raise RuntimeError("published calibration manifest record failed cross-check")
    for key, value in training_manifest_entry.items():
        if (
            key not in _CALIBRATION_MANIFEST_FIELDS
            and matches[0].get(key) != value
        ):
            raise RuntimeError("published calibration manifest training provenance changed")
    try:
        _bind_model_identity(manifest, report)
    except ValueError as error:
        raise RuntimeError("published calibrated model identity failed cross-check") from error


def _resolve_publication_paths(
    policy_output: Path,
    report_output: Path,
    manifest_path: Path,
) -> OutputPaths:
    raw = tuple(
        Path(os.path.abspath(path))
        for path in (policy_output, report_output, manifest_path)
    )
    _validate_protected_paths(
        (
            ("policy output", raw[0]),
            ("report output", raw[1]),
            ("manifest output", raw[2]),
        )
    )
    for path in raw:
        if path.is_symlink():
            raise ValueError(f"output target must not be a symlink: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"output target must be a regular file: {path}")
        for parent in path.parents:
            if parent.is_symlink():
                raise ValueError(f"output parent must not be symlinked: {parent}")
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"output parent must be a directory: {parent}")
    resolved = tuple(path.resolve(strict=False) for path in raw)
    if len(set(resolved)) != len(resolved):
        raise ValueError("policy, report, and manifest output paths collide")
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError("policy, report, and manifest output paths collide")
    existing = [path for path in raw if path.exists()]
    for index, first in enumerate(existing):
        for second in existing[index + 1 :]:
            if first.samefile(second):
                raise ValueError("policy, report, and manifest output paths collide")
    policy_path, report_path, manifest = resolved
    if policy_path.name != POLICY_FILENAME:
        raise ValueError(f"policy output filename must be exactly {POLICY_FILENAME}")
    if report_path.name != REPORT_FILENAME:
        raise ValueError(f"report output filename must be exactly {REPORT_FILENAME}")
    if manifest.name != "manifest.json":
        raise ValueError("manifest filename must be exactly manifest.json")
    return OutputPaths(policy_path, report_path, manifest)


def _resolve_rejection_report_path(
    config: CalibrationConfig,
    publication_paths: OutputPaths,
    *,
    training_report_path: Path | None = None,
) -> Path:
    raw = Path(os.path.abspath(config.rejection_report))
    _validate_protected_paths((("rejection report", raw),))
    if raw.is_symlink():
        raise ValueError("rejection report target must not be a symlink")
    for parent in raw.parents:
        if parent.is_symlink():
            raise ValueError("rejection report parent must not be symlinked")
        if parent.exists() and not parent.is_dir():
            raise ValueError("rejection report parent must be a directory")
    if raw.exists():
        raise ValueError("rejection report must not already exist")

    concrete_paths = (
        ("policy output", publication_paths.checkpoint),
        ("accepted report output", publication_paths.report),
        ("manifest output", publication_paths.manifest),
        ("cache", config.cache),
        ("archive", config.archive),
        ("model directory", config.model_dir),
        ("SmartDoc root", config.dataset_root),
        ("refiner ONNX", config.model_dir / MODEL_FILENAME),
    )
    for description, candidate in concrete_paths:
        if _paths_collide(raw, candidate):
            raise ValueError(f"rejection report collides with {description}")
    if training_report_path is not None and _paths_collide(raw, training_report_path):
        raise ValueError("rejection report collides with calibration training report")
    return raw.resolve(strict=False)


def _write_rejection_report(path: Path, report: Mapping[str, object]) -> None:
    payload = _json_bytes(report)
    if len(payload) < 1 or len(payload) > _MAX_REJECTION_REPORT_BYTES:
        raise ValueError("rejection report size is invalid")
    target = Path(os.path.abspath(path))
    if target.is_symlink():
        raise ValueError("rejection report target must not be a symlink")
    if target.exists():
        raise ValueError("rejection report must not already exist")
    for parent in target.parents:
        if parent.is_symlink():
            raise ValueError("rejection report parent must not be symlinked")
        if parent.exists() and not parent.is_dir():
            raise ValueError("rejection report parent must be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        status = temporary.stat()
        if not stat.S_ISREG(status.st_mode) or status.st_size != len(payload):
            raise RuntimeError("staged rejection report changed before replacement")
        os.replace(temporary, target)
        directory_fd = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def publish_calibration(
    policy: RefinerPolicy,
    report: dict[str, object],
    *,
    policy_output: Path,
    report_output: Path,
    manifest_path: Path,
    phase_hook: Callable[[str, Path], None] | None = None,
    training_manifest_binding: TrainingReportBinding | None = None,
) -> None:
    """Atomically publish policy, report, and manifest with all-or-old rollback."""
    if not isinstance(policy, RefinerPolicy):
        raise TypeError("policy must be a RefinerPolicy")
    if not isinstance(report, dict):
        raise TypeError("report must be a dict")
    if training_manifest_binding is not None and not isinstance(
        training_manifest_binding, TrainingReportBinding
    ):
        raise TypeError("training_manifest_binding must be a TrainingReportBinding")
    _finite_json(report)
    paths = _resolve_publication_paths(policy_output, report_output, manifest_path)
    validate_calibration_report(report, policy)
    if training_manifest_binding is None:
        raise ValueError("training manifest binding is required for publication")
    policy_payload = _json_bytes(policy_document(policy))
    report_payload = _json_bytes(report)
    record = _manifest_record(
        policy_payload,
        report_payload,
        report,
        training_manifest_entry=training_manifest_binding.manifest_entry,
        policy_filename=paths.checkpoint.name,
        report_filename=paths.report.name,
    )

    def validate_published_bytes(payload: bytes) -> None:
        _validate_policy_payload(payload, policy_payload)

    def validate_committed(payloads: Mapping[Path, bytes]) -> None:
        _validate_committed_calibration(
            payloads,
            paths=paths,
            policy_payload=policy_payload,
            report_payload=report_payload,
            manifest_record=record,
            training_manifest_entry=training_manifest_binding.manifest_entry,
        )

    def validate_manifest_snapshot(snapshot: object) -> None:
        if snapshot is None:
            raise ValueError("pre-calibration manifest snapshot changed before publication")
        payload = getattr(snapshot, "payload", None)
        if not isinstance(payload, bytes):
            raise ValueError("pre-calibration manifest snapshot changed before publication")
        entry, _ = _precalibration_manifest_entry(
            payload, manifest_path=paths.manifest
        )
        if entry != training_manifest_binding.manifest_entry:
            raise ValueError("pre-calibration manifest entry changed before publication")
        if payload != training_manifest_binding.manifest_payload:
            raise ValueError("pre-calibration manifest snapshot changed before publication")
        expected_identity = training_manifest_binding.manifest_identity
        if expected_identity is None:
            return
        try:
            identity = _ManifestSnapshotIdentity(
                device=snapshot.device,
                inode=snapshot.inode,
                size=snapshot.size,
                mtime_ns=snapshot.mtime_ns,
                ctime_ns=snapshot.ctime_ns,
                nlink=snapshot.nlink,
            )
        except AttributeError as error:
            raise ValueError(
                "pre-calibration manifest snapshot changed before publication"
            ) from error
        if identity != expected_identity:
            raise ValueError("pre-calibration manifest snapshot changed before publication")

    publish_training_outputs(
        paths,
        checkpoint_bytes=policy_payload,
        report=report,
        manifest_record=record,
        validate_published_bytes=validate_published_bytes,
        validate_committed=validate_committed,
        manifest_transform=lambda document: _bind_model_identity(
            document,
            report,
        ),
        phase_hook=phase_hook,
        maximum_manifest_snapshot_size=(
            _MAX_MANIFEST_BYTES
        ),
        manifest_snapshot_validator=(
            validate_manifest_snapshot
        ),
    )


class SmartDocGateError(RuntimeError):
    """The selected policy did not beat its unchanged-coarse baseline."""

    def __init__(self, message: str, report: Mapping[str, object]) -> None:
        super().__init__(message)
        self.report = copy.deepcopy(dict(report))
        _finite_json(self.report)


def _validate_config(config: CalibrationConfig) -> None:
    if not isinstance(config, CalibrationConfig):
        raise TypeError("config must be a CalibrationConfig")
    for name in (
        "cache",
        "dataset_root",
        "archive",
        "model_dir",
        "policy_output",
        "report_output",
        "rejection_report",
        "manifest",
    ):
        if not isinstance(getattr(config, name), Path):
            raise TypeError(f"{name} must be a pathlib.Path")
    if config.archive_sha256 != SMARTDOC_ARCHIVE_SHA256:
        raise ValueError("archive_sha256 must equal the pinned SmartDoc archive hash")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(config.cpu_threads, bool)
        or not isinstance(config.cpu_threads, int)
        or config.cpu_threads < 1
    ):
        raise ValueError("cpu_threads must be at least 1")


def _input_regular_bytes(path: Path, *, description: str, maximum: int) -> bytes:
    if maximum < 1:
        raise ValueError("maximum input size must be positive")
    absolute = Path(os.path.abspath(path))
    for parent in absolute.parents:
        if parent.is_symlink():
            raise ValueError(f"{description} path must not have a symlinked ancestor")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"{description} must be a regular non-symlinked file") from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size < 1 or status.st_size > maximum:
            raise ValueError(f"{description} has an invalid size or file type")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(maximum + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not payload or len(payload) > maximum:
        raise ValueError(f"{description} has an invalid size")
    return payload


@dataclass(frozen=True, slots=True)
class _ManifestInputSnapshot:
    payload: bytes
    identity: _ManifestSnapshotIdentity


def _input_regular_manifest_snapshot(
    path: Path, *, description: str, maximum: int
) -> _ManifestInputSnapshot:
    if maximum < 1:
        raise ValueError("maximum input size must be positive")
    absolute = Path(os.path.abspath(path))
    for parent in absolute.parents:
        if parent.is_symlink():
            raise ValueError(f"{description} path must not have a symlinked ancestor")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"{description} must be a regular non-symlinked file") from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size < 1:
            raise ValueError(f"{description} has an invalid size or file type")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(maximum + 1)
            snapshot_status = os.fstat(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not payload or len(payload) > maximum:
        raise ValueError(f"{description} has an invalid size")
    return _ManifestInputSnapshot(
        payload=payload,
        identity=_ManifestSnapshotIdentity.from_status(snapshot_status),
    )


def _training_report_document(payload: bytes) -> dict[str, object]:
    return _strict_json_object(payload, description="training report")


def _verify_training_provenance(
    document: Mapping[str, object],
    *,
    cache_path: Path,
    cache_payload: bytes,
    cache: RefinerCache,
) -> None:
    cache_document = document.get("cache")
    docaligner_document = document.get("docaligner")
    if not isinstance(cache_document, Mapping) or not isinstance(
        docaligner_document, Mapping
    ):
        raise ValueError("training report cache or DocAligner provenance is missing")
    cache_sha = hashlib.sha256(cache_payload).hexdigest()
    if (
        cache_document.get("filename") != Path(cache_path).name
        or cache_document.get("size_bytes") != len(cache_payload)
        or cache_document.get("sha256") != cache_sha
    ):
        raise ValueError("training report cache identity differs from calibration cache")
    expected_docaligner = {
        "adapter": cache.model_identity.adapter,
        "filename": cache.model_identity.filename,
        "size_bytes": cache.model_identity.size_bytes,
        "sha256": cache.model_identity.sha256,
    }
    if dict(docaligner_document) != expected_docaligner:
        raise ValueError("training report DocAligner identity differs from cache")
    assert_protected_hashes_absent(
        (cache_sha, cache.model_identity.sha256)
    )


def _cpu_session(model_payload: bytes, cpu_threads: int) -> object:
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model_payload,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        raise RuntimeError("refiner ONNX failed CPU-only loading") from error
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("refiner ONNX must use CPUExecutionProvider only")
    return session


def _reproducible_command(config: CalibrationConfig) -> str:
    return shlex.join(
        (
            "uv",
            "run",
            "--group",
            "ml",
            "python",
            "-m",
            "openscaner.training.calibrate_corner_refiner",
            "--cache",
            str(config.cache),
            "--dataset-root",
            str(config.dataset_root),
            "--archive",
            str(config.archive),
            "--archive-sha256",
            config.archive_sha256,
            "--model-dir",
            str(config.model_dir),
            "--seed",
            str(config.seed),
            "--cpu-threads",
            str(config.cpu_threads),
            "--policy-output",
            str(config.policy_output),
            "--report-output",
            str(config.report_output),
            "--rejection-report",
            str(config.rejection_report),
            "--manifest",
            str(config.manifest),
        )
    )


def calibrate_policy(
    config: CalibrationConfig,
    *,
    _preflight: TrainingReportBinding | None = None,
) -> tuple[RefinerPolicy, dict[str, object]]:
    """Calibrate all frozen policies from verified background05 cache views.

    Each verified view is materialized and inferred exactly once; its output is
    applied immediately to the frozen grid before the full view is released.
    """
    _validate_config(config)
    if _preflight is None:
        binding = validate_calibration_preflight(config)
    elif isinstance(_preflight, TrainingReportBinding):
        binding = _preflight
    else:
        raise TypeError("_preflight must be a TrainingReportBinding")
    validate_smartdoc_source_paths(config.dataset_root, config.archive)
    assert_protected_hashes_absent((config.archive_sha256,))
    initial_commit = source_commit()

    model_path = config.model_dir / MODEL_FILENAME
    training_report_path = binding.training_report_path
    training_report_payload = binding.training_report_payload
    training_document = _training_report_document(training_report_payload)
    onnx_document = training_document.get("onnx")
    if not isinstance(onnx_document, Mapping) or not isinstance(
        onnx_document.get("sha256"), str
    ):
        raise ValueError("training report ONNX identity is missing")
    assert_protected_hashes_absent((onnx_document["sha256"],))
    model_payload = _input_regular_bytes(
        model_path,
        description="refiner ONNX model",
        maximum=32 * 1024 * 1024,
    )
    model_binding = bind_model_and_training_report(
        model_path,
        model_payload,
        training_report_path,
        training_report_payload,
    )
    _validate_manifest_model_binding(binding, model_binding)
    validate_refiner_onnx_bytes(model_payload)

    verify_smartdoc_markers(config.dataset_root)
    source = verify_smartdoc_source(config.archive, config.dataset_root)
    if source.archive_sha256 != config.archive_sha256:
        raise ValueError("verified SmartDoc archive hash differs from supplied hash")
    assert_protected_hashes_absent(
        (source.archive_sha256, *(item.sha256 for item in source.files.values()))
    )
    cache_payload = _read_cache_payload(config.cache)
    cache_metadata = _parse_refiner_cache(cache_payload)
    assert_protected_hashes_absent(
        (
            hashlib.sha256(cache_payload).hexdigest(),
            cache_metadata.model_identity.sha256,
        )
    )
    records = tuple(
        load_smartdoc_records(
            config.dataset_root,
            stride=cache_metadata.stride,
            verified_files=source.files,
            backgrounds=(SMARTDOC_VALIDATION_BACKGROUND,),
        )
    )
    validate_smartdoc_record_paths(records)
    assert_protected_hashes_absent(
        record.image_sha256
        for record in records
        if getattr(record, "image_sha256", None) is not None
    )
    cache = load_verified_refiner_cache(
        config.cache,
        records,
        smartdoc_archive_sha256=source.archive_sha256,
        smartdoc_version=SMARTDOC_VERSION,
        stride=cache_metadata.stride,
        seed=config.seed,
        model_identity=cache_metadata.model_identity,
        backgrounds=(SMARTDOC_VALIDATION_BACKGROUND,),
        payload=cache_payload,
    )
    _verify_training_provenance(
        training_document,
        cache_path=config.cache,
        cache_payload=cache_payload,
        cache=cache,
    )
    validation_records = validate_validation_records(
        tuple(
            record
            for record in records
            if getattr(record, "background", None) == SMARTDOC_VALIDATION_BACKGROUND
        )
    )
    validation_cache = RefinerCache(
        smartdoc_archive_sha256=cache.smartdoc_archive_sha256,
        smartdoc_version=cache.smartdoc_version,
        stride=cache.stride,
        seed=cache.seed,
        model_identity=cache.model_identity,
        records=tuple(
            item for item in cache.records if item.split == "validation"
        ),
    )
    dataset = LocalCornerRefinerDataset(
        validation_cache,
        validation_records,  # type: ignore[arg-type]
        "validation",
        config.seed,
    )
    if len(dataset) < 1:
        raise ValueError("background05 cache has no valid DocAligner refinements")
    session = _cpu_session(model_payload, config.cpu_threads)
    grid, baselines = stream_calibration_results(dataset, session)
    selected = select_best_result(grid)
    policy = selected["policy"]
    if not isinstance(policy, RefinerPolicy):
        raise RuntimeError("selected calibration policy is invalid")
    smartdoc = {
        "name": "SmartDoc 2015",
        "version": SMARTDOC_VERSION,
        "archive_sha256": source.archive_sha256,
        "archive_path": str(config.archive),
        "splits": {
            "validation": [SMARTDOC_VALIDATION_BACKGROUND],
            "training_rejected": sorted(SMARTDOC_TRAIN_BACKGROUNDS),
        },
        "record_counts": {
            "all": len(records),
            "validation": len(validation_records),
            "validation_refinement_views": len(dataset),
        },
        "records": [
            f"{record.background}/{record.sequence}/{record.frame_index}"
            for record in validation_records
        ],
    }
    provenance: dict[str, object] = {
        **model_binding,
        "cache": {
            "filename": config.cache.name,
            "path": str(config.cache),
            "size_bytes": len(cache_payload),
            "sha256": hashlib.sha256(cache_payload).hexdigest(),
        },
        "smartdoc": smartdoc,
        "docaligner": {
            "adapter": cache.model_identity.adapter,
            **cache.model_identity.as_document(),
        },
        "seed": config.seed,
        "source_commit": initial_commit,
        "reproducible_command": _reproducible_command(config),
    }
    report = build_calibration_report(
        selected=selected,
        grid=grid,
        baselines=baselines,
        provenance=provenance,
    )
    if not report["smartdoc_gate"]["passed"]:  # type: ignore[index]
        raise SmartDocGateError(
            "selected policy failed SmartDoc gate: aggregate SSIM must be strictly "
            "greater than unchanged-coarse baseline and aggregate RMSE must not rise",
            report,
        )
    if source_commit() != initial_commit:
        raise RuntimeError("source repository commit changed during calibration")
    return policy, report


def calibrate_corner_refiner(config: CalibrationConfig) -> dict[str, object]:
    """Calibrate, pass the SmartDoc gate, then atomically publish all outputs."""
    binding = validate_calibration_preflight(config)
    policy, report = calibrate_policy(config, _preflight=binding)
    publish_calibration(
        policy,
        report,
        policy_output=config.policy_output,
        report_output=config.report_output,
        manifest_path=config.manifest,
        training_manifest_binding=binding,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = calibrate_corner_refiner(config)
    except SmartDocGateError as error:
        _write_rejection_report(config.rejection_report, error.report)
        sys.stderr.write(_json_bytes(error.report).decode("utf-8"))
        sys.stderr.flush()
        return 1
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_NAME",
    "CalibrationConfig",
    "CachedRefinerOutput",
    "MODEL_FILENAME",
    "POLICY_FILENAME",
    "REPORT_FILENAME",
    "SmartDocGateError",
    "bind_model_and_training_report",
    "build_calibration_report",
    "cache_onnx_outputs",
    "calibrate_corner_refiner",
    "calibrate_policy",
    "enumerate_policy_grid",
    "evaluate_coarse_baseline",
    "evaluate_policy",
    "main",
    "parse_args",
    "policy_document",
    "policy_key",
    "publish_calibration",
    "select_best_result",
    "smartdoc_gate",
    "validate_calibration_paths",
    "validate_calibration_preflight",
    "validate_validation_records",
]
