"""Deterministic SmartDoc-only calibration for document-boundary fusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import torch

from openscaner.fusion.candidates import FusionCandidate, generate_candidates
from openscaner.fusion.policy import (
    FusionFeatures,
    FusionPolicy,
    _prepare_scoring_context,
    _score_candidate_with_context,
)
from openscaner.fusion.signals import (
    FusionSignalPredictor,
    FusionSignals,
    load_fusion_signal_predictor,
)
from openscaner.training.smartdoc import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SMARTDOC_ARCHIVE_SHA256,
    SMARTDOC_VALIDATION_BACKGROUND,
    SmartDocCornerDataset,
    SmartDocRecord,
    VerifiedSmartDocSource,
    assert_protected_hashes_absent,
    load_smartdoc_records,
    verify_smartdoc_source,
)
from openscaner.training.train_corners import (
    OutputPaths,
    _json_bytes,
    publish_training_outputs,
)


WEIGHT_NAMES = (
    "mask_agreement",
    "corner_response",
    "edge_support",
    "coarse_agreement",
)
WEIGHT_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
MAXIMUM_CORNER_DISPLACEMENT_RATIOS = (0.04, 0.08, 0.12, 0.18)
MINIMUM_SCORES = (0.30, 0.40, 0.50, 0.60)
MAXIMUM_CANDIDATES = 64

ADAPTER_NAME = "docaligner_pp_lcnet_fusion"
POLICY_FILENAME = f"{ADAPTER_NAME}_policy.json"
POLICY_SCHEMA_VERSION = 1
SMARTDOC_VERSION = "2.0.0"
SMARTDOC_MARKER_SHA256 = {
    "VERSION": "83032357fad1290b27c1ebc7f551ae0df9b0a61676865fb9b224e9ef2e12f17d",
    "LICENCE": "fae21effd8909451cf43888c859b67206882958f429320fb6a8559cf4e78ce6c",
}
_PROTECTED_PATH_TOKENS = frozenset(
    {"evaluator", "groundtruth", "reference", "target"}
)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class _CandidateCache:
    candidates: tuple[FusionCandidate, ...]
    features: tuple[FusionFeatures, ...]
    fallback_index: int | None


@dataclass(frozen=True)
class _ViewCache:
    view: str
    normalized_label: np.ndarray
    image_size: tuple[int, int]
    ratios: Mapping[float, _CandidateCache]


def _policy_document(policy: FusionPolicy) -> dict[str, object]:
    return {
        "schema_version": policy.schema_version,
        "maximum_candidates": policy.maximum_candidates,
        "maximum_corner_displacement_ratio": (
            policy.maximum_corner_displacement_ratio
        ),
        "weights": {name: policy.weights[name] for name in WEIGHT_NAMES},
        "minimum_score": policy.minimum_score,
        "fallback": policy.fallback,
    }


def policy_key(policy: FusionPolicy) -> str:
    return json.dumps(
        _policy_document(policy),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def enumerate_policy_grid() -> tuple[FusionPolicy, ...]:
    policies = []
    for values in product(WEIGHT_VALUES, repeat=4):
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        weights = dict(zip(WEIGHT_NAMES, values, strict=True))
        for displacement_ratio in MAXIMUM_CORNER_DISPLACEMENT_RATIOS:
            for minimum_score in MINIMUM_SCORES:
                policies.append(
                    FusionPolicy(
                        schema_version=POLICY_SCHEMA_VERSION,
                        maximum_candidates=MAXIMUM_CANDIDATES,
                        maximum_corner_displacement_ratio=displacement_ratio,
                        weights=weights,
                        minimum_score=minimum_score,
                        fallback="docaligner",
                    )
                )
    return tuple(policies)


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in Path(os.path.abspath(path)).parts:
        normalized = part.lower().replace("ground-truth", "groundtruth")
        tokens.update(token for token in _TOKEN_SPLIT.split(normalized) if token)
    return tokens


def validate_source_paths(dataset_root: Path, archive_path: Path) -> None:
    for description, path in (
        ("dataset", Path(dataset_root)),
        ("archive", Path(archive_path)),
    ):
        overlap = _path_tokens(path) & _PROTECTED_PATH_TOKENS
        if overlap:
            raise ValueError(
                f"protected {description} path token is forbidden: {sorted(overlap)}"
            )


def _read_regular(path: Path, *, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{description} must be a regular non-symlinked file") from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"{description} must be a regular non-symlinked file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_marker(dataset_root: Path, name: str) -> None:
    payload = _read_regular(dataset_root / name, description=f"SmartDoc {name}")
    if hashlib.sha256(payload).hexdigest() != SMARTDOC_MARKER_SHA256[name]:
        raise ValueError(f"SmartDoc {name} does not identify version {SMARTDOC_VERSION}")


def verify_dataset(
    dataset_root: Path,
    archive_path: Path,
    archive_sha256: str,
    validation_stride: int,
) -> tuple[VerifiedSmartDocSource, tuple[SmartDocRecord, ...]]:
    if archive_sha256 != SMARTDOC_ARCHIVE_SHA256:
        raise ValueError("archive_sha256 must equal the pinned SmartDoc 2015 v2.0.0 hash")
    if isinstance(validation_stride, bool) or not isinstance(validation_stride, int):
        raise TypeError("validation_stride must be an integer")
    if validation_stride < 1:
        raise ValueError("validation_stride must be at least 1")
    dataset_root = Path(dataset_root)
    archive_path = Path(archive_path)
    validate_source_paths(dataset_root, archive_path)
    _verify_marker(dataset_root, "VERSION")
    _verify_marker(dataset_root, "LICENCE")
    source = verify_smartdoc_source(archive_path, dataset_root)
    if source.archive_sha256 != archive_sha256:
        raise ValueError("verified SmartDoc archive hash differs from supplied hash")
    assert_protected_hashes_absent(
        (source.archive_sha256, *(item.sha256 for item in source.files.values()))
    )
    records = load_smartdoc_records(
        dataset_root,
        stride=validation_stride,
        verified_files=source.files,
    )
    validation = tuple(
        sorted(
            (
                record
                for record in records
                if record.background == SMARTDOC_VALIDATION_BACKGROUND
            ),
            key=lambda record: (
                record.sequence,
                record.frame_index,
                record.image_relative_path.as_posix()
                if record.image_relative_path is not None
                else record.image_path.as_posix(),
            ),
        )
    )
    if not validation:
        raise ValueError("SmartDoc background05 validation split is empty")
    if any(record.background != SMARTDOC_VALIDATION_BACKGROUND for record in validation):
        raise ValueError("calibration accepts only SmartDoc background05 records")
    return source, validation


def tensor_rgb_to_bgr(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.shape[0] != 3 or array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("dataset image tensor must be finite CHW RGB")
    rgb = np.transpose(array, (1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN
    rgb_uint8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)


def candidate_features(
    image: np.ndarray,
    signals: FusionSignals,
    candidates: Sequence[FusionCandidate],
) -> tuple[FusionFeatures, ...]:
    if not candidates:
        return ()
    context = _prepare_scoring_context(image)
    neutral_policy = FusionPolicy(
        schema_version=POLICY_SCHEMA_VERSION,
        maximum_candidates=MAXIMUM_CANDIDATES,
        maximum_corner_displacement_ratio=MAXIMUM_CORNER_DISPLACEMENT_RATIOS[0],
        weights={name: 0.25 for name in WEIGHT_NAMES},
        minimum_score=MINIMUM_SCORES[0],
        fallback="docaligner",
    )
    return tuple(
        _score_candidate_with_context(candidate, signals, neutral_policy, context)[1]
        for candidate in candidates
    )


def _cache_candidates(
    image: np.ndarray, signals: FusionSignals
) -> dict[float, _CandidateCache]:
    result = {}
    feature_memo: dict[bytes, FusionFeatures] = {}
    for ratio in MAXIMUM_CORNER_DISPLACEMENT_RATIOS:
        candidates = generate_candidates(
            image,
            signals.docaligner,
            signals.corner_model,
            maximum_candidates=MAXIMUM_CANDIDATES,
            maximum_corner_displacement_ratio=ratio,
        )
        missing = tuple(
            candidate
            for candidate in candidates
            if candidate.corners.tobytes() not in feature_memo
        )
        if missing:
            for candidate, features in zip(
                missing,
                candidate_features(image, signals, missing),
                strict=True,
            ):
                feature_memo[candidate.corners.tobytes()] = features
        features = tuple(feature_memo[item.corners.tobytes()] for item in candidates)
        fallback_index = next(
            (index for index, item in enumerate(candidates) if item.family == "docaligner"),
            None,
        )
        result[ratio] = _CandidateCache(candidates, features, fallback_index)
    return result


def _view_cache(
    view: str,
    dataset: SmartDocCornerDataset,
    *,
    predictor: FusionSignalPredictor,
) -> tuple[_ViewCache, ...]:
    cached = []
    for index in range(len(dataset)):
        tensor, targets = dataset[index]
        image = tensor_rgb_to_bgr(tensor)
        label = targets["corners"].detach().cpu().numpy().astype(np.float64)
        if label.shape != (4, 2) or not np.isfinite(label).all():
            raise ValueError("SmartDoc complete-document labels must be finite TL/TR/BR/BL")
        signals = predictor.predict(image)
        cached.append(
            _ViewCache(
                view=view,
                normalized_label=label,
                image_size=image.shape[:2],
                ratios=_cache_candidates(image, signals),
            )
        )
    return tuple(cached)


def _prediction(
    view: _ViewCache, policy: FusionPolicy
) -> tuple[np.ndarray | None, bool]:
    cache = view.ratios[policy.maximum_corner_displacement_ratio]
    if not cache.candidates:
        return None, False
    values = np.asarray(
        [
            [getattr(features, name) for name in WEIGHT_NAMES]
            for features in cache.features
        ],
        dtype=np.float64,
    )
    stability = np.asarray(
        [features.stability for features in cache.features], dtype=np.float64
    )
    weights = np.asarray([policy.weights[name] for name in WEIGHT_NAMES])
    scores = np.clip(values @ weights * stability, 0.0, 1.0)
    best_index = int(np.argmax(scores))
    fallback = False
    if scores[best_index] < policy.minimum_score:
        if cache.fallback_index is None:
            return None, False
        best_index = cache.fallback_index
        fallback = True
    height, width = view.image_size
    normalized = cache.candidates[best_index].corners.astype(np.float64) / np.asarray(
        (width - 1.0, height - 1.0), dtype=np.float64
    )
    return normalized, fallback


def _evaluate(policy: FusionPolicy, views: Sequence[_ViewCache]) -> dict[str, object]:
    squared = {"clean": 0.0, "occluded": 0.0}
    coordinates = {"clean": 0, "occluded": 0}
    covered = 0
    fallback = 0
    for view in views:
        prediction, fallback_used = _prediction(view, policy)
        if prediction is None:
            continue
        error = prediction - view.normalized_label
        squared[view.view] += float(np.sum(error * error, dtype=np.float64))
        coordinates[view.view] += int(error.size)
        covered += 1
        fallback += int(fallback_used)
    total = len(views)
    full_coverage = covered == total and all(coordinates.values())
    aggregate_rmse = (
        math.sqrt(sum(squared.values()) / sum(coordinates.values()))
        if full_coverage
        else math.inf
    )
    return {
        "policy": policy,
        "policy_key": policy_key(policy),
        "aggregate_rmse": aggregate_rmse,
        "clean_rmse": math.sqrt(squared["clean"] / coordinates["clean"])
        if full_coverage
        else math.inf,
        "occluded_rmse": math.sqrt(squared["occluded"] / coordinates["occluded"])
        if full_coverage
        else math.inf,
        "coverage": covered / total,
        "covered": covered,
        "fallback": fallback,
        "fallback_rate": fallback / total,
    }


def select_best_result(results: Sequence[dict[str, object]]) -> dict[str, object]:
    finite = [
        item
        for item in results
        if float(item["coverage"]) == 1.0
        and math.isfinite(float(item["aggregate_rmse"]))
    ]
    if not finite:
        raise RuntimeError("calibration grid produced no finite full-coverage policy")
    return min(
        finite,
        key=lambda item: (
            float(item["aggregate_rmse"]),
            -float(item["coverage"]),
            str(item["policy_key"]),
        ),
    )


def source_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("source repository must be clean, including untracked files")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("source repository HEAD is not a full Git commit")
    return commit


def _identity(records: Sequence[SmartDocRecord]) -> str:
    identities = [
        f"{record.background}/{record.sequence}/{record.frame_index}"
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(identities, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _command(
    dataset_root: Path,
    archive_path: Path,
    archive_sha256: str,
    model_dir: Path,
    seed: int,
    validation_stride: int,
    cpu_threads: int,
) -> str:
    values = [
        "uv", "run", "--group", "ml", "python", "-m",
        "openscaner.training.calibrate_fusion",
        "--dataset-root", str(dataset_root), "--archive", str(archive_path),
        "--archive-sha256", archive_sha256, "--model-dir", str(model_dir),
        "--seed", str(seed), "--validation-stride", str(validation_stride),
        "--cpu-threads", str(cpu_threads),
        "--policy-output", str(Path(model_dir) / POLICY_FILENAME),
        "--report-output", "artifacts/fusion-calibration/docaligner_pp_lcnet_fusion.json",
        "--manifest", str(Path(model_dir) / "manifest.json"),
    ]
    return shlex.join(values)


def _finite_json(value: object) -> None:
    json.dumps(value, allow_nan=False)


def calibrate_policy(
    *,
    dataset_root,
    archive_path,
    archive_sha256,
    model_dir,
    seed,
    validation_stride,
    cpu_threads,
) -> tuple[FusionPolicy, dict[str, object]]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int):
        raise TypeError("cpu_threads must be an integer")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    dataset_root = Path(dataset_root)
    archive_path = Path(archive_path)
    model_dir = Path(model_dir)
    calibration_commit = source_commit()
    validate_source_paths(dataset_root, archive_path)
    source, records = verify_dataset(
        dataset_root, archive_path, archive_sha256, validation_stride
    )
    records = tuple(
        sorted(records, key=lambda item: (item.sequence, item.frame_index, str(item.image_path)))
    )
    if any(record.background != SMARTDOC_VALIDATION_BACKGROUND for record in records):
        raise ValueError("calibration accepts only SmartDoc background05 records")
    predictor = load_fusion_signal_predictor(model_dir, cpu_threads)
    assert_protected_hashes_absent(
        identity.sha256 for identity in predictor.artifacts.values()
    )
    model_artifacts = {
        name: identity.as_document()
        for name, identity in predictor.artifacts.items()
    }

    clean = SmartDocCornerDataset(records, training=False)
    occluded = SmartDocCornerDataset(
        records,
        training=True,
        global_seed=seed,
        rotation_choices=(0,),
        zoom_out_probability=0,
        photometric_probability=0,
        blur_probability=0,
        shadow_probability=0,
        occlusion_probability=1,
    )
    occluded.set_epoch(0)
    views = (
        *_view_cache("clean", clean, predictor=predictor),
        *_view_cache("occluded", occluded, predictor=predictor),
    )
    grid = enumerate_policy_grid()
    best = select_best_result([_evaluate(policy, views) for policy in grid])
    if source_commit() != calibration_commit:
        raise RuntimeError("source repository commit changed during calibration")
    policy = best["policy"]
    assert isinstance(policy, FusionPolicy)
    clean_count = sum(view.view == "clean" for view in views)
    occluded_count = sum(view.view == "occluded" for view in views)
    report: dict[str, object] = {
        "schema_version": 1,
        "adapter": ADAPTER_NAME,
        "dataset": {
            "name": "SmartDoc 2015",
            "version": SMARTDOC_VERSION,
            "background": SMARTDOC_VALIDATION_BACKGROUND,
            "archive_sha256": source.archive_sha256,
            "identity_sha256": _identity(records),
            "validation_stride": validation_stride,
        },
        "models": model_artifacts,
        "source_commit": calibration_commit,
        "reproducible_command": _command(
            dataset_root, archive_path, archive_sha256, model_dir,
            seed, validation_stride, cpu_threads,
        ),
        "configuration": {
            "seed": seed,
            "cpu_threads": cpu_threads,
            "views": ["clean", "deterministic_occluded_epoch_0"],
        },
        "declared_grid": {
            "weight_names": list(WEIGHT_NAMES),
            "weight_values": list(WEIGHT_VALUES),
            "maximum_corner_displacement_ratios": list(
                MAXIMUM_CORNER_DISPLACEMENT_RATIOS
            ),
            "minimum_scores": list(MINIMUM_SCORES),
            "maximum_candidates": MAXIMUM_CANDIDATES,
            "policy_count": len(grid),
            "stability": "multiplicative_not_searched",
        },
        "selected_policy": _policy_document(policy),
        "metrics": {
            "clean_normalized_corner_rmse": float(best["clean_rmse"]),
            "occluded_normalized_corner_rmse": float(best["occluded_rmse"]),
            "aggregate_normalized_corner_rmse": float(best["aggregate_rmse"]),
            "coverage": float(best["coverage"]),
            "fallback_rate": float(best["fallback_rate"]),
        },
        "counts": {
            "validation_records": len(records),
            "clean_views": clean_count,
            "occluded_views": occluded_count,
            "total_views": len(views),
            "covered_views": int(best["covered"]),
            "fallback_views": int(best["fallback"]),
        },
    }
    _finite_json(report)
    return policy, report


def _manifest_record(
    policy_payload: bytes,
    report_payload: bytes,
    report: Mapping[str, object],
    *,
    policy_filename: str,
    report_filename: str,
) -> dict[str, object]:
    policy_digest = hashlib.sha256(policy_payload).hexdigest()
    return {
        "adapter": ADAPTER_NAME,
        "availability": "locally_trained",
        "model_family": "DocAligner and PP-LCNet-0.5 document-boundary fusion",
        "local_filename": policy_filename,
        "checkpoint_size_bytes": len(policy_payload),
        "sha256": policy_digest,
        "required_runtime": "ONNX Runtime and PyTorch CPU",
        "runtime_detection": "policy filename, schema, byte size, and SHA-256 verified before loading",
        "source": "SmartDoc 2015 v2.0.0 background05 source-only calibration",
        "upstream": "http://smartdoc.univ-lr.fr/",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "notice_path": "src/openscaner/third_party/SMARTDOC15_NOTICE.txt",
        "license_text_path": "src/openscaner/third_party/licenses/CC-BY-4.0-SmartDoc15.txt",
        "fusion_policy": {
            "filename": policy_filename,
            "schema_version": POLICY_SCHEMA_VERSION,
            "sha256": policy_digest,
            "size_bytes": len(policy_payload),
        },
        "calibration_report": {
            "filename": report_filename,
            "sha256": hashlib.sha256(report_payload).hexdigest(),
            "size_bytes": len(report_payload),
        },
        "calibration": {
            "dataset": report.get("dataset", {}),
            "models": report.get("models", {}),
            "source_commit": report.get("source_commit"),
            "reproducible_command": report.get("reproducible_command"),
        },
    }


def _bind_model_identities(
    manifest: dict[str, object], report: Mapping[str, object]
) -> dict[str, object]:
    models = report.get("models")
    if not isinstance(models, Mapping):
        return manifest
    entries = manifest.get("models")
    if not isinstance(entries, list):
        raise ValueError("manifest must contain a models list")
    for adapter, identity in models.items():
        if not isinstance(adapter, str) or not isinstance(identity, Mapping):
            raise ValueError("calibration model identities are invalid")
        matches = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("adapter") == adapter
        ]
        if len(matches) != 1:
            raise ValueError(f"manifest must contain exactly one {adapter} entry")
        entry = matches[0]
        if (
            entry.get("local_filename") != identity.get("filename")
            or entry.get("sha256") != identity.get("sha256")
        ):
            raise ValueError(f"manifest {adapter} identity differs from loaded bytes")
        size_bytes = identity.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ValueError(f"calibration {adapter} model size is invalid")
        declared_size = entry.get("checkpoint_size_bytes", entry.get("size_bytes"))
        if declared_size is not None and declared_size != size_bytes:
            raise ValueError(f"manifest {adapter} size differs from loaded bytes")
        if declared_size is None:
            entry["size_bytes"] = size_bytes
    return manifest


def _validate_committed_calibration(
    payloads: Mapping[Path, bytes],
    *,
    paths: OutputPaths,
    policy_payload: bytes,
    report_payload: bytes,
    manifest_record: Mapping[str, object],
) -> None:
    if payloads[paths.checkpoint] != policy_payload:
        raise RuntimeError("published fusion policy bytes changed")
    if payloads[paths.report] != report_payload:
        raise RuntimeError("published calibration report bytes changed")
    try:
        policy_document = json.loads(payloads[paths.checkpoint])
        FusionPolicy(**policy_document)
        manifest = json.loads(payloads[paths.manifest])
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("published calibration JSON failed validation") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 3
        or not isinstance(manifest.get("models"), list)
        or not all(isinstance(item, dict) for item in manifest["models"])
    ):
        raise RuntimeError("published calibration manifest schema is invalid")
    matches = [
        item
        for item in manifest["models"]
        if isinstance(item, dict) and item.get("adapter") == ADAPTER_NAME
    ]
    if len(matches) != 1 or matches[0] != manifest_record:
        raise RuntimeError("published fusion manifest record failed cross-check")
    models = manifest_record.get("calibration", {}).get("models", {})
    for adapter, identity in models.items():
        source_matches = [
            item
            for item in manifest["models"]
            if isinstance(item, dict) and item.get("adapter") == adapter
        ]
        if len(source_matches) != 1:
            raise RuntimeError("published model identity is absent or duplicated")
        source_entry = source_matches[0]
        if (
            source_entry.get("local_filename") != identity.get("filename")
            or source_entry.get("sha256") != identity.get("sha256")
            or source_entry.get(
                "checkpoint_size_bytes", source_entry.get("size_bytes")
            )
            != identity.get("size_bytes")
        ):
            raise RuntimeError("published model identity failed cross-check")


def publish_calibration(
    policy: FusionPolicy,
    report: dict[str, object],
    *,
    policy_output: Path,
    report_output: Path,
    manifest_path: Path,
    phase_hook: Callable[[str, Path], None] | None = None,
) -> None:
    if not isinstance(policy, FusionPolicy):
        raise TypeError("policy must be a FusionPolicy")
    _finite_json(report)
    raw_paths = tuple(
        Path(os.path.abspath(path))
        for path in (policy_output, report_output, manifest_path)
    )
    for path in raw_paths:
        if path.is_symlink():
            raise ValueError(f"output target must not be a symlink: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"output target must be a regular file: {path}")
        for parent in path.parents:
            if parent.is_symlink():
                raise ValueError(f"output parent must not be symlinked: {parent}")
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"output parent must be a directory: {parent}")
    resolved = tuple(path.resolve(strict=False) for path in raw_paths)
    if len(set(resolved)) != 3:
        raise ValueError("policy, report, and manifest output paths collide")
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError("policy, report, and manifest output paths collide")
    existing = [path for path in raw_paths if path.exists()]
    for index, first in enumerate(existing):
        for second in existing[index + 1 :]:
            if first.samefile(second):
                raise ValueError("policy, report, and manifest output paths collide")
    policy_path, report_path, manifest = resolved
    if policy_path.name != POLICY_FILENAME:
        raise ValueError(f"policy output filename must be exactly {POLICY_FILENAME}")
    if manifest.name != "manifest.json":
        raise ValueError("manifest filename must be exactly manifest.json")
    policy_payload = _json_bytes(_policy_document(policy))
    report_payload = _json_bytes(report)
    record = _manifest_record(
        policy_payload,
        report_payload,
        report,
        policy_filename=policy_path.name,
        report_filename=report_path.name,
    )

    paths = OutputPaths(policy_path, report_path, manifest)

    def validate_committed(payloads: Mapping[Path, bytes]) -> None:
        _validate_committed_calibration(
            payloads,
            paths=paths,
            policy_payload=policy_payload,
            report_payload=report_payload,
            manifest_record=record,
        )

    publish_training_outputs(
        paths,
        checkpoint_bytes=policy_payload,
        report=report,
        manifest_record=record,
        phase_hook=phase_hook,
        validate_committed=validate_committed,
        manifest_transform=lambda document: _bind_model_identities(document, report),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path, dest="archive_path")
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--validation-stride", required=True, type=_positive_int)
    parser.add_argument("--cpu-threads", required=True, type=_positive_int)
    parser.add_argument("--policy-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, dest="manifest_path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy, report = calibrate_policy(
        dataset_root=args.dataset_root,
        archive_path=args.archive_path,
        archive_sha256=args.archive_sha256,
        model_dir=args.model_dir,
        seed=args.seed,
        validation_stride=args.validation_stride,
        cpu_threads=args.cpu_threads,
    )
    publish_calibration(
        policy,
        report,
        policy_output=args.policy_output,
        report_output=args.report_output,
        manifest_path=args.manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
