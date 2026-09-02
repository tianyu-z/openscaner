"""SmartDoc-only calibration for DocAligner-prompted MobileSAM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import shlex
from typing import Callable, Mapping, Sequence

import numpy as np

from openscaner.prompted_sam.policy import (
    PromptedSamPolicy,
    select_candidate,
)
from openscaner.prompted_sam.signals import (
    PromptedSamSignalPredictor,
    PromptedSamSignals,
    load_prompted_sam_predictor,
)
from openscaner.training.calibrate_fusion import (
    source_commit,
    tensor_rgb_to_bgr,
    validate_source_paths,
    verify_dataset,
)
from openscaner.training.smartdoc import (
    SMARTDOC_VALIDATION_BACKGROUND,
    SmartDocCornerDataset,
    SmartDocRecord,
    assert_protected_hashes_absent,
)
from openscaner.training.train_corners import (
    OutputPaths,
    _json_bytes,
    publish_training_outputs,
)


ADAPTER_NAME = "docaligner_prompted_mobile_sam"
POLICY_FILENAME = f"{ADAPTER_NAME}_policy.json"
POLICY_SCHEMA_VERSION = 1
SMARTDOC_VERSION = "2.0.0"
CROP_MARGIN_RATIOS = (0.05, 0.10, 0.15)
FOREGROUND_SCALES = (0.55, 0.70, 0.85)
BACKGROUND_CHOICES = (False, True)
MINIMUM_SAM_SCORES = (0.0, 0.55, 0.70)
WEIGHT_OPTIONS = (
    {
        "predicted_iou": 1.0,
        "mask_stability": 0.0,
        "edge_support": 0.0,
        "coarse_agreement": 0.0,
        "geometric_stability": 0.0,
    },
    {
        "predicted_iou": 0.5,
        "mask_stability": 0.2,
        "edge_support": 0.2,
        "coarse_agreement": 0.1,
        "geometric_stability": 0.0,
    },
    {
        "predicted_iou": 0.35,
        "mask_stability": 0.15,
        "edge_support": 0.25,
        "coarse_agreement": 0.15,
        "geometric_stability": 0.10,
    },
)


@dataclass(frozen=True, slots=True)
class ViewCache:
    view: str
    normalized_label: np.ndarray
    image_size: tuple[int, int]
    signals: Mapping[tuple[float, float, bool], PromptedSamSignals]

    def __post_init__(self) -> None:
        label = np.asarray(self.normalized_label, dtype=np.float64)
        if label.shape != (4, 2) or not np.isfinite(label).all():
            raise ValueError("normalized_label must be a finite 4x2 array")
        if self.view not in {"clean", "occluded"}:
            raise ValueError("view must be clean or occluded")
        if (
            len(self.image_size) != 2
            or any(type(value) is not int or value < 2 for value in self.image_size)
        ):
            raise ValueError("image_size must contain positive integer height and width")
        label = label.copy()
        label.setflags(write=False)
        object.__setattr__(self, "normalized_label", label)
        object.__setattr__(self, "signals", dict(self.signals))


def policy_document(policy: PromptedSamPolicy) -> dict[str, object]:
    return {
        "schema_version": policy.schema_version,
        "crop_margin_ratio": policy.crop_margin_ratio,
        "foreground_scale": policy.foreground_scale,
        "include_background": policy.include_background,
        "minimum_sam_score": policy.minimum_sam_score,
        "weights": dict(policy.weights),
        "fallback": policy.fallback,
    }


def policy_key(policy: PromptedSamPolicy) -> str:
    return json.dumps(
        policy_document(policy),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def geometry_key(policy: PromptedSamPolicy) -> tuple[float, float, bool]:
    return (
        policy.crop_margin_ratio,
        policy.foreground_scale,
        policy.include_background,
    )


def enumerate_policy_grid() -> tuple[PromptedSamPolicy, ...]:
    return tuple(
        PromptedSamPolicy(
            schema_version=POLICY_SCHEMA_VERSION,
            crop_margin_ratio=margin,
            foreground_scale=foreground_scale,
            include_background=include_background,
            minimum_sam_score=minimum_score,
            weights=weights,
            fallback="docaligner",
        )
        for margin, foreground_scale, include_background, minimum_score, weights in product(
            CROP_MARGIN_RATIOS,
            FOREGROUND_SCALES,
            BACKGROUND_CHOICES,
            MINIMUM_SAM_SCORES,
            WEIGHT_OPTIONS,
        )
    )


def geometry_policies(
    policies: Sequence[PromptedSamPolicy],
) -> tuple[PromptedSamPolicy, ...]:
    selected: dict[tuple[float, float, bool], PromptedSamPolicy] = {}
    for policy in sorted(policies, key=policy_key):
        selected.setdefault(geometry_key(policy), policy)
    return tuple(selected[key] for key in sorted(selected))


def view_cache(
    name: str,
    dataset,
    predictor: PromptedSamSignalPredictor,
    policies: Sequence[PromptedSamPolicy],
) -> tuple[ViewCache, ...]:
    geometries = geometry_policies(policies)
    cached: list[ViewCache] = []
    for index in range(len(dataset)):
        tensor, targets = dataset[index]
        image = tensor_rgb_to_bgr(tensor)
        label = targets["corners"].detach().cpu().numpy().astype(np.float64)
        outputs = predictor.predict_many(image, geometries)
        cached.append(
            ViewCache(
                view=name,
                normalized_label=label,
                image_size=image.shape[:2],
                signals={
                    geometry_key(policy): signals
                    for policy, signals in zip(geometries, outputs, strict=True)
                },
            )
        )
    return tuple(cached)


def _prediction(
    view: ViewCache,
    policy: PromptedSamPolicy,
) -> tuple[np.ndarray | None, bool]:
    signals = view.signals.get(geometry_key(policy))
    if signals is None or signals.coarse_corners is None:
        return None, False
    height, width = view.image_size
    source_shape = np.empty((height, width, 3), dtype=np.uint8)
    selected = select_candidate(
        signals.candidates,
        signals.coarse_corners,
        source_shape,
        policy,
    )
    scale = np.array([width - 1.0, height - 1.0], dtype=np.float64)
    return selected.corners.astype(np.float64) / scale, selected.fallback_used


def evaluate_policy(
    policy: PromptedSamPolicy,
    views: Sequence[ViewCache],
) -> dict[str, object]:
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
    coordinate_count = sum(coordinates.values())
    aggregate_rmse = (
        math.sqrt(sum(squared.values()) / coordinate_count)
        if coordinate_count
        else math.inf
    )

    def view_rmse(name: str) -> float | None:
        return (
            math.sqrt(squared[name] / coordinates[name])
            if coordinates[name]
            else None
        )

    total = len(views)
    return {
        "policy": policy,
        "policy_key": policy_key(policy),
        "aggregate_rmse": aggregate_rmse,
        "clean_rmse": view_rmse("clean"),
        "occluded_rmse": view_rmse("occluded"),
        "coverage": covered / total if total else 0.0,
        "covered": covered,
        "fallback": fallback,
        "fallback_rate": fallback / total if total else 0.0,
    }


def select_best_result(results: Sequence[dict[str, object]]) -> dict[str, object]:
    finite = [
        result
        for result in results
        if float(result["coverage"]) == 1.0
        and math.isfinite(float(result["aggregate_rmse"]))
    ]
    if not finite:
        raise RuntimeError("calibration grid produced no finite full-coverage policy")
    return min(
        finite,
        key=lambda result: (
            float(result["aggregate_rmse"]),
            policy_key(result["policy"]),
        ),
    )


def _identity(records: Sequence[SmartDocRecord]) -> str:
    names = [
        f"{record.background}/{record.sequence}/{record.frame_index}"
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(names, separators=(",", ":"), ensure_ascii=True).encode()
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
    return shlex.join(
        [
            "uv",
            "run",
            "--group",
            "ml",
            "python",
            "-m",
            "openscaner.training.calibrate_prompted_sam",
            "--dataset-root",
            str(dataset_root),
            "--archive",
            str(archive_path),
            "--archive-sha256",
            archive_sha256,
            "--model-dir",
            str(model_dir),
            "--seed",
            str(seed),
            "--validation-stride",
            str(validation_stride),
            "--cpu-threads",
            str(cpu_threads),
            "--policy-output",
            str(model_dir / POLICY_FILENAME),
            "--report-output",
            "artifacts/prompted-sam-calibration/docaligner_prompted_mobile_sam.json",
            "--manifest",
            str(model_dir / "manifest.json"),
        ]
    )


def calibrate_policy(
    *,
    dataset_root,
    archive_path,
    archive_sha256,
    model_dir,
    seed,
    validation_stride,
    cpu_threads,
) -> tuple[PromptedSamPolicy, dict[str, object]]:
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
        dataset_root,
        archive_path,
        archive_sha256,
        validation_stride,
    )
    records = tuple(
        sorted(records, key=lambda item: (item.sequence, item.frame_index, str(item.image_path)))
    )
    if any(record.background != SMARTDOC_VALIDATION_BACKGROUND for record in records):
        raise ValueError("calibration accepts only SmartDoc background05 records")
    predictor = load_prompted_sam_predictor(model_dir, cpu_threads)
    assert_protected_hashes_absent(
        identity.sha256 for identity in predictor.artifacts.values()
    )
    model_artifacts = {
        name: identity.as_document() for name, identity in predictor.artifacts.items()
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
    policies = enumerate_policy_grid()
    views = (
        *view_cache("clean", clean, predictor, policies),
        *view_cache("occluded", occluded, predictor, policies),
    )
    best = select_best_result([evaluate_policy(policy, views) for policy in policies])
    if source_commit() != calibration_commit:
        raise RuntimeError("source repository commit changed during calibration")
    policy = best["policy"]
    if not isinstance(policy, PromptedSamPolicy):
        raise RuntimeError("calibration selected an invalid policy")
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
            dataset_root,
            archive_path,
            archive_sha256,
            model_dir,
            seed,
            validation_stride,
            cpu_threads,
        ),
        "configuration": {
            "seed": seed,
            "cpu_threads": cpu_threads,
            "views": ["clean", "deterministic_occluded_epoch_0"],
        },
        "declared_grid": {
            "crop_margin_ratios": list(CROP_MARGIN_RATIOS),
            "foreground_scales": list(FOREGROUND_SCALES),
            "background_choices": list(BACKGROUND_CHOICES),
            "minimum_sam_scores": list(MINIMUM_SAM_SCORES),
            "weight_options": list(WEIGHT_OPTIONS),
            "policy_count": len(policies),
        },
        "selected_policy": policy_document(policy),
        "metrics": {
            "clean_normalized_corner_rmse": best["clean_rmse"],
            "occluded_normalized_corner_rmse": best["occluded_rmse"],
            "aggregate_normalized_corner_rmse": best["aggregate_rmse"],
            "coverage": best["coverage"],
            "fallback_rate": best["fallback_rate"],
        },
        "counts": {
            "validation_records": len(records),
            "clean_views": sum(view.view == "clean" for view in views),
            "occluded_views": sum(view.view == "occluded" for view in views),
            "total_views": len(views),
            "covered_views": best["covered"],
            "fallback_views": best["fallback"],
        },
    }
    json.dumps(report, allow_nan=False)
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
        "model_family": "DocAligner-prompted MobileSAM document boundary",
        "local_filename": policy_filename,
        "checkpoint_size_bytes": len(policy_payload),
        "sha256": policy_digest,
        "required_runtime": "ONNX Runtime and MobileSAM with PyTorch CPU",
        "runtime_detection": "policy and both dependency files verified by exact size and SHA-256",
        "source": "SmartDoc 2015 v2.0.0 background05 source-only calibration",
        "upstream": "https://github.com/DocsaidLab/DocAligner and https://github.com/ChaoningZhang/MobileSAM",
        "license": "Apache-2.0; calibration metadata CC BY 4.0",
        "notice_path": "src/openscaner/third_party/MOBILE_SAM_NOTICE.txt",
        "license_text_path": "src/openscaner/third_party/licenses/Apache-2.0.txt",
        "prompted_sam_policy": {
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
    manifest: dict[str, object],
    report: Mapping[str, object],
) -> dict[str, object]:
    entries = manifest.get("models")
    models = report.get("models")
    if not isinstance(entries, list) or not isinstance(models, Mapping):
        raise ValueError("manifest or calibration model identities are invalid")
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
        expected_size = identity.get("size_bytes")
        declared_size = entry.get("checkpoint_size_bytes", entry.get("size_bytes"))
        if declared_size is not None and declared_size != expected_size:
            raise ValueError(f"manifest {adapter} size differs from loaded bytes")
        if declared_size is None:
            entry["size_bytes"] = expected_size
    return manifest


def publish_calibration(
    policy: PromptedSamPolicy,
    report: dict[str, object],
    *,
    policy_output: Path,
    report_output: Path,
    manifest_path: Path,
    phase_hook: Callable[[str, Path], None] | None = None,
) -> None:
    if not isinstance(policy, PromptedSamPolicy):
        raise TypeError("policy must be PromptedSamPolicy")
    json.dumps(report, allow_nan=False)
    policy_output, report_output, manifest_path = (
        Path(os.path.abspath(path))
        for path in (policy_output, report_output, manifest_path)
    )
    if policy_output.name != POLICY_FILENAME:
        raise ValueError(f"policy output filename must be exactly {POLICY_FILENAME}")
    policy_payload = _json_bytes(policy_document(policy))
    report_payload = _json_bytes(report)
    record = _manifest_record(
        policy_payload,
        report_payload,
        report,
        policy_filename=policy_output.name,
        report_filename=report_output.name,
    )
    paths = OutputPaths(policy_output, report_output, manifest_path)

    def validate_committed(payloads: Mapping[Path, bytes]) -> None:
        if payloads[paths.checkpoint] != policy_payload:
            raise RuntimeError("published prompted-SAM policy bytes changed")
        if payloads[paths.report] != report_payload:
            raise RuntimeError("published prompted-SAM report bytes changed")
        manifest = json.loads(payloads[paths.manifest])
        matches = [
            item
            for item in manifest.get("models", [])
            if isinstance(item, dict) and item.get("adapter") == ADAPTER_NAME
        ]
        if len(matches) != 1 or matches[0] != record:
            raise RuntimeError("published prompted-SAM manifest record failed cross-check")

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
