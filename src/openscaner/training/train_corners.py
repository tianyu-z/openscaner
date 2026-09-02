"""Reproducible SmartDoc corner-model training entrypoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import platform
import random
import stat
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
import timm
import torch
import torchvision
from huggingface_hub import __version__ as huggingface_hub_version
from torch.utils.data import DataLoader, Dataset

from openscaner.models.corner_heatmap import (
    BACKBONE_ALIASES,
    LOCAL_SOFT_ARGMAX_RADIUS,
    PRETRAINED_WEIGHT_SPECS,
    build_model,
    decode_corner_heatmaps,
)
from openscaner.training.corner_loss import corner_heatmap_loss
from openscaner.training.smartdoc import (
    SMARTDOC_ARCHIVE_SHA256,
    SMARTDOC_GLOBAL_SEED,
    SMARTDOC_INPUT_SIZE,
    SMARTDOC_TARGET_SIZE,
    SmartDocCornerDataset,
    SmartDocRecord,
    VerifiedSmartDocSource,
    load_smartdoc_records,
    split_smartdoc_records,
    verify_smartdoc_source,
)

SMARTDOC_VERSION = "2.0.0"
SMARTDOC_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
SMARTDOC_MARKER_SHA256 = {
    "VERSION": "83032357fad1290b27c1ebc7f551ae0df9b0a61676865fb9b224e9ef2e12f17d",
    "LICENCE": "fae21effd8909451cf43888c859b67206882958f429320fb6a8559cf4e78ce6c",
}
SMARTDOC_SOURCE_URL = "http://smartdoc.univ-lr.fr/"
SMARTDOC_UPSTREAM = "https://github.com/jchazalon/smartdoc15-ch1-pywrapper"
SMARTDOC_NOTICE_PATH = "src/openscaner/third_party/SMARTDOC15_NOTICE.txt"
SMARTDOC_LICENSE_TEXT_PATH = (
    "src/openscaner/third_party/licenses/CC-BY-4.0-SmartDoc15.txt"
)
TIMM_UPSTREAM_URL = "https://github.com/huggingface/pytorch-image-models"
TIMM_NOTICE_PATH = "src/openscaner/third_party/TIMM_PRETRAINED_WEIGHTS_NOTICE.txt"
TIMM_LICENSE_TEXT_PATH = "src/openscaner/third_party/licenses/Apache-2.0.txt"
CONFIDENCE_THRESHOLD = 0.5
CONFIDENCE_COVERAGE_DEFINITION = (
    "fraction of validation samples whose mean corner confidence is at least 0.5; "
    "not geometric detection success"
)
MODEL_ALIASES = ("mobilenetv3_small", "pp_lcnet_050")
MODEL_FILENAMES = {
    "mobilenetv3_small": "mobilenetv3_small_corner.pth",
    "pp_lcnet_050": "pp_lcnet_050_corner.pth",
}
ADAPTER_NAMES = {
    "mobilenetv3_small": "mobilenetv3_small_corner",
    "pp_lcnet_050": "pp_lcnet_050_corner",
}
MODEL_FAMILIES = {
    "mobilenetv3_small": "MobileNetV3-Small corner heatmap model",
    "pp_lcnet_050": "PP-LCNet-0.5 corner heatmap model",
}
WEIGHT_DECAY = 1e-4
AUGMENTATION_CONTRACT = {
    "operation_order": [
        "rotation",
        "zoom_out",
        "photometric",
        "blur",
        "shadow",
        "corner_occlusion",
    ],
    "rotation_degrees": {
        "choices": [0, 90, 180, 270],
        "selection": "uniform",
    },
    "zoom_out": {
        "probability": 0.5,
        "scale_range": [0.45, 0.82],
        "canvas": "procedural_clutter",
        "placement": "uniform_valid_offset",
        "interpolation": "opencv_INTER_AREA",
    },
    "photometric": {
        "probability": 1.0,
        "gain_range": [0.78, 1.22],
        "bias_range": [-24.0, 24.0],
        "operation": "clip_uint8",
    },
    "blur": {
        "probability": 0.35,
        "sigma_range": [0.6, 2.0],
        "border": "opencv_BORDER_REFLECT_101",
    },
    "shadow": {
        "probability": 0.4,
        "center_fraction_range": [0.2, 0.8],
        "x_axis_fraction_range": [0.18, 0.35],
        "y_axis_fraction_range": [0.12, 0.28],
        "angle_degrees_range": [0.0, 180.0],
        "edge_sigma": {"fraction": 0.018, "minimum_pixels": 1.0},
        "strength_range": [0.22, 0.52],
    },
    "corner_occlusion": {
        "probability": 0.5,
        "corner_selection": "uniform_TL_TR_BR_BL",
        "shape": "solid_circle",
        "radius_fraction_range": [0.055, 0.1],
        "minimum_radius_pixels": 8,
        "fill": "uniform_random_RGB",
    },
    "heatmap_sigma": 1.5,
    "label_policy": "complete unoccluded corner heatmaps and mask remain unchanged by occlusion",
}
PREPROCESSING_METADATA = {
    "input_size": [SMARTDOC_INPUT_SIZE, SMARTDOC_INPUT_SIZE],
    "color_space": "RGB",
    "pixel_scale": "uint8_to_float32_0_1",
    "resize_interpolation": "opencv_INTER_AREA",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
TARGET_METADATA = {
    "size": [SMARTDOC_TARGET_SIZE, SMARTDOC_TARGET_SIZE],
    "channel_order": ["TL", "TR", "BR", "BL"],
    "corner_coordinates": "normalized_xy_0_1",
    "auxiliary_channel": "document_mask_training_only",
}
DECODER_METADATA = {
    "name": "local_soft_argmax",
    "local_soft_argmax_radius": LOCAL_SOFT_ARGMAX_RADIUS,
    "coordinates": "normalized_xy_0_1",
    "confidence": "mean_sigmoid_global_peak_logit",
    "detection_threshold": CONFIDENCE_THRESHOLD,
}


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: Path
    archive: Path
    model: str
    device: str
    model_dir: Path
    artifacts_dir: Path
    manifest: Path
    image_size: int = SMARTDOC_INPUT_SIZE
    target_size: int = SMARTDOC_TARGET_SIZE
    train_stride: int = 5
    validation_stride: int = 5
    epochs: int = 16
    batch_size: int = 8
    patience: int = 5
    learning_rate: float = 3e-4
    seed: int = SMARTDOC_GLOBAL_SEED
    cpu_threads: int = 1
    workers: int = 0


@dataclass(frozen=True)
class PreparedRecords:
    train_records: tuple[SmartDocRecord, ...]
    validation_records: tuple[SmartDocRecord, ...]
    train_identities: tuple[str, ...]
    validation_identities: tuple[str, ...]
    identity_sha256: str
    archive_sha256: str


@dataclass(frozen=True)
class ValidationMetrics:
    loss: dict[str, float]
    normalized_corner_rmse: float
    detection_rate: float
    mean_confidence: float


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    epochs_completed: int
    best_state_dict: dict[str, torch.Tensor]
    best_validation: ValidationMetrics
    history: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class OutputPaths:
    checkpoint: Path
    report: Path
    manifest: Path


@dataclass(frozen=True)
class _OutputSnapshot:
    payload: bytes
    mode: int
    atime_ns: int
    mtime_ns: int
    device: int
    inode: int
    size: int
    ctime_ns: int
    nlink: int


@dataclass(frozen=True)
class _OutputDirectory:
    path: Path
    descriptor: int
    device: int
    inode: int


def _exact_size(name: str, expected: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be exactly {expected}") from error
        if parsed != expected:
            raise argparse.ArgumentTypeError(f"{name} must be exactly {expected}")
        return parsed

    return parse


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a reproducible SmartDoc four-corner heatmap model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ALIASES, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-size",
        type=_exact_size("image-size", SMARTDOC_INPUT_SIZE),
        default=SMARTDOC_INPUT_SIZE,
        help="fixed square model input size",
    )
    parser.add_argument(
        "--target-size",
        type=_exact_size("target-size", SMARTDOC_TARGET_SIZE),
        default=SMARTDOC_TARGET_SIZE,
        help="fixed square heatmap target size",
    )
    parser.add_argument(
        "--train-stride", type=_positive_int, default=5, help="training frame stride"
    )
    parser.add_argument(
        "--validation-stride",
        type=_positive_int,
        default=5,
        help="validation frame stride",
    )
    parser.add_argument("--epochs", type=_positive_int, default=16, help="maximum epochs")
    parser.add_argument("--batch-size", type=_positive_int, default=8, help="batch size")
    parser.add_argument(
        "--patience", type=_positive_int, default=5, help="early-stopping patience"
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=3e-4,
        help="initial AdamW learning rate",
    )
    parser.add_argument(
        "--seed", type=int, default=SMARTDOC_GLOBAL_SEED, help="global random seed"
    )
    parser.add_argument(
        "--cpu-threads",
        type=_exact_size("cpu-threads", 1),
        default=1,
        help="fixed CPU thread count for latency measurement",
    )
    parser.add_argument(
        "--workers", type=_nonnegative_int, default=0, help="data-loader workers"
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse the public CLI into an immutable library configuration."""
    return TrainingConfig(**vars(_parser().parse_args(argv)))


def _verify_dataset_marker(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"SmartDoc {description} file is missing or invalid")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"SmartDoc {description} file is unreadable") from error
    if digest != SMARTDOC_MARKER_SHA256[description]:
        expected = SMARTDOC_VERSION if description == "VERSION" else SMARTDOC_LICENSE
        raise ValueError(f"SmartDoc {description} must identify {expected}")


def _record_identity(record: SmartDocRecord) -> str:
    return f"{record.background}/{record.sequence}/{record.frame_index}"


def load_verified_splits(
    config: TrainingConfig,
    *,
    record_loader: Callable[..., tuple[SmartDocRecord, ...]] = load_smartdoc_records,
    source_verifier: Callable[[Path, Path], VerifiedSmartDocSource] = (
        verify_smartdoc_source
    ),
) -> PreparedRecords:
    """Verify SmartDoc provenance, load efficiently, and freeze the split identity."""
    dataset_root = config.dataset_root.resolve()
    _verify_dataset_marker(dataset_root / "VERSION", "VERSION")
    _verify_dataset_marker(dataset_root / "LICENCE", "LICENCE")

    source = source_verifier(config.archive, dataset_root)
    archive_sha256 = source.archive_sha256
    if archive_sha256 != SMARTDOC_ARCHIVE_SHA256:
        raise ValueError(
            "SmartDoc archive SHA-256 mismatch: "
            f"expected {SMARTDOC_ARCHIVE_SHA256}, got {archive_sha256}"
        )
    loading_stride = math.gcd(config.train_stride, config.validation_stride)
    records = record_loader(
        dataset_root,
        stride=loading_stride,
        verified_files=source.files,
    )
    train_records, _ = split_smartdoc_records(records, config.train_stride)
    _, validation_records = split_smartdoc_records(records, config.validation_stride)
    if not train_records or not validation_records:
        raise ValueError("SmartDoc train and validation splits must both be non-empty")

    train_identities = tuple(_record_identity(record) for record in train_records)
    validation_identities = tuple(
        _record_identity(record) for record in validation_records
    )
    identity_payload = json.dumps(
        {"train": train_identities, "validation": validation_identities},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return PreparedRecords(
        train_records=train_records,
        validation_records=validation_records,
        train_identities=train_identities,
        validation_identities=validation_identities,
        identity_sha256=hashlib.sha256(identity_payload).hexdigest(),
        archive_sha256=archive_sha256,
    )


def select_device(requested: str) -> torch.device:
    """Resolve auto to MPS or CPU and reject every other accelerator."""
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested not in {"cpu", "mps"}:
        raise ValueError("corner training supports CPU or MPS only")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the CPU/MPS training pipeline."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create independently seeded deterministic train and validation loaders."""
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "persistent_workers": config.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        worker_init_fn=partial(_seed_worker, base_seed=config.seed),
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.seed + 1),
        worker_init_fn=partial(_seed_worker, base_seed=config.seed + 1),
        **common,
    )
    return train_loader, validation_loader


def _move_targets(
    targets: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in targets.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one optimization epoch with the shared corner heatmap loss."""
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0
    for images, targets in loader:
        images = images.to(device)
        moved_targets = _move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        losses = corner_heatmap_loss(model(images), moved_targets)
        losses["total"].backward()
        optimizer.step()
        batch_size = int(images.shape[0])
        sample_count += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * batch_size
    if sample_count == 0:
        raise ValueError("training loader must not be empty")
    return {name: total / sample_count for name, total in totals.items()}


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> ValidationMetrics:
    """Evaluate held-out RMSE, confidence coverage, mean confidence, and losses."""
    model.eval()
    loss_totals: dict[str, float] = {}
    sample_count = 0
    squared_error = 0.0
    coordinate_count = 0
    detected_count = 0
    confidence_total = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            moved_targets = _move_targets(targets, device)
            predictions = model(images)
            losses = corner_heatmap_loss(predictions, moved_targets)
            corners, confidence = decode_corner_heatmaps(
                predictions["corner_heatmaps"]
            )
            differences = corners - moved_targets["corners"]
            squared_error += float(differences.square().sum().item())
            coordinate_count += differences.numel()
            batch_size = int(images.shape[0])
            sample_count += batch_size
            detected_count += int(
                (confidence >= CONFIDENCE_THRESHOLD).sum().item()
            )
            confidence_total += float(confidence.sum().item())
            for name, value in losses.items():
                loss_totals[name] = loss_totals.get(name, 0.0) + float(value.item()) * batch_size
    if sample_count == 0 or coordinate_count == 0:
        raise ValueError("validation loader must not be empty")
    return ValidationMetrics(
        loss={name: total / sample_count for name, total in loss_totals.items()},
        normalized_corner_rmse=math.sqrt(squared_error / coordinate_count),
        detection_rate=detected_count / sample_count,
        mean_confidence=confidence_total / sample_count,
    )


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    *,
    train_epoch: Callable[..., dict[str, float]] = train_one_epoch,
    evaluate_epoch: Callable[..., ValidationMetrics] = evaluate_model,
) -> FitResult:
    """Fit with AdamW/cosine decay and select strictly by held-out RMSE."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    best_epoch = 0
    best_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: ValidationMetrics | None = None
    stale_epochs = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        set_epoch = getattr(train_loader.dataset, "set_epoch", None)
        if not callable(set_epoch):
            raise TypeError("training dataset must provide set_epoch(epoch)")
        set_epoch(epoch - 1)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate_epoch(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": dict(train_loss),
                "validation_loss": dict(validation.loss),
                "validation_normalized_corner_rmse": validation.normalized_corner_rmse,
                "validation_confidence_coverage_at_0_5": validation.detection_rate,
                "validation_mean_confidence": validation.mean_confidence,
            }
        )
        if validation.normalized_corner_rmse < best_rmse:
            best_rmse = validation.normalized_corner_rmse
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            best_validation = validation
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= config.patience:
            break

    if best_state is None or best_validation is None:
        raise RuntimeError("training completed without a validation result")
    return FitResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_state_dict=best_state,
        best_validation=best_validation,
        history=tuple(history),
    )


def resolve_output_paths(config: TrainingConfig) -> OutputPaths:
    """Resolve and validate the three independently named output targets."""
    filename = MODEL_FILENAMES[config.model]
    candidates = tuple(
        Path(os.path.abspath(path))
        for path in (
            config.model_dir / filename,
            config.artifacts_dir / f"{Path(filename).stem}.json",
            config.manifest,
        )
    )
    for path in candidates:
        if path.is_symlink():
            raise ValueError(f"output target must not be a symlink: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"output target must be a regular file: {path}")
        for ancestor in path.parents:
            if ancestor.is_symlink():
                raise ValueError(
                    f"output path must not have a symlinked ancestor: {ancestor}"
                )
            if ancestor.exists() and not ancestor.is_dir():
                raise ValueError(
                    f"output parent must be a directory: {ancestor}"
                )
    resolved = tuple(path.resolve(strict=False) for path in candidates)
    if len(set(resolved)) != len(resolved):
        raise ValueError("checkpoint, report, and manifest paths must not collide")
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError(
                    "checkpoint, report, and manifest targets must not be an "
                    "ancestor or descendant of one another"
                )
    existing = [path for path in candidates if path.exists()]
    for index, first in enumerate(existing):
        for second in existing[index + 1 :]:
            if first.samefile(second):
                raise ValueError("checkpoint, report, and manifest paths must not collide")
    return OutputPaths(*resolved)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _updated_manifest(
    payload: bytes | None, record: dict[str, object]
) -> dict[str, object]:
    try:
        manifest = json.loads(payload.decode("utf-8") if payload is not None else "")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest must be readable valid JSON") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 3
    ):
        raise ValueError("manifest schema_version must be exactly 3")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("manifest must contain a models list")
    if not all(isinstance(entry, dict) for entry in models):
        raise ValueError("manifest model entries must be objects")
    if record.get("availability") != "locally_trained":
        raise ValueError("manifest training record must be locally_trained")
    adapter = record.get("adapter")
    manifest["models"] = [
        entry
        for entry in models
        if entry.get("adapter") != adapter
    ] + [record]
    return manifest


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_chain(path: Path, *, create: bool) -> int:
    if not path.is_absolute():
        raise ValueError("output parent must be absolute")
    flags = _directory_flags()
    descriptor = os.open(os.sep, flags)
    current = Path(os.sep)
    try:
        for component in path.parts[1:]:
            child = current / component
            try:
                next_descriptor = os.open(
                    component, flags, dir_fd=descriptor
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            current = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_output_directories(paths: Sequence[Path]) -> dict[Path, _OutputDirectory]:
    directories: dict[Path, _OutputDirectory] = {}
    try:
        for parent in dict.fromkeys(path.parent for path in paths):
            try:
                descriptor = _open_directory_chain(parent, create=True)
            except OSError as error:
                raise ValueError(
                    f"output path has a symlinked ancestor or invalid parent: {parent}"
                ) from error
            status = os.fstat(descriptor)
            directories[parent] = _OutputDirectory(
                path=parent,
                descriptor=descriptor,
                device=status.st_dev,
                inode=status.st_ino,
            )
        return directories
    except BaseException:
        for directory in directories.values():
            os.close(directory.descriptor)
        raise


def _revalidate_output_directories(
    directories: Mapping[Path, _OutputDirectory],
) -> None:
    for directory in directories.values():
        try:
            descriptor = _open_directory_chain(directory.path, create=False)
        except OSError as error:
            raise ValueError(
                f"output parent changed during publication: {directory.path}"
            ) from error
        try:
            status = os.fstat(descriptor)
            if (status.st_dev, status.st_ino) != (
                directory.device,
                directory.inode,
            ):
                raise ValueError(
                    f"output parent changed during publication: {directory.path}"
                )
        finally:
            os.close(descriptor)


def _read_output(directory: _OutputDirectory, path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.name, flags, dir_fd=directory.descriptor)
    with _fdopen_owned(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"output target must be a regular file: {path}")
        return stream.read()


def _snapshot_output(
    directory: _OutputDirectory,
    path: Path,
    *,
    maximum_size: int | None = None,
) -> _OutputSnapshot | None:
    if maximum_size is not None and (
        isinstance(maximum_size, bool) or maximum_size < 1
    ):
        raise ValueError("maximum manifest snapshot size must be positive")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=directory.descriptor)
    except FileNotFoundError:
        return None
    with _fdopen_owned(descriptor, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"output target must be a regular file: {path}")
        payload = (
            stream.read()
            if maximum_size is None
            else stream.read(maximum_size + 1)
        )
        if maximum_size is not None and len(payload) > maximum_size:
            raise ValueError("manifest snapshot exceeds maximum size")
        snapshot_status = os.fstat(stream.fileno())
        return _OutputSnapshot(
            payload=payload,
            mode=stat.S_IMODE(status.st_mode),
            atime_ns=status.st_atime_ns,
            mtime_ns=status.st_mtime_ns,
            device=snapshot_status.st_dev,
            inode=snapshot_status.st_ino,
            size=snapshot_status.st_size,
            ctime_ns=snapshot_status.st_ctime_ns,
            nlink=snapshot_status.st_nlink,
        )


def _write_output(
    directory: _OutputDirectory,
    path: Path,
    payload: bytes,
    mode: int,
    write_bytes: Callable[[Path, bytes], None] | None,
) -> None:
    if write_bytes is not None:
        write_bytes(path, payload)
        return
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path.name, flags, 0o644, dir_fd=directory.descriptor)
    with _fdopen_owned(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fdopen_owned(descriptor: int, mode: str) -> BinaryIO:
    try:
        return os.fdopen(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        raise


def _replace_output(
    directory: _OutputDirectory,
    source: Path,
    destination: Path,
    replace: Callable[[Path, Path], None] | None,
) -> None:
    if replace is not None:
        replace(source, destination)
        return
    os.replace(
        source.name,
        destination.name,
        src_dir_fd=directory.descriptor,
        dst_dir_fd=directory.descriptor,
    )


def _unlink_output(directory: _OutputDirectory, path: Path) -> None:
    try:
        os.unlink(path.name, dir_fd=directory.descriptor)
    except FileNotFoundError:
        pass


def _fsync_output_directories(
    directories: Mapping[Path, _OutputDirectory], paths: Sequence[Path]
) -> None:
    seen: set[int] = set()
    for path in paths:
        descriptor = directories[path.parent].descriptor
        if descriptor not in seen:
            os.fsync(descriptor)
            seen.add(descriptor)


def publish_training_outputs(
    paths: OutputPaths,
    *,
    checkpoint_bytes: bytes,
    report: dict[str, object],
    manifest_record: dict[str, object],
    validate_published: Callable[[], None] | None = None,
    validate_published_bytes: Callable[[bytes], None] | None = None,
    write_bytes: Callable[[Path, bytes], None] | None = None,
    replace: Callable[[Path, Path], None] | None = None,
    phase_hook: Callable[[str, Path], None] | None = None,
    validate_committed: Callable[[Mapping[Path, bytes]], None] | None = None,
    manifest_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
    maximum_manifest_snapshot_size: int | None = None,
    manifest_snapshot_validator: Callable[[_OutputSnapshot | None], None] | None = None,
) -> None:
    """Publish outputs with per-file atomic replace and all-or-old rollback.

    The three paths are not globally atomic. The checkpoint and report are
    replaced first, then the manifest is replaced last as the logical commit
    point. Readers must treat the manifest as authoritative. Readers must
    validate its checkpoint hash and size before loading the referenced file.
    Failure cleanup does not remove newly created output directories because
    safely claiming a pathname after a concurrent replacement is impossible.
    """
    output_paths = (paths.checkpoint, paths.report, paths.manifest)
    directories = _open_output_directories(output_paths)
    staged: dict[Path, Path] = {}
    rollback_files: list[Path] = []
    replacement_attempts: list[Path] = []
    try:
        _revalidate_output_directories(directories)
        originals = {
            path: _snapshot_output(
                directories[path.parent],
                path,
                maximum_size=(
                    maximum_manifest_snapshot_size
                    if path == paths.manifest
                    else None
                ),
            )
            for path in output_paths
        }
        manifest_snapshot = originals[paths.manifest]
        if manifest_snapshot_validator is not None:
            manifest_snapshot_validator(manifest_snapshot)

        def validate_current_manifest_snapshot() -> None:
            if manifest_snapshot_validator is not None:
                manifest_snapshot_validator(
                    _snapshot_output(
                        directories[paths.manifest.parent],
                        paths.manifest,
                        maximum_size=maximum_manifest_snapshot_size,
                    )
                )

        manifest = _updated_manifest(
            None if manifest_snapshot is None else manifest_snapshot.payload,
            manifest_record,
        )
        if manifest_transform is not None:
            manifest = manifest_transform(manifest)
            if (
                not isinstance(manifest, dict)
                or type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != 3
                or not isinstance(manifest.get("models"), list)
                or not all(
                    isinstance(entry, dict) for entry in manifest["models"]
                )
            ):
                raise ValueError("transformed manifest schema is incompatible")
        payloads = (
            (paths.checkpoint, checkpoint_bytes),
            (paths.report, _json_bytes(report)),
            (paths.manifest, _json_bytes(manifest)),
        )
        for target, payload in payloads:
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            staged[target] = temporary
            _revalidate_output_directories(directories)
            if phase_hook is not None:
                phase_hook("before_stage_write", target)
            _write_output(
                directories[target.parent],
                temporary,
                payload,
                originals[target].mode if originals[target] is not None else 0o644,
                write_bytes,
            )
            _revalidate_output_directories(directories)
        for target in (paths.checkpoint, paths.report):
            _revalidate_output_directories(directories)
            if phase_hook is not None:
                phase_hook("before_replace", target)
            validate_current_manifest_snapshot()
            replacement_attempts.append(target)
            _replace_output(
                directories[target.parent], staged[target], target, replace
            )
            _revalidate_output_directories(directories)
        _fsync_output_directories(directories, (paths.checkpoint, paths.report))
        _revalidate_output_directories(directories)
        published_checkpoint = _read_output(
            directories[paths.checkpoint.parent], paths.checkpoint
        )
        if validate_published_bytes is not None:
            validate_published_bytes(published_checkpoint)
        if validate_published is not None:
            validate_published()
        _revalidate_output_directories(directories)
        if phase_hook is not None:
            phase_hook("before_replace", paths.manifest)
        validate_current_manifest_snapshot()
        replacement_attempts.append(paths.manifest)
        _replace_output(
            directories[paths.manifest.parent],
            staged[paths.manifest],
            paths.manifest,
            replace,
        )
        _revalidate_output_directories(directories)
        _fsync_output_directories(directories, (paths.manifest,))
        if validate_committed is not None:
            committed = {
                path: _read_output(directories[path.parent], path)
                for path in output_paths
            }
            validate_committed(committed)
    except BaseException:
        rollback_error: BaseException | None = None
        rollback_targets = reversed(replacement_attempts)
        for target in rollback_targets:
            try:
                try:
                    _revalidate_output_directories(directories)
                except ValueError:
                    pass
                if phase_hook is not None:
                    phase_hook("before_rollback", target)
                original = originals[target]
                directory = directories[target.parent]
                if original is None:
                    _unlink_output(directory, target)
                else:
                    rollback = target.with_name(
                        f".{target.name}.{uuid.uuid4().hex}.rollback.tmp"
                    )
                    rollback_files.append(rollback)
                    _write_output(
                        directory,
                        rollback,
                        original.payload,
                        original.mode,
                        write_bytes,
                    )
                    _replace_output(directory, rollback, target, replace)
                    os.chmod(
                        target.name,
                        original.mode,
                        dir_fd=directory.descriptor,
                        follow_symlinks=False,
                    )
                    os.utime(
                        target.name,
                        ns=(original.atime_ns, original.mtime_ns),
                        dir_fd=directory.descriptor,
                        follow_symlinks=False,
                    )
                _fsync_output_directories(directories, (target,))
            except BaseException as error:
                rollback_error = error
        if rollback_error is not None:
            raise RuntimeError("failed to roll back training outputs") from rollback_error
        raise
    finally:
        for temporary in (*staged.values(), *rollback_files):
            _unlink_output(directories[temporary.parent], temporary)
        for directory in directories.values():
            os.close(directory.descriptor)


def _default_dataset_factory(
    records: Sequence[SmartDocRecord], *, training: bool, global_seed: int
) -> Dataset:
    return SmartDocCornerDataset(
        records, training=training, global_seed=global_seed
    )


def _default_model_builder(alias: str, *, pretrained: bool) -> torch.nn.Module:
    return build_model(alias, pretrained=pretrained)


def _load_checkpoint_bytes_cpu(payload: bytes) -> dict[str, object]:
    loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError("corner checkpoint must contain a dictionary")
    return loaded


@dataclass(frozen=True)
class TrainingDependencies:
    source_verifier: Callable[[Path, Path], VerifiedSmartDocSource] = (
        verify_smartdoc_source
    )
    record_loader: Callable[..., tuple[SmartDocRecord, ...]] = load_smartdoc_records
    dataset_factory: Callable[..., Dataset] = _default_dataset_factory
    model_builder: Callable[..., torch.nn.Module] = _default_model_builder
    latency_runner: Callable[[torch.nn.Module, torch.Tensor, int], float] | None = None
    checkpoint_bytes_loader: Callable[[bytes], dict[str, object]] = (
        _load_checkpoint_bytes_cpu
    )
    source_repository_provider: Callable[[], dict[str, object]] = lambda: (
        _source_repository_metadata()
    )
    runtime_environment_provider: Callable[[], dict[str, object]] = lambda: (
        _runtime_environment_metadata()
    )
    write_bytes: Callable[[Path, bytes], None] | None = None
    replace: Callable[[Path, Path], None] | None = None


def cpu_latency_ms(
    model: torch.nn.Module, image: torch.Tensor, cpu_threads: int
) -> float:
    """Measure median one-thread CPU latency across exactly seven timed runs."""
    if cpu_threads != 1:
        raise ValueError("corner checkpoint latency must use exactly one CPU thread")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(cpu_threads)
        model = model.to("cpu").eval()
        sample = image.unsqueeze(0).to("cpu")
        with torch.inference_mode():
            for _ in range(2):
                model(sample)
            timings: list[float] = []
            for _ in range(7):
                started = time.perf_counter()
                model(sample)
                timings.append((time.perf_counter() - started) * 1000.0)
        return float(statistics.median(timings))
    finally:
        torch.set_num_threads(previous_threads)


def _checkpoint_bytes(checkpoint: Mapping[str, object]) -> bytes:
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue()


def _source_repository_metadata() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to record training source repository state") from error
    if len(commit) != 40:
        raise RuntimeError("training source commit must be a full Git SHA")
    return {"commit": commit, "clean": not bool(status.strip())}


def _runtime_environment_metadata() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "huggingface_hub": huggingface_hub_version,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "timm": timm.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }


def _training_config_metadata(config: TrainingConfig) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        metadata[field.name] = str(value) if isinstance(value, Path) else value
    return metadata


def _artifact_licensing_metadata() -> dict[str, object]:
    return {
        "consumer_notice": (
            "Review both layers before using or redistributing this locally "
            "trained checkpoint; the SmartDoc dataset license alone does not "
            "license the architecture or pretrained weights."
        ),
        "layers": {
            "architecture_and_pretrained_weights": {
                "license": "Apache-2.0",
                "license_text_path": TIMM_LICENSE_TEXT_PATH,
                "notice_path": TIMM_NOTICE_PATH,
                "upstream": TIMM_UPSTREAM_URL,
            },
            "training_dataset": {
                "license": SMARTDOC_LICENSE,
                "license_text_path": SMARTDOC_LICENSE_TEXT_PATH,
                "notice_path": SMARTDOC_NOTICE_PATH,
                "source": SMARTDOC_SOURCE_URL,
                "upstream": SMARTDOC_UPSTREAM,
            },
        },
    }


def _confidence_coverage_metadata(rate: float) -> dict[str, object]:
    return {
        "threshold": CONFIDENCE_THRESHOLD,
        "rate": rate,
        "definition": CONFIDENCE_COVERAGE_DEFINITION,
    }


def _cpu_latency_metadata(
    latency_ms: float, runtime_environment: dict[str, object]
) -> dict[str, object]:
    return {
        "threads": 1,
        "warmup_runs": 2,
        "timed_runs": 7,
        "median_ms": latency_ms,
        "environment_specific": True,
        "context": runtime_environment,
        "protocol": {
            "device": "cpu",
            "batch_size": 1,
            "threads": 1,
            "warmup_runs": 2,
            "timed_runs": 7,
            "statistic": "median",
            "timer": "time.perf_counter",
        },
    }


def _architecture_metadata(alias: str) -> dict[str, object]:
    spec = PRETRAINED_WEIGHT_SPECS[alias]
    immutable_download_url = (
        f"https://huggingface.co/{spec['repo_id']}/resolve/"
        f"{spec['revision']}/{spec['filename']}"
    )
    source = {
        **spec,
        "source_type": "huggingface_hub",
        "timm_backbone": BACKBONE_ALIASES[alias],
        "huggingface_model_url": f"https://huggingface.co/{spec['repo_id']}",
        "immutable_download_url": immutable_download_url,
        "timm_upstream_url": TIMM_UPSTREAM_URL,
        "verification_evidence": "training_time_verified_pinned_bytes",
        "training_time_pin_enforced": True,
        "training_loader": (
            "verified safetensors byte buffer passed to timm as state_dict"
        ),
    }
    return {
        "alias": alias,
        "family": MODEL_FAMILIES[alias],
        "timm_backbone": BACKBONE_ALIASES[alias],
        "library": "timm",
        "timm_version": timm.__version__,
        "pretrained": True,
        "initialization": (
            "ImageNet pretrained weights loaded from training-time verified immutable bytes"
        ),
        "pretrained_source": source,
        "license": "Apache-2.0",
        "notice_path": TIMM_NOTICE_PATH,
        "license_text_path": TIMM_LICENSE_TEXT_PATH,
    }


def _validate_library_config(config: TrainingConfig) -> None:
    if config.model not in MODEL_ALIASES:
        raise ValueError(f"unsupported model alias: {config.model}")
    if config.image_size != SMARTDOC_INPUT_SIZE:
        raise ValueError("image_size must be exactly 384")
    if config.target_size != SMARTDOC_TARGET_SIZE:
        raise ValueError("target_size must be exactly 96")
    if config.device not in {"auto", "cpu", "mps"}:
        raise ValueError("corner training supports CPU or MPS only")
    integer_values = (
        config.train_stride,
        config.validation_stride,
        config.epochs,
        config.batch_size,
        config.patience,
        config.cpu_threads,
    )
    if any(isinstance(value, bool) or value < 1 for value in integer_values):
        raise ValueError("training counts, strides, and CPU threads must be positive")
    if config.cpu_threads != 1:
        raise ValueError("cpu_threads must be exactly 1")
    if isinstance(config.workers, bool) or config.workers < 0:
        raise ValueError("workers must be non-negative")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")


def _common_metadata(
    config: TrainingConfig,
    prepared: PreparedRecords,
    fit: FitResult,
    architecture: dict[str, object],
    parameter_count: int,
    checkpoint: dict[str, object],
    device: torch.device,
    latency_ms: float,
    source_repository: dict[str, object],
    runtime_environment: dict[str, object],
) -> dict[str, object]:
    return {
        "artifact_licensing": _artifact_licensing_metadata(),
        "smartdoc": {
            "source_url": SMARTDOC_SOURCE_URL,
            "archive_sha256": prepared.archive_sha256,
            "version": SMARTDOC_VERSION,
            "license": SMARTDOC_LICENSE,
            "notice_path": SMARTDOC_NOTICE_PATH,
        },
        "dataset": {
            "background_split": {
                "train": [
                    "background01",
                    "background02",
                    "background03",
                    "background04",
                ],
                "validation": ["background05"],
            },
            "strides": {
                "train": config.train_stride,
                "validation": config.validation_stride,
            },
            "sample_counts": {
                "train": len(prepared.train_records),
                "validation": len(prepared.validation_records),
            },
            "sample_identities": {
                "train": list(prepared.train_identities),
                "validation": list(prepared.validation_identities),
            },
            "identity_sha256": prepared.identity_sha256,
        },
        "seed": config.seed,
        "augmentation": AUGMENTATION_CONTRACT,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": WEIGHT_DECAY,
        },
        "scheduler": {"name": "CosineAnnealingLR", "t_max": config.epochs},
        "loss": {
            "name": "corner_heatmap_loss",
            "corner": "focal_weighted_mse+positive_negative_balanced_bce",
            "mask": "0.25*(binary_cross_entropy_with_logits+soft_dice)",
        },
        "epochs_requested": config.epochs,
        "epochs_completed": fit.epochs_completed,
        "best_epoch": fit.best_epoch,
        "validation": {
            "normalized_corner_rmse": fit.best_validation.normalized_corner_rmse,
            "confidence_coverage": _confidence_coverage_metadata(
                fit.best_validation.detection_rate
            ),
            "mean_confidence": fit.best_validation.mean_confidence,
            "loss": dict(fit.best_validation.loss),
        },
        "architecture": architecture,
        "parameter_count": parameter_count,
        "checkpoint": checkpoint,
        "training_device": device.type,
        "reproducibility": {
            "training_config": _training_config_metadata(config),
            "resolved_device": device.type,
            "source_repository": source_repository,
            "environment": runtime_environment,
        },
        "cpu_latency": _cpu_latency_metadata(latency_ms, runtime_environment),
    }


def build_corner_manifest_record(
    report: Mapping[str, object], report_filename: str
) -> dict[str, object]:
    """Derive the compact central manifest record from a detailed report."""
    alias = report.get("model")
    if not isinstance(alias, str) or alias not in MODEL_ALIASES:
        raise ValueError("corner report must identify a supported model")
    common = {
        key: copy.deepcopy(value)
        for key, value in report.items()
        if key not in {"schema_version", "model", "history"}
    }
    dataset = common.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("corner report must contain dataset metadata")
    dataset.pop("sample_identities", None)
    checkpoint = common.get("checkpoint")
    latency = common.get("cpu_latency")
    validation = common.get("validation")
    if not isinstance(checkpoint, dict):
        raise ValueError("corner report must contain checkpoint metadata")
    if not isinstance(latency, dict):
        raise ValueError("corner report must contain CPU latency metadata")
    if not isinstance(validation, dict):
        raise ValueError("corner report must contain validation metadata")
    coverage = validation.get("confidence_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("corner report must define confidence coverage")
    report_bytes = _json_bytes(report)
    return {
        "adapter": ADAPTER_NAMES[alias],
        "availability": "locally_trained",
        "local_filename": checkpoint["filename"],
        "model_family": MODEL_FAMILIES[alias],
        "upstream": SMARTDOC_UPSTREAM,
        "source": SMARTDOC_SOURCE_URL,
        "license": SMARTDOC_LICENSE,
        "license_text_path": SMARTDOC_LICENSE_TEXT_PATH,
        "notice_path": SMARTDOC_NOTICE_PATH,
        "generic_manifest_key_scopes": {
            "license": "training_dataset",
            "license_text_path": "training_dataset",
            "notice_path": "training_dataset",
            "source": "training_dataset",
            "upstream": "training_dataset",
        },
        "required_runtime": "PyTorch CPU",
        "runtime_detection": (
            "readers must validate local_filename bytes against top-level sha256 "
            "and checkpoint_size_bytes before loading"
        ),
        "sha256": checkpoint["sha256"],
        "checkpoint_size_bytes": checkpoint["size_bytes"],
        "cpu_latency_threads": latency["protocol"]["threads"],
        "cpu_latency_ms_median": latency["median_ms"],
        "validation_normalized_corner_rmse": validation[
            "normalized_corner_rmse"
        ],
        "validation_confidence_coverage_at_0_5": coverage["rate"],
        "report": {
            "filename": report_filename,
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
            "size_bytes": len(report_bytes),
        },
        **common,
    }


def train_corner_model(
    config: TrainingConfig,
    dependencies: TrainingDependencies | None = None,
) -> dict[str, object]:
    """Train, freeze, publish, and CPU-verify one SmartDoc corner model."""
    dependencies = dependencies or TrainingDependencies()
    _validate_library_config(config)
    source_repository = dependencies.source_repository_provider()
    runtime_environment = dependencies.runtime_environment_provider()
    paths = resolve_output_paths(config)
    prepared = load_verified_splits(
        config,
        source_verifier=dependencies.source_verifier,
        record_loader=dependencies.record_loader,
    )
    seed_everything(config.seed)
    device = select_device(config.device)
    train_dataset = dependencies.dataset_factory(
        prepared.train_records, training=True, global_seed=config.seed
    )
    validation_dataset = dependencies.dataset_factory(
        prepared.validation_records, training=False, global_seed=config.seed
    )
    train_loader, validation_loader = build_loaders(
        train_dataset, validation_dataset, config
    )
    trained_model = dependencies.model_builder(config.model, pretrained=True).to(device)
    architecture = _architecture_metadata(config.model)
    fit = fit_model(
        trained_model, train_loader, validation_loader, config, device
    )

    cpu_model = dependencies.model_builder(config.model, pretrained=False).to("cpu")
    cpu_model.load_state_dict(fit.best_state_dict, strict=True)
    latency_runner = dependencies.latency_runner or cpu_latency_ms
    latency_ms = float(
        latency_runner(cpu_model, validation_dataset[0][0], config.cpu_threads)
    )
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise ValueError("CPU latency must be finite and non-negative")

    checkpoint = {
        "schema_version": 1,
        "architecture_alias": config.model,
        "model_state_dict": fit.best_state_dict,
        "preprocessing": PREPROCESSING_METADATA,
        "target": TARGET_METADATA,
        "decoder": DECODER_METADATA,
    }
    serialized_checkpoint = _checkpoint_bytes(checkpoint)
    checkpoint_metadata = {
        "filename": paths.checkpoint.name,
        "size_bytes": len(serialized_checkpoint),
        "sha256": hashlib.sha256(serialized_checkpoint).hexdigest(),
        "schema_version": checkpoint["schema_version"],
    }
    loaded_bytes = dependencies.checkpoint_bytes_loader(serialized_checkpoint)
    if loaded_bytes.get("architecture_alias") != config.model:
        raise ValueError("serialized checkpoint architecture alias mismatch")
    loaded_bytes_state = loaded_bytes.get("model_state_dict")
    if not isinstance(loaded_bytes_state, dict) or any(
        not isinstance(value, torch.Tensor) or value.device.type != "cpu"
        for value in loaded_bytes_state.values()
    ):
        raise ValueError("serialized checkpoint state_dict must load on CPU")
    cpu_model.load_state_dict(loaded_bytes_state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in cpu_model.parameters())
    common = _common_metadata(
        config,
        prepared,
        fit,
        architecture,
        parameter_count,
        checkpoint_metadata,
        device,
        latency_ms,
        source_repository,
        runtime_environment,
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "model": config.model,
        **common,
        "history": list(fit.history),
    }
    manifest_record = build_corner_manifest_record(report, paths.report.name)

    def validate_published_checkpoint(published_bytes: bytes) -> None:
        if len(published_bytes) != checkpoint_metadata["size_bytes"]:
            raise ValueError("written checkpoint size mismatch")
        if hashlib.sha256(published_bytes).hexdigest() != checkpoint_metadata["sha256"]:
            raise ValueError("written checkpoint SHA-256 mismatch")
        loaded_checkpoint = dependencies.checkpoint_bytes_loader(published_bytes)
        if loaded_checkpoint.get("schema_version") != checkpoint["schema_version"]:
            raise ValueError("written checkpoint schema version mismatch")
        if loaded_checkpoint.get("architecture_alias") != config.model:
            raise ValueError("written checkpoint architecture alias mismatch")
        loaded_state = loaded_checkpoint.get("model_state_dict")
        if not isinstance(loaded_state, dict) or any(
            not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            for value in loaded_state.values()
        ):
            raise ValueError("written checkpoint state_dict must load on CPU")
        cpu_model.load_state_dict(loaded_state, strict=True)

    publish_training_outputs(
        paths,
        checkpoint_bytes=serialized_checkpoint,
        report=report,
        manifest_record=manifest_record,
        validate_published_bytes=validate_published_checkpoint,
        write_bytes=dependencies.write_bytes,
        replace=dependencies.replace,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = train_corner_model(parse_args(sys.argv[1:] if argv is None else argv))
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
