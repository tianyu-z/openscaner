"""Deterministic SmartDoc local-corner-refiner training and publication."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch
from skimage.metrics import structural_similarity
from torch import nn
from torch.utils.data import DataLoader, Dataset

from openscaner.adapters import docaligner
from openscaner.fusion.artifacts import ModelArtifactIdentity
from openscaner.geometry import validate_quad, warp_document
from openscaner.models.corner_heatmap import PRETRAINED_WEIGHT_SPECS
from openscaner.models.local_corner_refiner import (
    build_local_corner_refiner,
    differentiable_local_soft_argmax,
    export_refiner_onnx,
)
from openscaner.training.refiner_data import (
    PATCH_RADIUS_RATIOS,
    RefinerCache,
    RefinerExample,
    LocalCornerRefinerDataset,
    _parse_refiner_cache,
    _read_cache_payload,
    load_verified_refiner_cache,
    refiner_cache_bytes,
)
from openscaner.training.refiner_loss import local_corner_refiner_loss
from openscaner.training.refiner_progress import (
    ProgressBinding,
    TrainingProgress,
    load_progress_checkpoint,
    save_progress_checkpoint,
)
from openscaner.training.smartdoc import (
    SMARTDOC_ARCHIVE_SHA256,
    SMARTDOC_TRAIN_BACKGROUNDS,
    SMARTDOC_VALIDATION_BACKGROUND,
    SMARTDOC_VERSION,
    SmartDocRecord,
    VerifiedSmartDocSource,
    assert_protected_hashes_absent,
    load_smartdoc_records,
    split_smartdoc_records,
    validate_smartdoc_record_paths,
    validate_smartdoc_source_paths,
    verify_smartdoc_markers,
    verify_smartdoc_source,
)
from openscaner.training.train_corners import (
    OutputPaths,
    _json_bytes,
    publish_training_outputs,
)


ADAPTER_NAME = "docaligner_local_corner_refiner"
MODEL_FILENAME = "docaligner_local_corner_refiner.onnx"
REPORT_FILENAME = "docaligner_local_corner_refiner.json"
MODEL_FAMILY = "DocAligner local-corner PP-LCNet-0.5 refiner"
SMARTDOC_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
SMARTDOC_SOURCE_URL = "http://smartdoc.univ-lr.fr/"
SMARTDOC_NOTICE_PATH = "src/openscaner/third_party/SMARTDOC15_NOTICE.txt"
SMARTDOC_LICENSE_TEXT_PATH = (
    "src/openscaner/third_party/licenses/CC-BY-4.0-SmartDoc15.txt"
)
TIMM_UPSTREAM_URL = "https://github.com/huggingface/pytorch-image-models"
TIMM_NOTICE_PATH = "src/openscaner/third_party/TIMM_PRETRAINED_WEIGHTS_NOTICE.txt"
TIMM_LICENSE_TEXT_PATH = "src/openscaner/third_party/licenses/Apache-2.0.txt"
DEFAULT_SEED = 20260825
DEFAULT_EPOCHS = 16
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
ONNX_INPUT_NAME = "patches"
ONNX_INPUT_SHAPE = (4, 3, 256, 256)
ONNX_OUTPUT_CONTRACT = (
    ("corner_logits", (4, 1, 64, 64)),
    ("edge_logits", (4, 2, 64, 64)),
    ("confidence", (4, 1)),
)
_DIRECT_MANIFEST_PROVENANCE = (
    "parameter_count",
    "seed",
    "configuration",
    "optimizer",
    "scheduler",
    "epochs_requested",
    "epochs_completed",
    "selected_epoch",
    "history",
    "environment",
    "onnx",
    "loss",
    "smartdoc",
    "cache",
    "docaligner",
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    cache: Path
    dataset_root: Path
    archive: Path
    archive_sha256: str
    model_dir: Path
    artifacts_dir: Path
    manifest: Path
    seed: int = DEFAULT_SEED
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    device: str = "auto"
    workers: int = 0
    progress_checkpoint: Path | None = None
    resume: bool = False


@dataclass(frozen=True, slots=True)
class ValidationSlice:
    normalized_corner_rmse: float
    warp_ssim: float
    example_count: int
    refined_count: int
    fallback_count: int

    @property
    def complete(self) -> bool:
        return bool(
            type(self.example_count) is int
            and self.example_count > 0
            and type(self.refined_count) is int
            and type(self.fallback_count) is int
            and self.refined_count >= 0
            and self.fallback_count >= 0
            and self.refined_count + self.fallback_count == self.example_count
            and math.isfinite(self.normalized_corner_rmse)
            and self.normalized_corner_rmse >= 0.0
            and math.isfinite(self.warp_ssim)
            and -1.0 <= self.warp_ssim <= 1.0
        )


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    clean: ValidationSlice
    augmented: ValidationSlice
    aggregate: ValidationSlice
    radii: Mapping[str, ValidationSlice]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, ValidationSlice)
            for value in (self.clean, self.augmented, self.aggregate)
        ):
            raise TypeError("validation summaries must be ValidationSlice values")
        converted = dict(sorted(dict(self.radii).items()))
        if any(not isinstance(value, ValidationSlice) for value in converted.values()):
            raise TypeError("radius summaries must be ValidationSlice values")
        object.__setattr__(self, "radii", MappingProxyType(converted))

    @property
    def complete(self) -> bool:
        expected_radii = {f"{radius:.2f}" for radius in PATCH_RADIUS_RATIOS}
        return bool(
            set(self.radii) == expected_radii
            and self.clean.complete
            and self.augmented.complete
            and self.aggregate.complete
            and all(value.complete for value in self.radii.values())
            and self.aggregate.example_count
            == self.clean.example_count + self.augmented.example_count
            and self.aggregate.refined_count
            == self.clean.refined_count + self.augmented.refined_count
            and self.aggregate.fallback_count
            == self.clean.fallback_count + self.augmented.fallback_count
            and self.aggregate.example_count
            == sum(value.example_count for value in self.radii.values())
            and self.aggregate.refined_count
            == sum(value.refined_count for value in self.radii.values())
            and self.aggregate.fallback_count
            == sum(value.fallback_count for value in self.radii.values())
        )


@dataclass(frozen=True, slots=True)
class FitResult:
    selected_epoch: int
    epochs_completed: int
    best_state_dict: dict[str, torch.Tensor]
    best_validation: ValidationMetrics
    history: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class PreparedTrainingData:
    source: VerifiedSmartDocSource
    records: tuple[Any, ...]
    train_records: tuple[Any, ...]
    validation_records: tuple[Any, ...]
    cache: Any
    cache_payload: bytes


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _zero_workers(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workers must be exactly 0") from error
    if parsed != 0:
        raise argparse.ArgumentTypeError("workers must be exactly 0")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=_positive_int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        "--learning-rate", type=_positive_float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument(
        "--weight-decay", type=_nonnegative_float, default=DEFAULT_WEIGHT_DECAY
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--workers", type=_zero_workers, default=0)
    parser.add_argument("--progress-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse the public command line into an immutable configuration."""
    return TrainingConfig(**vars(_parser().parse_args(argv)))


def _validate_config(config: TrainingConfig) -> None:
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    for field_name in (
        "cache",
        "dataset_root",
        "archive",
        "model_dir",
        "artifacts_dir",
        "manifest",
    ):
        if not isinstance(getattr(config, field_name), Path):
            raise TypeError(f"{field_name} must be a pathlib.Path")
    if config.progress_checkpoint is not None and not isinstance(
        config.progress_checkpoint, Path
    ):
        raise TypeError("progress_checkpoint must be a pathlib.Path")
    if not isinstance(config.resume, bool):
        raise TypeError("resume must be a boolean")
    if config.resume and config.progress_checkpoint is None:
        raise ValueError("resume requires an existing progress checkpoint")
    if (
        config.resume
        and config.progress_checkpoint is not None
        and not config.progress_checkpoint.exists()
    ):
        raise ValueError("resume requires an existing progress checkpoint")
    if config.archive_sha256 != SMARTDOC_ARCHIVE_SHA256:
        raise ValueError("archive_sha256 must equal the pinned SmartDoc archive hash")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise TypeError("seed must be an integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (config.epochs, config.batch_size)
    ):
        raise ValueError("epochs and batch_size must be positive integers")
    for field_name in ("learning_rate", "weight_decay"):
        value = getattr(config, field_name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be a real number")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    if config.device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if isinstance(config.workers, bool) or config.workers != 0:
        raise ValueError("workers must be exactly 0 for grouped view cache correctness")


def select_device(requested: str) -> torch.device:
    """Resolve an explicitly available deterministic training device."""
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Seed every RNG and enable deterministic PyTorch execution."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def source_commit() -> str:
    """Return full HEAD only when all tracked and untracked source is clean."""
    repository = Path(__file__).resolve().parents[3]
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise RuntimeError(
                "source repository must be clean, including untracked files"
            )
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to verify source repository state") from error
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("source repository HEAD is not a full Git commit")
    return commit


def _require_source_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError("source repository HEAD is not a full Git commit")
    return value


def resolve_output_paths(config: TrainingConfig) -> OutputPaths:
    """Resolve fixed artifact names and reject aliases and path attacks."""
    _validate_config(config)
    candidates = tuple(
        Path(os.path.abspath(path))
        for path in (
            config.model_dir / MODEL_FILENAME,
            config.artifacts_dir / REPORT_FILENAME,
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
                raise ValueError(f"output parent must be a directory: {ancestor}")
    resolved = tuple(path.resolve(strict=False) for path in candidates)
    if len(set(resolved)) != len(resolved):
        raise ValueError("model, report, and manifest output paths collide")
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError("model, report, and manifest output paths collide")
    existing = [path for path in candidates if path.exists()]
    for index, first in enumerate(existing):
        for second in existing[index + 1 :]:
            if first.samefile(second):
                raise ValueError("model, report, and manifest output paths collide")

    dataset = Path(os.path.abspath(config.dataset_root)).resolve(strict=False)
    source_files = {
        Path(os.path.abspath(config.cache)).resolve(strict=False),
        Path(os.path.abspath(config.archive)).resolve(strict=False),
        (
            Path(os.path.abspath(config.model_dir)) / docaligner.MODEL_FILENAME
        ).resolve(strict=False),
    }
    for output in resolved:
        if output in source_files or output == dataset or dataset in output.parents:
            raise ValueError("training output collides with verified input data")
        if output.exists() and any(
            source.exists() and output.samefile(source) for source in source_files
        ):
            raise ValueError("training output collides with verified input data")
    return OutputPaths(*resolved)


class _IndexedDataset(Dataset[tuple[int, Any]]):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset.load_example(index)


def _collate_validation_examples(
    batch: Sequence[tuple[int, Any]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[Any, ...]]:
    """Stack patches while preserving each materialized validation example."""
    indices, examples = zip(*batch, strict=True)
    patches = torch.stack([example.patches for example in examples])
    return torch.tensor(indices), patches, examples


def build_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Use each dataset's view-grouped sampler with synchronous loading."""
    _validate_config(config)
    for name, dataset in (
        ("training", train_dataset),
        ("validation", validation_dataset),
    ):
        if not callable(getattr(dataset, "grouped_sampler", None)):
            raise TypeError(f"{name} dataset must provide grouped_sampler()")
    if not callable(getattr(validation_dataset, "load_example", None)):
        raise TypeError("validation dataset must provide load_example()")
    train_sampler = train_dataset.grouped_sampler()
    validation_sampler = validation_dataset.grouped_sampler()
    if tuple(iter(validation_sampler)) != tuple(range(len(validation_dataset))):
        raise ValueError("validation grouped sampler must be sequential")
    common = {
        "batch_size": config.batch_size,
        "num_workers": 0,
        "persistent_workers": False,
    }
    return (
        DataLoader(train_dataset, sampler=train_sampler, **common),
        DataLoader(
            _IndexedDataset(validation_dataset),
            sampler=validation_sampler,
            collate_fn=_collate_validation_examples,
            **common,
        ),
    )


def flatten_refiner_batch(
    patches: torch.Tensor,
    labels: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Flatten only the example and four-corner dimensions."""
    if not isinstance(patches, torch.Tensor):
        raise TypeError("patches must be a torch.Tensor")
    if patches.ndim != 5 or tuple(patches.shape[1:]) != (4, 3, 256, 256):
        raise ValueError("patches must have shape B x 4 x 3 x 256 x 256")
    if patches.shape[0] < 1:
        raise ValueError("patch batch must not be empty")
    if not isinstance(labels, Mapping):
        raise TypeError("labels must be a mapping")
    expected = {
        "corner_heatmap": (1, 64, 64),
        "edge_maps": (2, 64, 64),
        "corner_xy": (2,),
        "corner_valid": (),
    }
    if set(labels) != set(expected):
        raise ValueError(f"labels must contain exact keys {sorted(expected)}")
    flattened: dict[str, torch.Tensor] = {}
    batch_size = int(patches.shape[0])
    for name, trailing in expected.items():
        value = labels[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"labels['{name}'] must be a torch.Tensor")
        if tuple(value.shape) != (batch_size, 4, *trailing):
            raise ValueError(f"labels['{name}'] has invalid grouped shape")
        flattened[name] = value.reshape(batch_size * 4, *trailing)
    return patches.reshape(batch_size * 4, 3, 256, 256), flattened


def tensor_group_is_finite(values: Sequence[torch.Tensor]) -> bool:
    tensors = tuple(values)
    if not tensors:
        raise ValueError("tensor group must not be empty")
    checks = torch.stack(tuple(torch.isfinite(value).all() for value in tensors))
    return bool(checks.all().detach().to(device="cpu").item())


def detached_scalar_group(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    names = tuple(sorted(values))
    if not names:
        raise ValueError("scalar group must not be empty")
    stacked = torch.stack(tuple(values[name].detach() for name in names))
    scalars = stacked.to(device="cpu", dtype=torch.float64).tolist()
    result = {name: float(value) for name, value in zip(names, scalars, strict=True)}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("training loss component is non-finite")
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one full-precision epoch with the exact local-corner objective."""
    model.train()
    totals: dict[str, float] = {}
    patch_count = 0
    for patches, labels in loader:
        flat_patches, flat_labels = flatten_refiner_batch(patches, labels)
        flat_patches = flat_patches.to(device)
        moved_labels = {name: value.to(device) for name, value in flat_labels.items()}
        optimizer.zero_grad(set_to_none=True)
        losses = local_corner_refiner_loss(model(flat_patches), moved_labels)
        total = losses["total"]
        if not torch.isfinite(total):
            raise RuntimeError("training loss is non-finite")
        total.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("training produced no gradients")
        if not tensor_group_is_finite(gradients):
            raise RuntimeError("training produced non-finite gradients")
        optimizer.step()
        count = int(flat_patches.shape[0])
        patch_count += count
        for name, scalar in detached_scalar_group(losses).items():
            totals[name] = totals.get(name, 0.0) + scalar * count
    if patch_count == 0:
        raise ValueError("training loader must not be empty")
    return {name: total / patch_count for name, total in sorted(totals.items())}


def _fallback(example: RefinerExample) -> tuple[np.ndarray, bool]:
    return example.coarse_corners, True


def _mapped_refinement(
    decoded: torch.Tensor,
    example: RefinerExample,
) -> tuple[np.ndarray, bool]:
    if not isinstance(decoded, torch.Tensor) or decoded.shape != (4, 2):
        return _fallback(example)
    coordinates = decoded.detach().to(device="cpu").to(dtype=torch.float64).numpy()
    if (
        not np.isfinite(coordinates).all()
        or np.any(coordinates < 0.0)
        or np.any(coordinates > 1.0)
    ):
        return _fallback(example)
    patch_pixels = coordinates * np.float64(255.0)
    if (
        not np.isfinite(patch_pixels).all()
        or np.any(patch_pixels < 0.0)
        or np.any(patch_pixels > 255.0)
    ):
        return _fallback(example)
    try:
        refined = np.stack(
            [
                transform.patch_to_source_points(patch_pixels[index : index + 1])[0]
                for index, transform in enumerate(example.transforms)
            ]
        ).astype(np.float32)
    except (TypeError, ValueError, FloatingPointError):
        return _fallback(example)
    height, width = example.source_bgr.shape[:2]
    if (
        not np.isfinite(refined).all()
        or np.any(refined[:, 0] < 0.0)
        or np.any(refined[:, 0] >= width)
        or np.any(refined[:, 1] < 0.0)
        or np.any(refined[:, 1] >= height)
    ):
        return _fallback(example)
    valid, _ = validate_quad(
        refined,
        example.source_bgr.shape,
        min_area_ratio=0.0,
        reorder=False,
    )
    if not valid:
        return _fallback(example)
    return refined, False


def refine_or_fallback(
    corner_logits: torch.Tensor,
    example: RefinerExample,
    *,
    decoder_fn: Callable[[torch.Tensor], torch.Tensor] = differentiable_local_soft_argmax,
) -> tuple[np.ndarray, bool]:
    """Decode four corners or return the exact cached coarse quadrilateral."""
    try:
        decoded = decoder_fn(corner_logits)
    except (TypeError, ValueError, RuntimeError, FloatingPointError):
        return _fallback(example)
    return _mapped_refinement(decoded, example)


@dataclass(slots=True)
class _MetricAccumulator:
    squared_error: float = 0.0
    corner_count: int = 0
    ssim_total: float = 0.0
    example_count: int = 0
    refined_count: int = 0
    fallback_count: int = 0

    def add(
        self,
        prediction: np.ndarray,
        example: RefinerExample,
        *,
        used_fallback: bool,
    ) -> None:
        height, width = example.source_bgr.shape[:2]
        difference = (
            np.asarray(prediction, dtype=np.float64)
            - np.asarray(example.ground_truth_corners, dtype=np.float64)
        )
        diagonal_squared = float(width * width + height * height)
        self.squared_error += float(
            np.sum(difference * difference, dtype=np.float64) / diagonal_squared
        )
        self.corner_count += int(difference.shape[0])

        predicted_warp = warp_document(example.source_bgr, prediction)
        truth_warp = warp_document(
            example.source_bgr, example.ground_truth_corners
        )
        predicted_gray = cv2.cvtColor(predicted_warp, cv2.COLOR_BGR2GRAY)
        truth_gray = cv2.cvtColor(truth_warp, cv2.COLOR_BGR2GRAY)
        predicted_gray = cv2.resize(
            predicted_gray,
            (truth_gray.shape[1], truth_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        score = float(
            structural_similarity(
                truth_gray,
                predicted_gray,
                data_range=255,
            )
        )
        if not math.isfinite(score):
            raise RuntimeError("validation warp SSIM is non-finite")
        self.ssim_total += score
        self.example_count += 1
        self.fallback_count += int(used_fallback)
        self.refined_count += int(not used_fallback)

    def finish(self) -> ValidationSlice:
        if self.example_count < 1 or self.corner_count < 1:
            return ValidationSlice(float("nan"), float("nan"), 0, 0, 0)
        return ValidationSlice(
            normalized_corner_rmse=math.sqrt(
                self.squared_error / self.corner_count
            ),
            warp_ssim=self.ssim_total / self.example_count,
            example_count=self.example_count,
            refined_count=self.refined_count,
            fallback_count=self.fallback_count,
        )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    dataset: Any,
    device: torch.device,
    *,
    decoder_fn: Callable[[torch.Tensor], torch.Tensor] = differentiable_local_soft_argmax,
) -> ValidationMetrics:
    """Evaluate every clean/augmented/radius view with full fallback coverage.

    Normalized corner RMSE is the square root of the mean pixel Euclidean
    corner squared error divided by each source image's diagonal squared.
    """
    if not callable(getattr(dataset, "load_example", None)):
        raise TypeError("validation dataset must provide load_example()")
    model.eval()
    accumulators = {
        "aggregate": _MetricAccumulator(),
        "clean": _MetricAccumulator(),
        "augmented": _MetricAccumulator(),
        **{
            f"radius:{radius:.2f}": _MetricAccumulator()
            for radius in PATCH_RADIUS_RATIOS
        },
    }
    with torch.inference_mode():
        for indices, patches, examples in loader:
            if (
                not isinstance(patches, torch.Tensor)
                or patches.ndim != 5
                or tuple(patches.shape[1:]) != (4, 3, 256, 256)
            ):
                raise ValueError("validation patches must have shape B x 4 x 3 x 256 x 256")
            index_values = [int(value) for value in torch.as_tensor(indices).tolist()]
            if len(index_values) != patches.shape[0] or len(examples) != patches.shape[0]:
                raise RuntimeError("validation batch metadata and patch counts differ")
            predictions = model(patches.flatten(0, 1).to(device))
            if not isinstance(predictions, Mapping) or "corner_logits" not in predictions:
                raise ValueError("model must return corner_logits")
            corner_logits = predictions["corner_logits"]
            expected_shape = (patches.shape[0] * 4, 1, 64, 64)
            if (
                not isinstance(corner_logits, torch.Tensor)
                or tuple(corner_logits.shape) != expected_shape
            ):
                raise RuntimeError(
                    "validation model must return "
                    f"{expected_shape[0]} corner heatmaps"
                )
            for batch_index, example in enumerate(examples):
                prediction, used_fallback = refine_or_fallback(
                    corner_logits[batch_index * 4 : (batch_index + 1) * 4],
                    example,
                    decoder_fn=decoder_fn,
                )
                radius_key = f"radius:{example.radius_ratio:.2f}"
                if example.view not in {"clean", "augmented"}:
                    raise ValueError("validation view must be clean or augmented")
                if radius_key not in accumulators:
                    raise ValueError("validation radius is outside the production set")
                for key in ("aggregate", example.view, radius_key):
                    accumulators[key].add(
                        prediction,
                        example,
                        used_fallback=used_fallback,
                    )
    result = ValidationMetrics(
        clean=accumulators["clean"].finish(),
        augmented=accumulators["augmented"].finish(),
        aggregate=accumulators["aggregate"].finish(),
        radii={
            f"{radius:.2f}": accumulators[f"radius:{radius:.2f}"].finish()
            for radius in PATCH_RADIUS_RATIOS
        },
    )
    if not result.complete or result.aggregate.example_count != len(dataset):
        raise RuntimeError("validation did not produce finite full coverage")
    return result


def checkpoint_key(metrics: ValidationMetrics, epoch: int) -> tuple[float, float, int]:
    """Rank finite full-coverage checkpoints by RMSE, SSIM, then epoch."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if not isinstance(metrics, ValidationMetrics):
        raise TypeError("metrics must be ValidationMetrics")
    if not metrics.complete:
        return (float("inf"), float("inf"), sys.maxsize)
    return (
        metrics.aggregate.normalized_corner_rmse,
        -metrics.aggregate.warp_ssim,
        epoch,
    )


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _slice_document(value: ValidationSlice) -> dict[str, object]:
    return {
        "normalized_corner_rmse": value.normalized_corner_rmse,
        "warp_ssim": value.warp_ssim,
        "example_count": value.example_count,
        "refined_count": value.refined_count,
        "fallback_count": value.fallback_count,
        "coverage": 1.0 if value.complete else 0.0,
    }


def _metrics_document(value: ValidationMetrics) -> dict[str, object]:
    return {
        "clean": _slice_document(value.clean),
        "augmented": _slice_document(value.augmented),
        "aggregate": _slice_document(value.aggregate),
        "radii": {
            name: _slice_document(item) for name, item in value.radii.items()
        },
        "coordinate_rmse_definition": (
            "sqrt(sum(||predicted_corner-ground_truth_corner||^2/"
            "(source_width^2+source_height^2))/corner_count) with ordered roles"
        ),
    }


def _validation_slice_from_document(value: object) -> ValidationSlice:
    if not isinstance(value, Mapping):
        raise ValueError("validation slice must be a mapping")
    try:
        return ValidationSlice(
            normalized_corner_rmse=float(value["normalized_corner_rmse"]),
            warp_ssim=float(value["warp_ssim"]),
            example_count=int(value["example_count"]),
            refined_count=int(value["refined_count"]),
            fallback_count=int(value["fallback_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("validation slice is malformed") from error


def _validation_metrics_from_document(value: object) -> ValidationMetrics:
    if not isinstance(value, Mapping):
        raise ValueError("validation metrics must be a mapping")
    radii = value.get("radii")
    if not isinstance(radii, Mapping):
        raise ValueError("validation radii must be a mapping")
    metrics = ValidationMetrics(
        clean=_validation_slice_from_document(value.get("clean")),
        augmented=_validation_slice_from_document(value.get("augmented")),
        aggregate=_validation_slice_from_document(value.get("aggregate")),
        radii={
            str(name): _validation_slice_from_document(item)
            for name, item in radii.items()
        },
    )
    if not metrics.complete:
        raise ValueError("validation metrics are incomplete")
    return metrics


def _progress_binding(
    config: TrainingConfig,
    prepared: PreparedTrainingData,
    *,
    source_commit_value: str,
) -> ProgressBinding:
    return ProgressBinding(
        source_commit=_require_source_commit(source_commit_value),
        cache_sha256=hashlib.sha256(prepared.cache_payload).hexdigest(),
        cache_size_bytes=len(prepared.cache_payload),
        archive_sha256=prepared.source.archive_sha256,
        seed=config.seed,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        model_family=MODEL_FAMILY,
    )


def _cuda_rng_states() -> tuple[torch.Tensor, ...]:
    if not torch.cuda.is_available():
        return ()
    return tuple(state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all())


def _restore_rng_state(progress: TrainingProgress) -> None:
    torch.set_rng_state(progress.torch_rng_state.detach().cpu())
    if progress.cuda_rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.detach().cpu() for state in progress.cuda_rng_states]
        )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    validation_dataset: Any,
    config: TrainingConfig,
    device: torch.device,
    *,
    train_epoch_fn: Callable[..., dict[str, float]] = train_one_epoch,
    validation_fn: Callable[..., ValidationMetrics] = evaluate_model,
    progress_binding: ProgressBinding | None = None,
) -> FitResult:
    """Fit every requested epoch and retain the deterministic best CPU state."""
    _validate_config(config)
    if config.progress_checkpoint is None and progress_binding is not None:
        raise ValueError("progress binding requires a progress checkpoint")
    if config.progress_checkpoint is not None and progress_binding is None:
        raise ValueError("progress checkpoint requires a progress binding")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    best_key = (float("inf"), float("inf"), sys.maxsize)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: ValidationMetrics | None = None
    best_epoch = 0
    history: list[dict[str, object]] = []
    completed_epochs = 0
    if config.resume:
        if config.progress_checkpoint is None or progress_binding is None:
            raise ValueError("resume requires a progress checkpoint and binding")
        progress = load_progress_checkpoint(config.progress_checkpoint, progress_binding)
        model.load_state_dict(progress.model_state_dict, strict=True)
        optimizer.load_state_dict(progress.optimizer_state_dict)
        scheduler.load_state_dict(progress.scheduler_state_dict)
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in progress.best_state_dict.items()
        }
        best_epoch = progress.best_epoch
        best_validation = _validation_metrics_from_document(progress.best_validation)
        history = [dict(item) for item in progress.history]
        completed_epochs = progress.completed_epochs
        if completed_epochs > config.epochs:
            raise ValueError("progress checkpoint completed more epochs than requested")
        best_key = checkpoint_key(best_validation, best_epoch)
        _restore_rng_state(progress)
    sampler = train_loader.sampler
    if not callable(getattr(sampler, "set_epoch", None)):
        raise TypeError("training grouped sampler must provide set_epoch()")
    for epoch_index in range(completed_epochs, config.epochs):
        sampler.set_epoch(epoch_index)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_epoch_fn(model, train_loader, optimizer, device)
        validation = validation_fn(
            model,
            validation_loader,
            validation_dataset,
            device,
        )
        epoch = epoch_index + 1
        key = checkpoint_key(validation, epoch)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": dict(sorted(train_metrics.items())),
                "validation": _metrics_document(validation),
            }
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_validation = validation
            best_state = _cpu_state_dict(model)
        scheduler.step()
        if config.progress_checkpoint is not None and progress_binding is not None:
            save_progress_checkpoint(
                config.progress_checkpoint,
                progress_binding,
                TrainingProgress(
                    completed_epochs=epoch,
                    model_state_dict=_cpu_state_dict(model),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    best_state_dict={
                        name: value.detach().cpu().clone()
                        for name, value in (best_state or {}).items()
                    },
                    best_epoch=best_epoch,
                    best_validation=_metrics_document(best_validation),
                    history=tuple(history),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_states=_cuda_rng_states(),
                ),
            )
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "epochs_requested": config.epochs,
                    "selected_epoch": best_epoch,
                    "learning_rate": learning_rate,
                    "train": dict(sorted(train_metrics.items())),
                    "validation": _metrics_document(validation),
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
    if best_state is None or best_validation is None or not best_validation.complete:
        raise RuntimeError("training produced no selectable validation checkpoint")
    return FitResult(
        selected_epoch=best_epoch,
        epochs_completed=config.epochs,
        best_state_dict=best_state,
        best_validation=best_validation,
        history=tuple(history),
    )


def _configuration_document(config: TrainingConfig) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        result[item.name] = str(value) if isinstance(value, Path) else value
    return result


def _record_identity(record: Any) -> str:
    return f"{record.background}/{record.sequence}/{record.frame_index}"


def _cache_counts(cache: Any) -> dict[str, object]:
    records = tuple(cache.records)
    by_split: dict[str, int] = {"train": 0, "validation": 0}
    by_view: dict[str, int] = {"clean": 0, "augmented": 0}
    exclusions: dict[str, int] = {}
    exclusion_splits: dict[str, int] = {}
    exclusion_views: dict[str, int] = {}
    usable = 0
    for record in records:
        by_split[record.split] = by_split.get(record.split, 0) + 1
        by_view[record.view] = by_view.get(record.view, 0) + 1
        if record.coarse_corners is None:
            reason = str(record.exclusion_reason)
            exclusions[reason] = exclusions.get(reason, 0) + 1
            exclusion_splits[record.split] = (
                exclusion_splits.get(record.split, 0) + 1
            )
            exclusion_views[record.view] = exclusion_views.get(record.view, 0) + 1
        else:
            usable += 1
    return {
        "total_views": len(records),
        "usable_views": usable,
        "excluded_views": len(records) - usable,
        "by_split": dict(sorted(by_split.items())),
        "by_view": dict(sorted(by_view.items())),
        "exclusions_by_reason": dict(sorted(exclusions.items())),
        "exclusions": {
            "total": len(records) - usable,
            "by_reason": dict(sorted(exclusions.items())),
            "by_split": dict(sorted(exclusion_splits.items())),
            "by_view": dict(sorted(exclusion_views.items())),
        },
    }


def _reproducible_command(config: TrainingConfig) -> str:
    values = [
        "uv",
        "run",
        "--group",
        "ml",
        "python",
        "-m",
        "openscaner.training.train_corner_refiner",
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
        "--artifacts-dir",
        str(config.artifacts_dir),
        "--manifest",
        str(config.manifest),
        "--seed",
        str(config.seed),
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--device",
        config.device,
        "--workers",
        str(config.workers),
    ]
    if config.progress_checkpoint is not None:
        values.extend(("--progress-checkpoint", str(config.progress_checkpoint)))
    if config.resume:
        values.append("--resume")
    return shlex.join(values)


def runtime_environment() -> dict[str, object]:
    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "huggingface-hub": version("huggingface-hub"),
            "numpy": np.__version__,
            "onnx": version("onnx"),
            "onnxruntime": ort.__version__,
            "opencv-python": cv2.__version__,
            "scikit-image": version("scikit-image"),
            "timm": timm.__version__,
            "torch": torch.__version__,
            "torchvision": version("torchvision"),
        },
    }


def build_training_report(
    config: TrainingConfig,
    prepared: PreparedTrainingData,
    fit: FitResult,
    paths: OutputPaths,
    *,
    onnx_payload: bytes,
    parameter_count: int,
    source_commit_value: str,
    environment: Mapping[str, object],
) -> dict[str, object]:
    """Build the complete canonical finite training report."""
    _validate_config(config)
    source_sha = _require_source_commit(source_commit_value)
    if not isinstance(onnx_payload, bytes) or not onnx_payload:
        raise ValueError("ONNX payload must be non-empty bytes")
    if isinstance(parameter_count, bool) or parameter_count < 1:
        raise ValueError("parameter_count must be positive")
    cache_sha = hashlib.sha256(prepared.cache_payload).hexdigest()
    pretrained = copy.deepcopy(PRETRAINED_WEIGHT_SPECS["pp_lcnet_050"])
    docaligner_identity = prepared.cache.model_identity
    report: dict[str, object] = {
        "schema_version": 1,
        "adapter": ADAPTER_NAME,
        "model_family": MODEL_FAMILY,
        "source_commit": source_sha,
        "reproducible_command": _reproducible_command(config),
        "configuration": _configuration_document(config),
        "cache": {
            "filename": config.cache.name,
            "path": str(config.cache),
            "size_bytes": len(prepared.cache_payload),
            "sha256": cache_sha,
        },
        "docaligner": {
            "adapter": docaligner_identity.adapter,
            **docaligner_identity.as_document(),
        },
        "smartdoc": {
            "name": "SmartDoc 2015",
            "version": SMARTDOC_VERSION,
            "archive_sha256": prepared.source.archive_sha256,
            "archive_path": str(config.archive),
            "source_url": SMARTDOC_SOURCE_URL,
            "license": SMARTDOC_LICENSE,
            "notice_path": SMARTDOC_NOTICE_PATH,
            "license_text_path": SMARTDOC_LICENSE_TEXT_PATH,
            "stride": prepared.cache.stride,
            "splits": {
                "train": sorted(SMARTDOC_TRAIN_BACKGROUNDS),
                "validation": [SMARTDOC_VALIDATION_BACKGROUND],
            },
            "record_counts": {
                "all": len(prepared.records),
                "train": len(prepared.train_records),
                "validation": len(prepared.validation_records),
            },
            "record_identities": {
                "train": [_record_identity(item) for item in prepared.train_records],
                "validation": [
                    _record_identity(item) for item in prepared.validation_records
                ],
            },
            "cache_counts": _cache_counts(prepared.cache),
        },
        "seed": config.seed,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        "scheduler": {"name": "CosineAnnealingLR", "t_max": config.epochs},
        "loss": {
            "name": "local_corner_refiner_loss",
            "corner_heatmaps": "focal_weighted_mse+positive_negative_balanced_bce",
            "corner_heatmap_weight": 1.0,
            "residual": "valid-corner smooth_l1 coordinate mean",
            "residual_weight": 0.5,
            "edges": "mean(positive_negative_balanced_bce+soft_dice)",
            "edge_weight": 0.25,
        },
        "epochs_requested": config.epochs,
        "epochs_completed": fit.epochs_completed,
        "selected_epoch": fit.selected_epoch,
        "history": list(fit.history),
        "validation": _metrics_document(fit.best_validation),
        "architecture": {
            "name": "LocalCornerRefiner",
            "backbone": "PP-LCNet-0.5",
            "library": "timm",
            "pretrained": True,
            "pretrained_source": pretrained,
            "upstream": TIMM_UPSTREAM_URL,
            "notice_path": TIMM_NOTICE_PATH,
            "license_text_path": TIMM_LICENSE_TEXT_PATH,
        },
        "parameter_count": parameter_count,
        "onnx": {
            "filename": paths.checkpoint.name,
            "path": str(paths.checkpoint),
            "size_bytes": len(onnx_payload),
            "sha256": hashlib.sha256(onnx_payload).hexdigest(),
            "static_batch_size": 4,
            "input": {
                "name": ONNX_INPUT_NAME,
                "shape": list(ONNX_INPUT_SHAPE),
                "dtype": "float32",
                "color_space": "RGB",
                "pixel_scale": "uint8_to_float32_0_1_then_imagenet_normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "outputs": [
                {"name": name, "shape": list(shape), "dtype": "float32"}
                for name, shape in ONNX_OUTPUT_CONTRACT
            ],
        },
        "training_device": select_device(config.device).type,
        "environment": copy.deepcopy(dict(environment)),
    }
    _json_bytes(report)
    return report


def _manifest_provenance(report: Mapping[str, object]) -> dict[str, object]:
    if report.get("schema_version") != 1 or report.get("adapter") != ADAPTER_NAME:
        raise ValueError("training report schema or adapter mismatch")
    missing = [key for key in _DIRECT_MANIFEST_PROVENANCE if key not in report]
    if missing:
        raise ValueError(f"training report is missing provenance fields: {missing}")
    for key in (
        "configuration",
        "optimizer",
        "scheduler",
        "environment",
        "onnx",
        "loss",
        "smartdoc",
        "cache",
        "docaligner",
    ):
        if not isinstance(report[key], Mapping):
            raise ValueError(f"training report provenance field {key} must be an object")
    for key in (
        "parameter_count",
        "seed",
        "epochs_requested",
        "epochs_completed",
        "selected_epoch",
    ):
        if isinstance(report[key], bool) or not isinstance(report[key], int):
            raise ValueError(f"training report provenance field {key} must be an integer")
    if report["parameter_count"] < 1:
        raise ValueError("training report parameter_count must be positive")
    requested = int(report["epochs_requested"])
    completed = int(report["epochs_completed"])
    selected = int(report["selected_epoch"])
    history = report["history"]
    if (
        requested < 1
        or completed < 1
        or completed > requested
        or selected < 1
        or selected > completed
        or not isinstance(history, list)
        or len(history) != completed
        or not all(isinstance(item, Mapping) for item in history)
    ):
        raise ValueError("training report epoch provenance is inconsistent")
    _json_bytes(report)
    return {
        key: copy.deepcopy(report[key]) for key in _DIRECT_MANIFEST_PROVENANCE
    }


def _report_bound_manifest_fields(report: Mapping[str, object]) -> dict[str, object]:
    """Derive every manifest field whose canonical value comes from the report."""
    fields = _manifest_provenance(report)
    onnx = fields["onnx"]
    validation = report.get("validation")
    architecture = report.get("architecture")
    smartdoc = fields["smartdoc"]
    cache = fields["cache"]
    docaligner_identity = fields["docaligner"]
    if not all(
        isinstance(value, Mapping)
        for value in (onnx, validation, architecture, smartdoc, cache, docaligner_identity)
    ):
        raise ValueError("training report is missing required identities")
    pretrained_source = architecture.get("pretrained_source")
    if not isinstance(pretrained_source, Mapping):
        raise ValueError("training report is missing pretrained source")
    if not isinstance(validation.get("aggregate"), Mapping):
        raise ValueError("training report is missing aggregate validation metrics")
    fields.update(
        {
            "architecture": copy.deepcopy(dict(architecture)),
            "pretrained_source": copy.deepcopy(dict(pretrained_source)),
            "dataset_provenance": {
                "smartdoc_archive_sha256": smartdoc["archive_sha256"],
                "smartdoc_version": smartdoc["version"],
                "stride": smartdoc["stride"],
                "cache_filename": cache["filename"],
                "cache_size_bytes": cache["size_bytes"],
                "cache_sha256": cache["sha256"],
                "docaligner": copy.deepcopy(dict(docaligner_identity)),
            },
            "selected_validation": copy.deepcopy(dict(validation)),
            "source_commit": report["source_commit"],
            "reproducible_command": report["reproducible_command"],
        }
    )
    return fields


def build_manifest_record(
    report: Mapping[str, object],
    report_payload: bytes,
    onnx_payload: bytes,
    paths: OutputPaths,
) -> dict[str, object]:
    """Derive the schema-3 locally-trained model entry from exact bytes."""
    report_bound_fields = _report_bound_manifest_fields(report)
    onnx = report_bound_fields["onnx"]
    if not isinstance(onnx, Mapping):
        raise ValueError("training report is missing ONNX identity")
    model_sha = hashlib.sha256(onnx_payload).hexdigest()
    if onnx.get("sha256") != model_sha or onnx.get("size_bytes") != len(onnx_payload):
        raise ValueError("training report ONNX identity differs from bytes")
    return {
        "adapter": ADAPTER_NAME,
        "availability": "locally_trained",
        "model_family": MODEL_FAMILY,
        "local_filename": paths.checkpoint.name,
        "checkpoint_size_bytes": len(onnx_payload),
        "sha256": model_sha,
        "required_runtime": "ONNX Runtime CPU",
        "runtime_detection": (
            "verify exact byte size and SHA-256, require CPUExecutionProvider, "
            "and validate the static batch-four tensor contract"
        ),
        **report_bound_fields,
        "training_report": {
            "filename": paths.report.name,
            "path": str(paths.report),
            "size_bytes": len(report_payload),
            "sha256": hashlib.sha256(report_payload).hexdigest(),
        },
        "source": SMARTDOC_SOURCE_URL,
        "upstream": TIMM_UPSTREAM_URL,
        "license": f"Apache-2.0; training data {SMARTDOC_LICENSE}",
        "notice_path": TIMM_NOTICE_PATH,
        "license_text_path": TIMM_LICENSE_TEXT_PATH,
    }


def validate_refiner_onnx_bytes(payload: bytes) -> None:
    """Load exact bytes on CPU and verify the static production contract."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("ONNX payload must be non-empty bytes")
    try:
        session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    except Exception as error:
        raise RuntimeError("published refiner ONNX failed CPU loading") from error
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("published refiner ONNX must use CPUExecutionProvider only")
    actual_inputs = [
        (item.name, tuple(item.shape), item.type) for item in session.get_inputs()
    ]
    expected_inputs = [(ONNX_INPUT_NAME, ONNX_INPUT_SHAPE, "tensor(float)")]
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"published refiner ONNX input contract mismatch: {actual_inputs}")
    actual_outputs = [
        (item.name, tuple(item.shape), item.type) for item in session.get_outputs()
    ]
    expected_outputs = [
        (name, shape, "tensor(float)") for name, shape in ONNX_OUTPUT_CONTRACT
    ]
    if actual_outputs != expected_outputs:
        raise RuntimeError(f"published refiner ONNX output contract mismatch: {actual_outputs}")
    fixed_input = np.linspace(
        -1.0,
        1.0,
        num=int(np.prod(ONNX_INPUT_SHAPE)),
        dtype=np.float32,
    ).reshape(ONNX_INPUT_SHAPE)
    try:
        outputs = session.run(None, {ONNX_INPUT_NAME: fixed_input})
    except Exception as error:
        raise RuntimeError("published refiner ONNX CPU execution failed") from error
    if len(outputs) != len(ONNX_OUTPUT_CONTRACT):
        raise RuntimeError("published refiner ONNX returned the wrong output count")
    for output, (_, shape) in zip(outputs, ONNX_OUTPUT_CONTRACT, strict=True):
        if (
            output.shape != shape
            or not np.issubdtype(output.dtype, np.floating)
            or not np.isfinite(output).all()
        ):
            raise RuntimeError("published refiner ONNX returned invalid output")


def _validate_committed_outputs(
    payloads: Mapping[Path, bytes],
    *,
    paths: OutputPaths,
    onnx_payload: bytes,
    report_payload: bytes,
    manifest_record: Mapping[str, object],
) -> None:
    if payloads.get(paths.checkpoint) != onnx_payload:
        raise RuntimeError("published ONNX bytes changed")
    if payloads.get(paths.report) != report_payload:
        raise RuntimeError("published training report bytes changed")
    try:
        report = json.loads(report_payload)
        manifest = json.loads(payloads[paths.manifest])
    except (KeyError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("published training JSON failed validation") from error
    if report.get("adapter") != ADAPTER_NAME:
        raise RuntimeError("published training report adapter changed")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 3
        or not isinstance(manifest.get("models"), list)
        or not all(isinstance(item, dict) for item in manifest["models"])
    ):
        raise RuntimeError("published manifest schema is invalid")
    matches = [
        item for item in manifest["models"] if item.get("adapter") == ADAPTER_NAME
    ]
    if len(matches) != 1 or matches[0] != manifest_record:
        raise RuntimeError("published refiner manifest entry failed cross-check")
    try:
        expected_provenance = _report_bound_manifest_fields(report)
    except ValueError as error:
        raise RuntimeError("published training report provenance is invalid") from error
    for field, expected in expected_provenance.items():
        if manifest_record.get(field) != expected:
            raise RuntimeError(
                f"published manifest provenance field {field} differs from report"
            )
    onnx = report.get("onnx", {})
    training_report = manifest_record.get("training_report", {})
    if (
        onnx.get("sha256") != hashlib.sha256(onnx_payload).hexdigest()
        or onnx.get("size_bytes") != len(onnx_payload)
        or training_report.get("sha256") != hashlib.sha256(report_payload).hexdigest()
        or training_report.get("size_bytes") != len(report_payload)
    ):
        raise RuntimeError("published report and manifest identities differ")


def publish_refiner_outputs(
    paths: OutputPaths,
    onnx_payload: bytes,
    report: dict[str, object],
    manifest_record: dict[str, object],
    *,
    onnx_validator: Callable[[bytes], None] = validate_refiner_onnx_bytes,
    phase_hook: Callable[[str, Path], None] | None = None,
) -> None:
    """Atomically publish model/report/manifest with staged ONNX verification."""
    report_payload = _json_bytes(report)
    expected_sha = hashlib.sha256(onnx_payload).hexdigest()
    expected_size = len(onnx_payload)

    def validate_published_bytes(payload: bytes) -> None:
        if len(payload) != expected_size:
            raise RuntimeError("staged ONNX size differs from exported bytes")
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise RuntimeError("staged ONNX SHA-256 differs from exported bytes")
        onnx_validator(payload)

    def validate_committed(payloads: Mapping[Path, bytes]) -> None:
        _validate_committed_outputs(
            payloads,
            paths=paths,
            onnx_payload=onnx_payload,
            report_payload=report_payload,
            manifest_record=manifest_record,
        )

    publish_training_outputs(
        paths,
        checkpoint_bytes=onnx_payload,
        report=report,
        manifest_record=manifest_record,
        validate_published_bytes=validate_published_bytes,
        validate_committed=validate_committed,
        phase_hook=phase_hook,
    )


def _load_cache_metadata(path: Path) -> RefinerCache:
    return _parse_refiner_cache(_read_cache_payload(path))


def _load_docaligner_identity(model_dir: Path) -> ModelArtifactIdentity:
    _, identity = docaligner._read_docaligner_artifact(model_dir)
    return identity


def _dataset_factory(
    cache: RefinerCache,
    records: Sequence[SmartDocRecord],
    split: str,
    seed: int,
) -> LocalCornerRefinerDataset:
    return LocalCornerRefinerDataset(cache, records, split, seed)


def _model_builder(*, pretrained: bool) -> nn.Module:
    return build_local_corner_refiner(pretrained=pretrained)


@dataclass(frozen=True, slots=True)
class TrainingDependencies:
    source_path_validator: Callable[[Path, Path], None] = validate_smartdoc_source_paths
    marker_verifier: Callable[[Path], None] = verify_smartdoc_markers
    source_verifier: Callable[[Path, Path], VerifiedSmartDocSource] = verify_smartdoc_source
    record_loader: Callable[..., tuple[SmartDocRecord, ...]] = load_smartdoc_records
    record_path_validator: Callable[[Sequence[SmartDocRecord]], None] = (
        validate_smartdoc_record_paths
    )
    cache_metadata_loader: Callable[[Path], RefinerCache] = _load_cache_metadata
    docaligner_identity_loader: Callable[[Path], ModelArtifactIdentity] = (
        _load_docaligner_identity
    )
    cache_loader: Callable[..., RefinerCache] = load_verified_refiner_cache
    cache_payload_reader: Callable[[Path], bytes] = _read_cache_payload
    cache_serializer: Callable[[RefinerCache], bytes] = refiner_cache_bytes
    dataset_factory: Callable[..., Dataset] = _dataset_factory
    model_builder: Callable[..., nn.Module] = _model_builder
    fit_runner: Callable[..., FitResult] | None = None
    onnx_exporter: Callable[[nn.Module], bytes] = export_refiner_onnx
    onnx_validator: Callable[[bytes], None] = validate_refiner_onnx_bytes
    source_commit_provider: Callable[[], str] = source_commit
    environment_provider: Callable[[], dict[str, object]] = runtime_environment
    prepared_data_loader: Callable[[TrainingConfig], PreparedTrainingData] | None = None
    publisher: Callable[..., None] = publish_refiner_outputs


def prepare_training_data(
    config: TrainingConfig,
    dependencies: TrainingDependencies | None = None,
) -> PreparedTrainingData:
    """Verify pinned source bytes, stride-selected records, and cache replay."""
    _validate_config(config)
    dependencies = dependencies or TrainingDependencies()
    dependencies.source_path_validator(config.dataset_root, config.archive)
    dependencies.source_path_validator(
        config.cache,
        config.model_dir / docaligner.MODEL_FILENAME,
    )
    dependencies.marker_verifier(config.dataset_root)
    source = dependencies.source_verifier(config.archive, config.dataset_root)
    if source.archive_sha256 != config.archive_sha256:
        raise ValueError("verified SmartDoc archive hash differs from supplied hash")
    assert_protected_hashes_absent(
        (source.archive_sha256, *(item.sha256 for item in source.files.values()))
    )

    metadata = dependencies.cache_metadata_loader(config.cache)
    if metadata.smartdoc_archive_sha256 != source.archive_sha256:
        raise ValueError("refiner cache archive identity differs from verified source")
    if metadata.smartdoc_version != SMARTDOC_VERSION:
        raise ValueError("refiner cache SmartDoc version mismatch")
    if metadata.seed != config.seed:
        raise ValueError("refiner cache seed differs from training seed")
    docaligner_identity = dependencies.docaligner_identity_loader(config.model_dir)
    if docaligner_identity != metadata.model_identity:
        raise ValueError("refiner cache DocAligner identity differs from verified model")
    assert_protected_hashes_absent((docaligner_identity.sha256,))

    records = tuple(
        dependencies.record_loader(
            config.dataset_root,
            stride=metadata.stride,
            verified_files=source.files,
        )
    )
    dependencies.record_path_validator(records)
    if any(record.frame_index % metadata.stride != 0 for record in records):
        raise ValueError("SmartDoc records do not match refiner cache stride")
    train_records, validation_records = split_smartdoc_records(records, stride=1)
    if not train_records or not validation_records:
        raise ValueError("SmartDoc train and validation records must both be non-empty")
    train_backgrounds = {record.background for record in train_records}
    if train_backgrounds != SMARTDOC_TRAIN_BACKGROUNDS:
        raise ValueError(
            "training records must contain all SmartDoc backgrounds01-04 only"
        )
    if any(
        record.background != SMARTDOC_VALIDATION_BACKGROUND
        for record in validation_records
    ):
        raise ValueError("validation records must use SmartDoc background05 only")
    assert_protected_hashes_absent(
        record.image_sha256
        for record in records
        if getattr(record, "image_sha256", None) is not None
    )
    cache = dependencies.cache_loader(
        config.cache,
        records,
        smartdoc_archive_sha256=source.archive_sha256,
        smartdoc_version=SMARTDOC_VERSION,
        stride=metadata.stride,
        seed=config.seed,
        model_identity=docaligner_identity,
    )
    cache_payload = dependencies.cache_payload_reader(config.cache)
    if not isinstance(cache_payload, bytes) or not cache_payload:
        raise ValueError("refiner cache payload must be non-empty bytes")
    if cache_payload != dependencies.cache_serializer(cache):
        raise ValueError("refiner cache bytes changed after verified loading")
    assert_protected_hashes_absent((hashlib.sha256(cache_payload).hexdigest(),))
    return PreparedTrainingData(
        source=source,
        records=records,
        train_records=train_records,
        validation_records=validation_records,
        cache=cache,
        cache_payload=cache_payload,
    )


def train_corner_refiner(
    config: TrainingConfig,
    dependencies: TrainingDependencies | None = None,
) -> dict[str, object]:
    """Train, validate, export, verify, and atomically publish the refiner."""
    _validate_config(config)
    dependencies = dependencies or TrainingDependencies()
    initial_commit = _require_source_commit(dependencies.source_commit_provider())
    paths = resolve_output_paths(config)
    prepared = (
        dependencies.prepared_data_loader(config)
        if dependencies.prepared_data_loader is not None
        else prepare_training_data(config, dependencies)
    )
    seed_everything(config.seed)
    device = select_device(config.device)
    train_dataset = dependencies.dataset_factory(
        prepared.cache, prepared.records, "train", config.seed
    )
    validation_dataset = dependencies.dataset_factory(
        prepared.cache, prepared.records, "validation", config.seed
    )
    if len(train_dataset) < 1 or len(validation_dataset) < 1:
        raise ValueError("refiner train and validation datasets must both be non-empty")
    train_loader, validation_loader = build_loaders(
        train_dataset, validation_dataset, config
    )
    trained_model = dependencies.model_builder(pretrained=True).to(device)
    runner = dependencies.fit_runner or fit_model
    if config.progress_checkpoint is None:
        fit = runner(
            trained_model,
            train_loader,
            validation_loader,
            validation_dataset,
            config,
            device,
        )
    else:
        fit = runner(
            trained_model,
            train_loader,
            validation_loader,
            validation_dataset,
            config,
            device,
            progress_binding=_progress_binding(
                config,
                prepared,
                source_commit_value=initial_commit,
            ),
        )
    cpu_model = dependencies.model_builder(pretrained=False).to("cpu")
    cpu_model.load_state_dict(fit.best_state_dict, strict=True)
    cpu_model.eval()
    onnx_payload = dependencies.onnx_exporter(cpu_model)
    if not isinstance(onnx_payload, bytes) or not onnx_payload:
        raise ValueError("ONNX exporter must return non-empty bytes")
    parameter_count = sum(parameter.numel() for parameter in cpu_model.parameters())
    report = build_training_report(
        config,
        prepared,
        fit,
        paths,
        onnx_payload=onnx_payload,
        parameter_count=parameter_count,
        source_commit_value=initial_commit,
        environment=dependencies.environment_provider(),
    )
    report_payload = _json_bytes(report)
    manifest_record = build_manifest_record(
        report,
        report_payload,
        onnx_payload,
        paths,
    )
    final_commit = _require_source_commit(dependencies.source_commit_provider())
    if final_commit != initial_commit:
        raise RuntimeError("source repository commit changed during fit/export")
    dependencies.publisher(
        paths,
        onnx_payload,
        report,
        manifest_record,
        onnx_validator=dependencies.onnx_validator,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = train_corner_refiner(parse_args(sys.argv[1:] if argv is None else argv))
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_NAME",
    "MODEL_FILENAME",
    "REPORT_FILENAME",
    "FitResult",
    "PreparedTrainingData",
    "TrainingConfig",
    "TrainingDependencies",
    "ValidationMetrics",
    "ValidationSlice",
    "build_loaders",
    "build_manifest_record",
    "build_training_report",
    "checkpoint_key",
    "evaluate_model",
    "fit_model",
    "flatten_refiner_batch",
    "main",
    "parse_args",
    "prepare_training_data",
    "publish_refiner_outputs",
    "refine_or_fallback",
    "resolve_output_paths",
    "seed_everything",
    "select_device",
    "source_commit",
    "tensor_group_is_finite",
    "train_corner_refiner",
    "train_one_epoch",
    "validate_refiner_onnx_bytes",
]
