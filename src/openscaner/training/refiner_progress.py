"""Identity-bound progress checkpoints for resumable refiner training."""

from __future__ import annotations

import dataclasses
import io
import math
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


MAX_PROGRESS_CHECKPOINT_BYTES = 256 * 1024 * 1024
PROGRESS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProgressBinding:
    source_commit: str
    cache_sha256: str
    cache_size_bytes: int
    archive_sha256: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    model_family: str


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    completed_epochs: int
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
    scheduler_state_dict: dict[str, object]
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_validation: dict[str, object]
    history: tuple[dict[str, object], ...]
    torch_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]


def _hex_digest(value: object, *, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _binding_document(binding: ProgressBinding) -> dict[str, object]:
    if not isinstance(binding, ProgressBinding):
        raise TypeError("progress binding must be a ProgressBinding")
    _hex_digest(binding.source_commit, length=40, name="source_commit")
    _hex_digest(binding.cache_sha256, length=64, name="cache_sha256")
    _hex_digest(binding.archive_sha256, length=64, name="archive_sha256")
    _validate_positive_int(binding.cache_size_bytes, name="cache_size_bytes")
    _validate_positive_int(binding.seed, name="seed")
    _validate_positive_int(binding.epochs, name="epochs")
    _validate_positive_int(binding.batch_size, name="batch_size")
    _validate_finite_float(binding.learning_rate, name="learning_rate")
    if (
        isinstance(binding.weight_decay, bool)
        or not isinstance(binding.weight_decay, (int, float))
        or not math.isfinite(float(binding.weight_decay))
        or float(binding.weight_decay) < 0.0
    ):
        raise ValueError("weight_decay must be finite and non-negative")
    if not isinstance(binding.model_family, str) or not binding.model_family:
        raise ValueError("model_family must be a non-empty string")
    return dataclasses.asdict(binding)


def _tensor_map(value: object, *, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, torch.Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"{name} values must be tensors")
        _validate_storable(item, name=f"{name}.{key}")
        result[key] = item.detach().cpu().clone()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _validate_tensor(value: torch.Tensor, *, name: str) -> None:
    if value.is_floating_point() or value.is_complex():
        finite = bool(torch.isfinite(value).all().detach().to(device="cpu").item())
        if not finite:
            raise ValueError(f"{name} tensor must contain only finite values")


def _validate_storable(value: object, *, name: str) -> object:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        _validate_tensor(value, name=name)
        return value.detach().cpu().clone()
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, tuple):
        return tuple(
            _validate_storable(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _validate_storable(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            _validate_storable_key(key, name=name): _validate_storable(
                item, name=f"{name}.{key}"
            )
            for key, item in value.items()
        }
    raise TypeError(f"{name} has unsupported checkpoint value {type(value).__name__}")


def _validate_storable_key(value: object, *, name: str) -> object:
    if isinstance(value, bool):
        raise TypeError(f"{name} mapping keys must not be bool")
    if isinstance(value, (str, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} mapping key must be finite")
        return value
    raise TypeError(f"{name} mapping keys must be strings or numbers")


def _progress_document(progress: TrainingProgress) -> dict[str, object]:
    if not isinstance(progress, TrainingProgress):
        raise TypeError("progress must be a TrainingProgress")
    completed_epochs = _validate_nonnegative_int(
        progress.completed_epochs, name="completed_epochs"
    )
    best_epoch = _validate_nonnegative_int(progress.best_epoch, name="best_epoch")
    if best_epoch > completed_epochs:
        raise ValueError("best_epoch must not exceed completed_epochs")
    if len(progress.history) != completed_epochs:
        raise ValueError("history length must equal completed_epochs")
    if not isinstance(progress.torch_rng_state, torch.Tensor):
        raise TypeError("torch_rng_state must be a tensor")
    if not isinstance(progress.cuda_rng_states, tuple) or any(
        not isinstance(item, torch.Tensor) for item in progress.cuda_rng_states
    ):
        raise TypeError("cuda_rng_states must be a tuple of tensors")
    return {
        "completed_epochs": completed_epochs,
        "model_state_dict": _tensor_map(progress.model_state_dict, name="model_state_dict"),
        "optimizer_state_dict": _validate_storable(
            progress.optimizer_state_dict,
            name="optimizer_state_dict",
        ),
        "scheduler_state_dict": _validate_storable(
            progress.scheduler_state_dict,
            name="scheduler_state_dict",
        ),
        "best_state_dict": _tensor_map(progress.best_state_dict, name="best_state_dict"),
        "best_epoch": best_epoch,
        "best_validation": _validate_storable(
            progress.best_validation,
            name="best_validation",
        ),
        "history": _validate_storable(progress.history, name="history"),
        "torch_rng_state": _validate_storable(
            progress.torch_rng_state,
            name="torch_rng_state",
        ),
        "cuda_rng_states": _validate_storable(
            progress.cuda_rng_states,
            name="cuda_rng_states",
        ),
    }


def _deserialize_progress(value: object) -> TrainingProgress:
    if not isinstance(value, Mapping):
        raise ValueError("progress payload is malformed")
    fields = {field.name for field in dataclasses.fields(TrainingProgress)}
    if set(value) != fields:
        raise ValueError("progress payload schema is invalid")
    return TrainingProgress(
        completed_epochs=_validate_nonnegative_int(
            value["completed_epochs"], name="completed_epochs"
        ),
        model_state_dict=_tensor_map(
            value["model_state_dict"], name="model_state_dict"
        ),
        optimizer_state_dict=_mapping_copy(
            value["optimizer_state_dict"], name="optimizer_state_dict"
        ),
        scheduler_state_dict=_mapping_copy(
            value["scheduler_state_dict"], name="scheduler_state_dict"
        ),
        best_state_dict=_tensor_map(value["best_state_dict"], name="best_state_dict"),
        best_epoch=_validate_nonnegative_int(value["best_epoch"], name="best_epoch"),
        best_validation=_mapping_copy(
            value["best_validation"], name="best_validation"
        ),
        history=tuple(
            _mapping_copy(item, name="history item")
            for item in _sequence_copy(value["history"], name="history")
        ),
        torch_rng_state=_require_tensor(value["torch_rng_state"], name="torch_rng_state"),
        cuda_rng_states=tuple(
            _require_tensor(item, name="cuda_rng_states")
            for item in _sequence_copy(value["cuda_rng_states"], name="cuda_rng_states")
        ),
    )


def _mapping_copy(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(_validate_storable(value, name=name))


def _sequence_copy(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(_validate_storable(value, name=name))


def _require_tensor(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    _validate_tensor(value, name=name)
    return value.detach().cpu().clone()


def _check_target_for_save(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for ancestor in reversed(absolute.parents):
        if ancestor.exists() and ancestor.is_symlink():
            raise ValueError(f"progress checkpoint parent must not be a symlink: {ancestor}")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.parent.is_symlink() or not absolute.parent.is_dir():
        raise ValueError("progress checkpoint parent must be a real directory")
    if absolute.is_symlink():
        raise ValueError("progress checkpoint must not be a symlink")
    if absolute.exists() and not absolute.is_file():
        raise ValueError("progress checkpoint must be a regular file")
    return absolute


def _open_existing_regular(path: Path) -> tuple[int, os.stat_result]:
    absolute = Path(os.path.abspath(path))
    if absolute.is_symlink():
        raise ValueError("progress checkpoint must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError("progress checkpoint could not be opened") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("progress checkpoint must be a regular file")
        if metadata.st_size < 1 or metadata.st_size > MAX_PROGRESS_CHECKPOINT_BYTES:
            raise ValueError("progress checkpoint size is invalid")
        return descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


def _checkpoint_bytes(payload: Mapping[str, object]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    data = buffer.getvalue()
    if len(data) < 1 or len(data) > MAX_PROGRESS_CHECKPOINT_BYTES:
        raise ValueError("progress checkpoint size is invalid")
    return data


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_progress_checkpoint(
    path: Path,
    binding: ProgressBinding,
    progress: TrainingProgress,
) -> None:
    """Atomically save a bounded checkpoint tied to exact training identity."""
    absolute = _check_target_for_save(path)
    payload = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "binding": _binding_document(binding),
        "progress": _progress_document(progress),
    }
    data = _checkpoint_bytes(payload)
    temporary = absolute.with_name(f".{absolute.name}.{uuid.uuid4().hex}.tmp")
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
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = temporary.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("progress checkpoint temporary must be a regular file")
        if metadata.st_size != len(data):
            raise RuntimeError("progress checkpoint temporary size changed")
        os.replace(temporary, absolute)
        _fsync_directory(absolute.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def load_progress_checkpoint(
    path: Path,
    binding: ProgressBinding,
) -> TrainingProgress:
    """Load and validate a progress checkpoint for exactly one training identity."""
    expected_binding = _binding_document(binding)
    descriptor, metadata = _open_existing_regular(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            payload = stream.read(metadata.st_size + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) != metadata.st_size:
        raise ValueError("progress checkpoint size changed while reading")
    try:
        raw = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("progress checkpoint could not be loaded") from error
    if not isinstance(raw, Mapping):
        raise ValueError("progress checkpoint must be a mapping")
    if set(raw) != {"schema_version", "binding", "progress"}:
        raise ValueError("progress checkpoint top-level keys are invalid")
    if raw["schema_version"] != PROGRESS_SCHEMA_VERSION:
        raise ValueError("progress checkpoint schema version is invalid")
    if raw["binding"] != expected_binding:
        raise ValueError("progress checkpoint binding does not match")
    progress = _deserialize_progress(raw["progress"])
    if len(progress.history) != progress.completed_epochs:
        raise ValueError("progress history length is inconsistent")
    if progress.best_epoch > progress.completed_epochs:
        raise ValueError("progress best epoch is inconsistent")
    return progress


__all__ = [
    "MAX_PROGRESS_CHECKPOINT_BYTES",
    "PROGRESS_SCHEMA_VERSION",
    "ProgressBinding",
    "TrainingProgress",
    "load_progress_checkpoint",
    "save_progress_checkpoint",
]
