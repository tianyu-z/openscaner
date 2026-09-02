"""Shared CPU inference support for trained corner-heatmap adapters."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import stat

import cv2
import numpy as np
import torch

from openscaner.adapters._model_integrity import verified_model_bytes
from openscaner.adapters.base import AdapterOutput, AdapterUnavailable
from openscaner.fusion.artifacts import ModelArtifactIdentity
from openscaner.fusion.signals import CornerModelSignals
from openscaner.models.corner_heatmap import (
    BACKBONE_ALIASES,
    build_model,
    decode_corner_heatmaps,
)


_MANIFEST_FILENAME = "manifest.json"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_PREPROCESSING_METADATA = {
    "input_size": [384, 384],
    "color_space": "RGB",
    "pixel_scale": "uint8_to_float32_0_1",
    "resize_interpolation": "opencv_INTER_AREA",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
_TARGET_METADATA = {
    "size": [96, 96],
    "channel_order": ["TL", "TR", "BR", "BL"],
    "corner_coordinates": "normalized_xy_0_1",
    "auxiliary_channel": "document_mask_training_only",
}
_DECODER_METADATA = {
    "name": "local_soft_argmax",
    "local_soft_argmax_radius": 2,
    "coordinates": "normalized_xy_0_1",
    "confidence": "mean_sigmoid_global_peak_logit",
    "detection_threshold": 0.5,
}
_CHECKPOINT_FIELDS = {
    "schema_version",
    "architecture_alias",
    "model_state_dict",
    "preprocessing",
    "target",
    "decoder",
}
_SIGNAL_DTYPES = frozenset({torch.float16, torch.float32, torch.float64})


class _NonFiniteCornerHeatmaps(AdapterUnavailable):
    """Signal that preserves the standalone adapter's no-detection behavior."""


def _read_manifest(model_dir: str | Path, *, model_family: str) -> dict[str, object] | None:
    directory = Path(model_dir)
    manifest_path = directory / _MANIFEST_FILENAME
    if directory.is_symlink():
        raise AdapterUnavailable(f"{model_family} model directory must not be a symlink")
    if manifest_path.is_symlink():
        raise AdapterUnavailable(f"{model_family} manifest must not be a symlink")
    if not manifest_path.exists():
        return None

    descriptor = None
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(manifest_path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AdapterUnavailable(f"{model_family} manifest is not a regular file")
        if metadata.st_size > _MAX_MANIFEST_BYTES:
            raise AdapterUnavailable(f"{model_family} manifest is too large")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(_MAX_MANIFEST_BYTES + 1)
        if len(payload) > _MAX_MANIFEST_BYTES:
            raise AdapterUnavailable(f"{model_family} manifest is too large")
    except AdapterUnavailable:
        raise
    except OSError as error:
        raise AdapterUnavailable(f"{model_family} manifest could not be read") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)

    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterUnavailable(f"{model_family} manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 3
    ):
        raise AdapterUnavailable(f"{model_family} manifest schema is incompatible")
    models = manifest.get("models")
    if not isinstance(models, list) or not all(isinstance(entry, dict) for entry in models):
        raise AdapterUnavailable(f"{model_family} manifest model entries are invalid")
    return manifest


def _find_manifest_entry(
    manifest: dict[str, object],
    *,
    adapter_name: str,
    architecture_alias: str,
    model_family: str,
) -> dict[str, object] | None:
    models = manifest["models"]
    assert isinstance(models, list)
    matches = [entry for entry in models if entry.get("adapter") == adapter_name]
    if not matches:
        return None
    if len(matches) != 1:
        raise AdapterUnavailable(f"{model_family} manifest entry is duplicated")
    entry = matches[0]
    architecture = entry.get("architecture")
    if (
        not isinstance(architecture, dict)
        or architecture.get("alias") != architecture_alias
        or architecture.get("timm_backbone") != BACKBONE_ALIASES[architecture_alias]
    ):
        raise AdapterUnavailable(f"{model_family} manifest architecture is incompatible")
    return entry


def _checkpoint_spec(
    entry: dict[str, object],
    *,
    expected_filename: str,
    model_family: str,
) -> tuple[str, str, int]:
    checkpoint = entry.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or type(checkpoint.get("schema_version")) is not int
        or checkpoint["schema_version"] != 1
    ):
        raise AdapterUnavailable(f"{model_family} manifest checkpoint schema is incompatible")
    filename = entry.get("local_filename")
    digest = entry.get("sha256")
    size_bytes = entry.get("checkpoint_size_bytes")
    if (
        not isinstance(filename, str)
        or not filename
        or filename != expected_filename
        or Path(filename).name != filename
        or filename in {".", ".."}
        or checkpoint.get("filename") != filename
    ):
        raise AdapterUnavailable(f"{model_family} manifest checkpoint filename is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or checkpoint.get("sha256") != digest
    ):
        raise AdapterUnavailable(f"{model_family} manifest checkpoint SHA-256 is invalid")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 1
        or checkpoint.get("size_bytes") != size_bytes
    ):
        raise AdapterUnavailable(f"{model_family} manifest checkpoint size is invalid")
    if size_bytes > _MAX_CHECKPOINT_BYTES:
        raise AdapterUnavailable(f"{model_family} manifest checkpoint is too large")
    return filename, digest, size_bytes


def _load_model(
    checkpoint_bytes: bytes,
    *,
    architecture_alias: str,
    model_family: str,
):
    try:
        checkpoint = torch.load(
            BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise AdapterUnavailable(
            f"{model_family} checkpoint is incompatible: could not be deserialized"
        ) from error

    try:
        if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_FIELDS:
            raise ValueError("checkpoint fields do not match schema version 1")
        if (
            type(checkpoint["schema_version"]) is not int
            or checkpoint["schema_version"] != 1
        ):
            raise ValueError("checkpoint schema version is incompatible")
        if checkpoint["architecture_alias"] != architecture_alias:
            raise ValueError("checkpoint architecture does not match adapter")
        if checkpoint["preprocessing"] != _PREPROCESSING_METADATA:
            raise ValueError("checkpoint preprocessing metadata is incompatible")
        if checkpoint["target"] != _TARGET_METADATA:
            raise ValueError("checkpoint target metadata is incompatible")
        if checkpoint["decoder"] != _DECODER_METADATA:
            raise ValueError("checkpoint decoder metadata is incompatible")
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict) or not all(
            isinstance(key, str) for key in state_dict
        ):
            raise TypeError("checkpoint model state must be a mapping with string keys")
    except Exception as error:
        raise AdapterUnavailable(
            f"{model_family} checkpoint metadata is incompatible"
        ) from error

    model = build_model(architecture_alias, pretrained=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise AdapterUnavailable(
            f"{model_family} checkpoint is incompatible with its architecture"
        ) from error
    return model


def _preprocess(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_AREA)
    normalized = rgb.astype(np.float32) / 255.0
    normalized = (
        normalized - np.asarray(_PREPROCESSING_METADATA["mean"], dtype=np.float32)
    ) / np.asarray(_PREPROCESSING_METADATA["std"], dtype=np.float32)
    chw = np.transpose(normalized, (2, 0, 1)).copy()
    return torch.from_numpy(chw).unsqueeze(0)


def _validate_model_order_quad(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[bool, str]:
    """Validate a convex clockwise TL, TR, BR, BL quadrilateral."""
    quad = np.asarray(corners, dtype=np.float64)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return False, "invalid_shape"

    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        return False, "invalid_image"
    if (
        (quad[:, 0] < 0).any()
        or (quad[:, 0] >= width).any()
        or (quad[:, 1] < 0).any()
        or (quad[:, 1] >= height).any()
    ):
        return False, "out_of_bounds"

    edges = np.roll(quad, -1, axis=0) - quad
    cross_products = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    cross_epsilon = np.finfo(np.float64).eps * max(width, height) ** 2 * 32.0
    if np.any(cross_products <= cross_epsilon):
        reason = (
            "reversed_winding"
            if np.all(cross_products < -cross_epsilon)
            else "not_convex"
        )
        return False, reason

    next_quad = np.roll(quad, -1, axis=0)
    signed_area = 0.5 * float(
        np.sum(quad[:, 0] * next_quad[:, 1] - quad[:, 1] * next_quad[:, 0])
    )
    if signed_area / float(width * height) < 0.01:
        return False, "too_small"

    tl, tr, br, bl = quad
    center = quad.mean(axis=0)
    top_midpoint = 0.5 * (tl + tr)
    bottom_midpoint = 0.5 * (bl + br)
    left_midpoint = 0.5 * (tl + bl)
    right_midpoint = 0.5 * (tr + br)
    semantic_epsilon = np.finfo(np.float64).eps * max(width, height) * 32.0
    if not (
        top_midpoint[1] < center[1] - semantic_epsilon
        and bottom_midpoint[1] > center[1] + semantic_epsilon
        and left_midpoint[0] < center[0] - semantic_epsilon
        and right_midpoint[0] > center[0] + semantic_epsilon
    ):
        return False, "invalid_semantic_order"
    return True, "ok"


def _model_prediction(model, image: np.ndarray, *, model_family: str) -> dict:
    model.to("cpu")
    model.eval()
    tensor = _preprocess(image)
    with torch.inference_mode():
        prediction = model(tensor)
    if not isinstance(prediction, dict) or set(prediction) != {
        "corner_heatmaps",
        "mask_logits",
    }:
        raise AdapterUnavailable(f"{model_family} returned an incompatible output")
    return prediction


def _decode_heatmaps(
    heatmaps: torch.Tensor,
    *,
    model_family: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        normalized_corners, confidence = decode_corner_heatmaps(heatmaps)
    except (TypeError, ValueError, RuntimeError) as error:
        raise AdapterUnavailable(f"{model_family} corner decoding failed") from error
    return normalized_corners, confidence


def _infer_corner_model_signals(
    model,
    image: np.ndarray,
    *,
    model_family: str,
) -> CornerModelSignals:
    prediction = _model_prediction(model, image, model_family=model_family)
    heatmaps = prediction["corner_heatmaps"]
    mask_logits = prediction["mask_logits"]
    if not isinstance(heatmaps, torch.Tensor) or heatmaps.shape != (1, 4, 96, 96):
        raise AdapterUnavailable(
            f"{model_family} returned incompatible corner heatmaps"
        )
    if not isinstance(mask_logits, torch.Tensor) or mask_logits.shape != (
        1,
        1,
        96,
        96,
    ):
        raise AdapterUnavailable(f"{model_family} returned incompatible mask logits")
    for name, tensor in (
        ("corner heatmaps", heatmaps),
        ("mask logits", mask_logits),
    ):
        if tensor.device.type != "cpu":
            raise AdapterUnavailable(f"{model_family} {name} must be CPU tensors")
        if not tensor.is_floating_point():
            raise AdapterUnavailable(
                f"{model_family} {name} must use a floating-point dtype"
            )
        if tensor.dtype not in _SIGNAL_DTYPES:
            raise AdapterUnavailable(
                f"{model_family} {name} uses an unsupported dtype"
            )
        if not torch.isfinite(tensor).all():
            error_type = (
                _NonFiniteCornerHeatmaps
                if name == "corner heatmaps"
                else AdapterUnavailable
            )
            raise error_type(
                f"{model_family} {name} must contain only finite values"
            )

    normalized_corners, confidence = _decode_heatmaps(
        heatmaps,
        model_family=model_family,
    )
    if (
        not isinstance(normalized_corners, torch.Tensor)
        or normalized_corners.shape != (1, 4, 2)
        or normalized_corners.device.type != "cpu"
        or not normalized_corners.is_floating_point()
    ):
        raise AdapterUnavailable(
            f"{model_family} corner decoder returned incompatible coordinates"
        )
    if normalized_corners.dtype not in _SIGNAL_DTYPES:
        raise AdapterUnavailable(
            f"{model_family} corner decoder coordinates use an unsupported dtype"
        )
    if not torch.isfinite(normalized_corners).all():
        raise AdapterUnavailable(
            f"{model_family} corner decoder returned incompatible coordinates"
        )
    if (
        not isinstance(confidence, torch.Tensor)
        or confidence.shape != (1,)
        or confidence.device.type != "cpu"
        or not confidence.is_floating_point()
    ):
        raise AdapterUnavailable(
            f"{model_family} corner decoder returned incompatible confidence"
        )
    if confidence.dtype not in _SIGNAL_DTYPES:
        raise AdapterUnavailable(
            f"{model_family} corner decoder confidence uses an unsupported dtype"
        )
    if not torch.isfinite(confidence).all():
        raise AdapterUnavailable(
            f"{model_family} corner decoder returned incompatible confidence"
        )
    model_confidence = float(confidence[0].item())
    mask_probabilities = torch.sigmoid(mask_logits).to(dtype=torch.float32)
    if not torch.isfinite(mask_probabilities).all():
        raise AdapterUnavailable(
            f"{model_family} mask probabilities must contain only finite values"
        )
    return CornerModelSignals(
        normalized_corners=normalized_corners[0].detach().numpy(),
        confidence=model_confidence,
        corner_heatmaps=heatmaps[0].detach().numpy(),
        mask_probabilities=mask_probabilities[0, 0].detach().numpy(),
    )


def _adapter_output_from_signals(
    signals: CornerModelSignals,
    image: np.ndarray,
) -> AdapterOutput:
    model_confidence = signals.confidence
    threshold = float(_DECODER_METADATA["detection_threshold"])
    if model_confidence < threshold:
        return AdapterOutput(
            corners=None,
            confidence=0.0,
            backend="torch:cpu",
            diagnostics={
                "reason": "mean_confidence_below_threshold",
                "input_size": 384,
                "model_confidence": model_confidence,
                "detection_threshold": threshold,
            },
        )
    height, width = image.shape[:2]
    source_scale = np.asarray(
        [width - 1, height - 1],
        dtype=signals.normalized_corners.dtype,
    )
    corners = (signals.normalized_corners * source_scale).astype(np.float32)
    valid, reason = _validate_model_order_quad(corners, image.shape)
    if not valid:
        return AdapterOutput(
            corners=None,
            confidence=0.0,
            backend="torch:cpu",
            diagnostics={"reason": reason, "input_size": 384},
        )
    return AdapterOutput(
        corners=corners,
        confidence=model_confidence,
        backend="torch:cpu",
        diagnostics={"input_size": 384},
    )


def _validate_cpu_threads(cpu_threads) -> int:
    if (
        isinstance(cpu_threads, bool)
        or not isinstance(cpu_threads, int)
        or cpu_threads < 1
    ):
        raise ValueError("cpu_threads must be a positive integer")
    return cpu_threads


def _validate_source(image) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        raise ValueError("image must be a non-empty three-channel image")
    if source.dtype != np.uint8:
        raise ValueError("image must use uint8 values")
    return source


def _load_corner_model_artifact(
    model_dir,
    *,
    adapter_name: str,
    architecture_alias: str,
    model_family: str,
    checkpoint_optional: bool,
    expected_identity: ModelArtifactIdentity | None = None,
) -> tuple[object, ModelArtifactIdentity] | None:
    loaded = _read_corner_model_bytes_with_identity(
        model_dir,
        adapter_name=adapter_name,
        architecture_alias=architecture_alias,
        model_family=model_family,
        checkpoint_optional=checkpoint_optional,
        expected_identity=expected_identity,
    )
    if loaded is None:
        return None
    checkpoint_bytes, loaded_identity = loaded
    model = _load_model(
        checkpoint_bytes,
        architecture_alias=architecture_alias,
        model_family=model_family,
    )
    return model, loaded_identity


def _read_corner_model_bytes_with_identity(
    model_dir,
    *,
    adapter_name: str,
    architecture_alias: str,
    model_family: str,
    checkpoint_optional: bool = False,
    expected_identity: ModelArtifactIdentity | None = None,
) -> tuple[bytes, ModelArtifactIdentity] | None:
    manifest = _read_manifest(model_dir, model_family=model_family)
    entry = (
        None
        if manifest is None
        else _find_manifest_entry(
            manifest,
            adapter_name=adapter_name,
            architecture_alias=architecture_alias,
            model_family=model_family,
        )
    )
    if entry is None:
        if checkpoint_optional:
            return None
        raise AdapterUnavailable(f"{model_family} manifest entry is absent")

    filename, expected_sha256, expected_size_bytes = _checkpoint_spec(
        entry,
        expected_filename=f"{adapter_name}.pth",
        model_family=model_family,
    )
    declared_identity = ModelArtifactIdentity(
        adapter=adapter_name,
        filename=filename,
        sha256=expected_sha256,
        size_bytes=expected_size_bytes,
    )
    if expected_identity is not None and declared_identity != expected_identity:
        raise AdapterUnavailable(
            f"{model_family} identity does not match frozen fusion dependency"
        )
    checkpoint_bytes = verified_model_bytes(
        model_dir,
        filename=filename,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        model_family=model_family,
    )
    loaded_identity = ModelArtifactIdentity(
        adapter=adapter_name,
        filename=filename,
        sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        size_bytes=len(checkpoint_bytes),
    )
    if expected_identity is not None and loaded_identity != expected_identity:
        raise AdapterUnavailable(
            f"{model_family} bytes do not match frozen fusion dependency"
        )
    return checkpoint_bytes, loaded_identity


def _load_corner_model_with_identity(
    model_dir,
    *,
    adapter_name: str,
    architecture_alias: str,
    model_family: str,
    expected_identity: ModelArtifactIdentity | None = None,
) -> tuple[object, ModelArtifactIdentity]:
    loaded = _load_corner_model_artifact(
        model_dir,
        adapter_name=adapter_name,
        architecture_alias=architecture_alias,
        model_family=model_family,
        checkpoint_optional=False,
        expected_identity=expected_identity,
    )
    if loaded is None:
        raise AdapterUnavailable(f"{model_family} required checkpoint is unavailable")
    return loaded


def _load_adapter_model(
    model_dir,
    *,
    adapter_name: str,
    architecture_alias: str,
    model_family: str,
    checkpoint_optional: bool,
):
    loaded = _load_corner_model_artifact(
        model_dir,
        adapter_name=adapter_name,
        architecture_alias=architecture_alias,
        model_family=model_family,
        checkpoint_optional=checkpoint_optional,
    )
    return None if loaded is None else loaded[0]


def _predict_corner_model_signals(
    source: np.ndarray,
    model_dir,
    cpu_threads: int,
    *,
    adapter_name: str,
    architecture_alias: str,
    model_family: str,
) -> CornerModelSignals:
    torch.set_num_threads(cpu_threads)
    model = _load_adapter_model(
        model_dir,
        adapter_name=adapter_name,
        architecture_alias=architecture_alias,
        model_family=model_family,
        checkpoint_optional=False,
    )
    if model is None:
        raise AdapterUnavailable(f"{model_family} required checkpoint is unavailable")
    return _infer_corner_model_signals(model, source, model_family=model_family)


def run_corner_heatmap(
    image,
    model_dir,
    cpu_threads,
    *,
    adapter_name: str,
    architecture_alias: str,
    model_family: str,
    checkpoint_optional: bool,
) -> AdapterOutput:
    threads = _validate_cpu_threads(cpu_threads)
    torch.set_num_threads(threads)
    source = _validate_source(image)
    model = _load_adapter_model(
        model_dir,
        adapter_name=adapter_name,
        architecture_alias=architecture_alias,
        model_family=model_family,
        checkpoint_optional=checkpoint_optional,
    )
    if model is None:
        return AdapterOutput(
            corners=None,
            confidence=0.0,
            backend="torch:cpu",
            diagnostics={"reason": "checkpoint_not_available"},
        )
    try:
        signals = _infer_corner_model_signals(
            model,
            source,
            model_family=model_family,
        )
    except _NonFiniteCornerHeatmaps:
        return AdapterOutput(
            corners=None,
            confidence=0.0,
            backend="torch:cpu",
            diagnostics={"reason": "non_finite_heatmaps", "input_size": 384},
        )
    return _adapter_output_from_signals(signals, source)
