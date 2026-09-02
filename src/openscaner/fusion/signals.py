"""Immutable raw signals emitted by frozen document-boundary models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from openscaner.adapters.base import AdapterUnavailable
from openscaner.fusion.artifacts import (
    FUSION_DEPENDENCY_ARTIFACTS,
    FUSION_POLICY_ARTIFACT,
    ModelArtifactIdentity,
)


def _confidence(value: float) -> float:
    confidence = float(value)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and in the range [0, 1]")
    return confidence


def _frozen_array(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.array(array, dtype=dtype, copy=True, order="C")
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values after conversion")
    frozen = np.frombuffer(converted.tobytes(order="C"), dtype=converted.dtype).reshape(
        converted.shape
    )
    return frozen


@dataclass(frozen=True, slots=True)
class DocAlignerSignals:
    """Raw DocAligner heatmaps and their deterministic centroid decode."""

    corners: np.ndarray | None
    confidence: float
    heatmaps: np.ndarray
    backend: str

    def __post_init__(self) -> None:
        heatmaps = np.asarray(self.heatmaps)
        if (
            heatmaps.ndim != 4
            or heatmaps.shape[0] != 1
            or heatmaps.shape[1] != 4
            or heatmaps.shape[2] < 1
            or heatmaps.shape[3] < 1
        ):
            raise ValueError("heatmaps must have shape 1 x 4 x H x W")
        object.__setattr__(
            self,
            "heatmaps",
            _frozen_array(
                heatmaps,
                name="heatmaps",
                shape=heatmaps.shape,
            ),
        )
        if self.corners is not None:
            object.__setattr__(
                self,
                "corners",
                _frozen_array(
                    self.corners,
                    name="corners",
                    shape=(4, 2),
                    dtype=np.float32,
                ),
            )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CornerModelSignals:
    """Decoded corners and dense outputs from a frozen corner model."""

    normalized_corners: np.ndarray
    confidence: float
    corner_heatmaps: np.ndarray
    mask_probabilities: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalized_corners",
            _frozen_array(
                self.normalized_corners,
                name="normalized_corners",
                shape=(4, 2),
                dtype=np.float32,
            ),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "corner_heatmaps",
            _frozen_array(
                self.corner_heatmaps,
                name="corner_heatmaps",
                shape=(4, 96, 96),
                dtype=np.float32,
            ),
        )
        object.__setattr__(
            self,
            "mask_probabilities",
            _frozen_array(
                self.mask_probabilities,
                name="mask_probabilities",
                shape=(96, 96),
                dtype=np.float32,
            ),
        )


@dataclass(frozen=True, slots=True)
class FusionSignals:
    """Raw signals from DocAligner and the PP-LCNet corner model."""

    docaligner: DocAlignerSignals
    corner_model: CornerModelSignals

    def __post_init__(self) -> None:
        if not isinstance(self.docaligner, DocAlignerSignals):
            raise TypeError("docaligner must be DocAlignerSignals")
        if not isinstance(self.corner_model, CornerModelSignals):
            raise TypeError("corner_model must be CornerModelSignals")


def _require_frozen_dependency(
    name: str,
    identity: ModelArtifactIdentity,
) -> None:
    if identity != FUSION_DEPENDENCY_ARTIFACTS[name]:
        raise AdapterUnavailable(
            f"{name} identity does not match frozen fusion dependency"
        )


def _validate_fusion_manifest_dependencies(model_dir) -> None:
    from openscaner.adapters import _corner_heatmap

    manifest = _corner_heatmap._read_manifest(
        model_dir,
        model_family="document-boundary fusion",
    )
    if manifest is None:
        raise AdapterUnavailable("fusion dependency manifest is absent")
    entries = manifest["models"]
    for name in FUSION_DEPENDENCY_ARTIFACTS:
        matches = [entry for entry in entries if entry.get("adapter") == name]
        if len(matches) != 1:
            raise AdapterUnavailable(
                f"{name} fusion dependency entry is absent or duplicated"
            )
        entry = matches[0]
        try:
            declared = ModelArtifactIdentity(
                adapter=name,
                filename=entry.get("local_filename"),
                sha256=entry.get("sha256"),
                size_bytes=entry.get("checkpoint_size_bytes"),
            )
        except (TypeError, ValueError) as error:
            raise AdapterUnavailable(
                f"{name} fusion dependency identity is invalid"
            ) from error
        _require_frozen_dependency(name, declared)

    fusion_matches = [
        entry
        for entry in entries
        if entry.get("adapter") == FUSION_POLICY_ARTIFACT.adapter
    ]
    calibration = (
        fusion_matches[0].get("calibration") if len(fusion_matches) == 1 else None
    )
    models = calibration.get("models") if isinstance(calibration, Mapping) else None
    expected_models = {
        name: identity.as_document()
        for name, identity in FUSION_DEPENDENCY_ARTIFACTS.items()
    }
    if models != expected_models:
        raise AdapterUnavailable("fusion calibration dependency identities are invalid")


class FusionSignalPredictor:
    """Reusable CPU runtimes bound to one verified snapshot of each model."""

    def __init__(
        self,
        *,
        docaligner_session: object,
        corner_model: object,
        cpu_threads: int,
        artifacts: Mapping[str, ModelArtifactIdentity],
    ) -> None:
        if dict(artifacts) != dict(FUSION_DEPENDENCY_ARTIFACTS):
            raise AdapterUnavailable(
                "loaded model identity does not match frozen fusion dependency"
            )
        self._docaligner_session = docaligner_session
        self._corner_model = corner_model
        self._cpu_threads = cpu_threads
        self.artifacts = MappingProxyType(dict(artifacts))

    def predict(self, image) -> FusionSignals:
        from openscaner.adapters import _corner_heatmap, docaligner

        source = _corner_heatmap._validate_source(image)
        return FusionSignals(
            docaligner=docaligner._infer_docaligner_signals(
                self._docaligner_session,
                source,
            ),
            corner_model=_corner_heatmap._infer_corner_model_signals(
                self._corner_model,
                source,
                model_family="PP-LCNet-0.5 corner heatmap model",
            ),
        )


def load_fusion_signal_predictor(model_dir, cpu_threads) -> FusionSignalPredictor:
    """Load both verified fusion models once for repeated CPU inference."""
    from openscaner.adapters import _corner_heatmap, docaligner

    threads = _corner_heatmap._validate_cpu_threads(cpu_threads)
    _validate_fusion_manifest_dependencies(model_dir)
    docaligner_bytes, docaligner_identity = docaligner._read_docaligner_artifact(
        model_dir
    )
    _require_frozen_dependency("docaligner", docaligner_identity)
    corner_bytes, corner_identity = (
        _corner_heatmap._read_corner_model_bytes_with_identity(
            model_dir,
            adapter_name="pp_lcnet_050_corner",
            architecture_alias="pp_lcnet_050",
            model_family="PP-LCNet-0.5 corner heatmap model",
            expected_identity=FUSION_DEPENDENCY_ARTIFACTS[
                "pp_lcnet_050_corner"
            ],
        )
    )
    _require_frozen_dependency("pp_lcnet_050_corner", corner_identity)
    _corner_heatmap.torch.set_num_threads(threads)
    docaligner_session = docaligner._build_docaligner_session(
        docaligner_bytes,
        threads,
    )
    corner_model = _corner_heatmap._load_model(
        corner_bytes,
        architecture_alias="pp_lcnet_050",
        model_family="PP-LCNet-0.5 corner heatmap model",
    )
    return FusionSignalPredictor(
        docaligner_session=docaligner_session,
        corner_model=corner_model,
        cpu_threads=threads,
        artifacts={
            docaligner_identity.adapter: docaligner_identity,
            corner_identity.adapter: corner_identity,
        },
    )


def predict_docaligner_signals(image, model_dir, cpu_threads) -> DocAlignerSignals:
    """Run the verified DocAligner model and return its frozen raw signals."""
    from openscaner.adapters import docaligner

    threads = docaligner._validate_cpu_threads(cpu_threads)
    source = docaligner._validate_source(image)
    return docaligner._predict_docaligner_signals(source, model_dir, threads)


def predict_corner_model_signals(
    image,
    model_dir,
    cpu_threads,
    *,
    adapter_name,
    architecture_alias,
    model_family,
) -> CornerModelSignals:
    """Run one verified corner model and return its frozen raw signals."""
    from openscaner.adapters import _corner_heatmap

    threads = _corner_heatmap._validate_cpu_threads(cpu_threads)
    source = _corner_heatmap._validate_source(image)
    return _corner_heatmap._predict_corner_model_signals(
        source,
        model_dir,
        threads,
        adapter_name=adapter_name,
        architecture_alias=architecture_alias,
        model_family=model_family,
    )


def predict_fusion_signals(image, model_dir, cpu_threads) -> FusionSignals:
    """Return DocAligner and frozen PP-LCNet-0.5 signals for one image."""
    predictor = load_fusion_signal_predictor(model_dir, cpu_threads)
    return predictor.predict(image)


__all__ = [
    "CornerModelSignals",
    "DocAlignerSignals",
    "FusionSignalPredictor",
    "FusionSignals",
    "ModelArtifactIdentity",
    "load_fusion_signal_predictor",
    "predict_corner_model_signals",
    "predict_docaligner_signals",
    "predict_fusion_signals",
]
