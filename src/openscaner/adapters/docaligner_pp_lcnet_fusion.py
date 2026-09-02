"""CPU-only DocAligner and PP-LCNet document-boundary fusion adapter."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from openscaner.adapters.base import AdapterOutput, AdapterUnavailable
from openscaner.fusion.artifacts import FUSION_POLICY_ARTIFACT
from openscaner.fusion.candidates import FusionCandidate, generate_candidates
from openscaner.fusion.policy import (
    FusionFeatures,
    ScoredCandidate,
    load_verified_policy,
    select_candidate,
)
from openscaner.fusion.signals import (
    CornerModelSignals,
    DocAlignerSignals,
    FusionSignals,
    predict_fusion_signals,
)
from openscaner.geometry import validate_quad


_ADAPTER_NAME = FUSION_POLICY_ARTIFACT.adapter


def _not_detected(policy_sha: str, reason: str) -> AdapterOutput:
    return AdapterOutput(
        corners=None,
        confidence=0.0,
        backend="torch:cpu",
        diagnostics={"policy_sha256": policy_sha, "reason": reason},
    )


def _validate_source(image) -> np.ndarray:
    source = np.asarray(image)
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or source.size == 0
        or min(source.shape[:2]) < 2
    ):
        raise ValueError("image must be a non-empty three-channel image")
    if source.dtype != np.uint8:
        raise ValueError("image must use uint8 values")
    return source


def _finite_array(
    value: object,
    *,
    shape: tuple[int, ...] | None = None,
    leading_shape: tuple[int, ...] | None = None,
) -> bool:
    if not isinstance(value, np.ndarray) or not np.issubdtype(
        value.dtype, np.floating
    ):
        return False
    if shape is not None and value.shape != shape:
        return False
    if leading_shape is not None and (
        value.ndim != len(leading_shape) + 2
        or value.shape[: len(leading_shape)] != leading_shape
        or min(value.shape[-2:]) < 1
    ):
        return False
    return bool(np.isfinite(value).all())


def _valid_signals(signals: object) -> bool:
    if not isinstance(signals, FusionSignals):
        return False
    docaligner = getattr(signals, "docaligner", None)
    corner_model = getattr(signals, "corner_model", None)
    if not isinstance(docaligner, DocAlignerSignals) or not isinstance(
        corner_model, CornerModelSignals
    ):
        return False
    for confidence in (
        getattr(docaligner, "confidence", None),
        getattr(corner_model, "confidence", None),
    ):
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float, np.integer, np.floating))
            or not np.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            return False
    backend = getattr(docaligner, "backend", None)
    if not isinstance(backend, str) or not backend:
        return False
    coarse = getattr(docaligner, "corners", None)
    if coarse is not None and not _finite_array(coarse, shape=(4, 2)):
        return False
    if not _finite_array(
        getattr(docaligner, "heatmaps", None),
        leading_shape=(1, 4),
    ):
        return False
    for value, shape in (
        (getattr(corner_model, "normalized_corners", None), (4, 2)),
        (getattr(corner_model, "corner_heatmaps", None), (4, 96, 96)),
        (getattr(corner_model, "mask_probabilities", None), (96, 96)),
    ):
        if not _finite_array(value, shape=shape):
            return False
    return True


def _valid_selection(selected: object, image_shape: tuple[int, ...]) -> bool:
    if not isinstance(selected, ScoredCandidate):
        return False
    candidate = getattr(selected, "candidate", None)
    features = getattr(selected, "features", None)
    fallback_used = getattr(selected, "fallback_used", None)
    score = getattr(selected, "score", None)
    if (
        not isinstance(candidate, FusionCandidate)
        or not isinstance(features, FusionFeatures)
        or type(fallback_used) is not bool
        or isinstance(score, bool)
        or not isinstance(score, (int, float, np.integer, np.floating))
        or not np.isfinite(float(score))
    ):
        return False
    if (
        getattr(candidate, "family", None)
        not in {"docaligner", "corner_heatmap", "mask", "hybrid"}
        or isinstance(getattr(candidate, "source_index", None), bool)
        or not isinstance(getattr(candidate, "source_index", None), int)
        or candidate.source_index < 0
    ):
        return False
    for name in (
        "mask_agreement",
        "corner_response",
        "edge_support",
        "coarse_agreement",
        "stability",
    ):
        value = getattr(features, name, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            return False
    corners = getattr(candidate, "corners", None)
    if (
        not isinstance(corners, np.ndarray)
        or corners.shape != (4, 2)
        or corners.dtype != np.float32
        or not np.isfinite(corners).all()
    ):
        return False
    valid_corners, _ = validate_quad(corners, image_shape, reorder=False)
    return bool(valid_corners)


def run(image, model_dir, cpu_threads):
    source = _validate_source(image)
    policy, policy_sha = load_verified_policy(
        model_dir,
        adapter_name=_ADAPTER_NAME,
    )
    if policy_sha != FUSION_POLICY_ARTIFACT.sha256:
        raise AdapterUnavailable("fusion policy checksum does not match frozen policy")
    signals = predict_fusion_signals(source, model_dir, cpu_threads)
    if not _valid_signals(signals):
        return _not_detected(policy_sha, "invalid_fusion_data")
    candidates = generate_candidates(
        source,
        signals.docaligner,
        signals.corner_model,
        maximum_candidates=policy.maximum_candidates,
        maximum_corner_displacement_ratio=(
            policy.maximum_corner_displacement_ratio
        ),
    )
    selected = select_candidate(candidates, source, signals, policy)
    if selected is None:
        return _not_detected(policy_sha, "no_valid_candidate")
    if not _valid_selection(selected, source.shape):
        return _not_detected(policy_sha, "invalid_fusion_data")
    return AdapterOutput(
        corners=selected.candidate.corners,
        confidence=float(np.clip(selected.score, 0.0, 1.0)),
        backend="torch:cpu",
        diagnostics={
            "policy_sha256": policy_sha,
            "selected_family": selected.candidate.family,
            "features": asdict(selected.features),
            "fallback_used": selected.fallback_used,
        },
    )
