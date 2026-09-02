"""Verified prompted-SAM policy and deterministic source-only selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np

from openscaner.adapters._model_integrity import verified_model_bytes
from openscaner.adapters.base import AdapterUnavailable
from openscaner.geometry import order_quad, validate_quad
from openscaner.prompted_sam.artifacts import PROMPTED_SAM_POLICY_ARTIFACT


SCORE_FIELDS = (
    "predicted_iou",
    "mask_stability",
    "edge_support",
    "coarse_agreement",
    "geometric_stability",
)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "crop_margin_ratio",
        "foreground_scale",
        "include_background",
        "minimum_sam_score",
        "weights",
        "fallback",
    }
)


def _bounded_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and in [0, 1]") from error
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _immutable_quad(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (4, 2)
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite 4x2 quadrilateral")
    corners = order_quad(array).astype(np.float32)
    corners.setflags(write=False)
    return corners


@dataclass(frozen=True, slots=True)
class PromptedSamPolicy:
    schema_version: int
    crop_margin_ratio: float
    foreground_scale: float
    include_background: bool
    minimum_sam_score: float
    weights: Mapping[str, float]
    fallback: Literal["docaligner"]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != PROMPTED_SAM_POLICY_ARTIFACT.schema_version
        ):
            raise ValueError(
                "policy schema_version must be exactly "
                f"{PROMPTED_SAM_POLICY_ARTIFACT.schema_version}"
            )
        margin = _bounded_float(self.crop_margin_ratio, name="crop_margin_ratio")
        scale = _bounded_float(self.foreground_scale, name="foreground_scale")
        if not 0.0 < scale < 1.0:
            raise ValueError("foreground_scale must be strictly between zero and one")
        if type(self.include_background) is not bool:
            raise TypeError("include_background must be boolean")
        minimum = _bounded_float(self.minimum_sam_score, name="minimum_sam_score")
        if not isinstance(self.weights, Mapping) or set(self.weights) != set(SCORE_FIELDS):
            raise ValueError(f"weights must contain exactly {sorted(SCORE_FIELDS)}")
        weights = {
            name: _bounded_float(self.weights[name], name=f"weights[{name}]")
            for name in SCORE_FIELDS
        }
        if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("weights must sum to 1 within 1e-6")
        if self.fallback != "docaligner":
            raise ValueError("fallback must be exactly 'docaligner'")
        object.__setattr__(self, "crop_margin_ratio", margin)
        object.__setattr__(self, "foreground_scale", scale)
        object.__setattr__(self, "minimum_sam_score", minimum)
        object.__setattr__(self, "weights", MappingProxyType(weights))


@dataclass(frozen=True, slots=True)
class PromptedSamFeatures:
    predicted_iou: float
    mask_stability: float
    edge_support: float
    coarse_agreement: float
    geometric_stability: float

    def __post_init__(self) -> None:
        for name in SCORE_FIELDS:
            object.__setattr__(
                self,
                name,
                _bounded_float(getattr(self, name), name=name),
            )


@dataclass(frozen=True, slots=True)
class PromptedSamCandidate:
    corners: np.ndarray
    family: str
    mask_index: int
    predicted_iou: float
    mask_stability: float
    edge_support: float
    geometric_stability: float

    def __post_init__(self) -> None:
        corners = _immutable_quad(self.corners, name="corners")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be a non-empty string")
        if type(self.mask_index) is not int or self.mask_index < 0:
            raise ValueError("mask_index must be a non-negative integer")
        for name in (
            "predicted_iou",
            "mask_stability",
            "edge_support",
            "geometric_stability",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_float(getattr(self, name), name=name),
            )
        object.__setattr__(self, "corners", corners)


@dataclass(frozen=True, slots=True)
class PromptedSamSelection:
    corners: np.ndarray
    score: float
    features: PromptedSamFeatures | None
    family: str
    mask_index: int | None
    fallback_used: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "corners", _immutable_quad(self.corners, name="corners"))
        object.__setattr__(self, "score", _bounded_float(self.score, name="score"))
        if self.features is not None and not isinstance(self.features, PromptedSamFeatures):
            raise TypeError("features must be PromptedSamFeatures or None")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be a non-empty string")
        if self.mask_index is not None and (
            type(self.mask_index) is not int or self.mask_index < 0
        ):
            raise ValueError("mask_index must be a non-negative integer or None")
        if type(self.fallback_used) is not bool:
            raise TypeError("fallback_used must be boolean")


def score_candidate(features: PromptedSamFeatures, policy: PromptedSamPolicy) -> float:
    if not isinstance(features, PromptedSamFeatures):
        raise TypeError("features must be PromptedSamFeatures")
    if not isinstance(policy, PromptedSamPolicy):
        raise TypeError("policy must be PromptedSamPolicy")
    score = sum(policy.weights[name] * getattr(features, name) for name in SCORE_FIELDS)
    return float(np.clip(score, 0.0, 1.0))


def _source_image(source: object) -> np.ndarray:
    image = np.asarray(source)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("source must be a non-empty three-channel image")
    return image


def _coarse_agreement(
    candidate: np.ndarray,
    coarse: np.ndarray,
    image_shape: tuple[int, ...],
) -> float:
    height, width = image_shape[:2]
    diagonal = float(np.hypot(width, height))
    mean_distance = float(np.linalg.norm(candidate - coarse, axis=1).mean())
    return float(np.clip(1.0 - mean_distance / max(diagonal, 1.0), 0.0, 1.0))


def select_candidate(
    candidates: Sequence[PromptedSamCandidate],
    coarse_corners: object,
    source: object,
    policy: PromptedSamPolicy,
) -> PromptedSamSelection:
    if not isinstance(policy, PromptedSamPolicy):
        raise TypeError("policy must be PromptedSamPolicy")
    image = _source_image(source)
    try:
        coarse = _immutable_quad(coarse_corners, name="coarse quadrilateral")
    except ValueError as error:
        raise ValueError("coarse quadrilateral is invalid") from error
    valid, reason = validate_quad(coarse, image.shape, reorder=False)
    if not valid:
        raise ValueError(f"coarse quadrilateral is invalid: {reason}")

    scored: list[tuple[float, PromptedSamCandidate, PromptedSamFeatures]] = []
    for candidate in candidates:
        if not isinstance(candidate, PromptedSamCandidate):
            raise TypeError("candidates must contain PromptedSamCandidate values")
        valid, _ = validate_quad(candidate.corners, image.shape, reorder=False)
        if not valid:
            continue
        features = PromptedSamFeatures(
            predicted_iou=candidate.predicted_iou,
            mask_stability=candidate.mask_stability,
            edge_support=candidate.edge_support,
            coarse_agreement=_coarse_agreement(candidate.corners, coarse, image.shape),
            geometric_stability=candidate.geometric_stability,
        )
        scored.append((score_candidate(features, policy), candidate, features))

    if scored:
        score, candidate, features = min(
            scored,
            key=lambda item: (
                -item[0],
                item[1].family,
                item[1].mask_index,
                item[1].corners.tobytes(),
            ),
        )
        if score >= policy.minimum_sam_score:
            return PromptedSamSelection(
                corners=candidate.corners,
                score=score,
                features=features,
                family=candidate.family,
                mask_index=candidate.mask_index,
                fallback_used=False,
            )

    return PromptedSamSelection(
        corners=coarse,
        score=0.0,
        features=None,
        family="docaligner",
        mask_index=None,
        fallback_used=True,
    )


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate policy field: {key}")
        document[key] = value
    return document


def load_verified_policy(
    model_dir: object,
    *,
    adapter_name: str,
) -> tuple[PromptedSamPolicy, str]:
    artifact = PROMPTED_SAM_POLICY_ARTIFACT
    if adapter_name != artifact.adapter:
        raise AdapterUnavailable("prompted-SAM adapter identity is invalid")
    payload = verified_model_bytes(
        model_dir,
        filename=artifact.filename,
        expected_sha256=artifact.sha256,
        expected_size_bytes=artifact.size_bytes,
        model_family="Prompted MobileSAM policy",
    )
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
        )
        if not isinstance(document, dict) or set(document) != _POLICY_FIELDS:
            raise ValueError(f"policy fields must be exactly {sorted(_POLICY_FIELDS)}")
        policy = PromptedSamPolicy(**document)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise AdapterUnavailable(f"Prompted MobileSAM policy is invalid: {error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    return policy, digest


__all__ = [
    "PromptedSamCandidate",
    "PromptedSamFeatures",
    "PromptedSamPolicy",
    "PromptedSamSelection",
    "SCORE_FIELDS",
    "load_verified_policy",
    "score_candidate",
    "select_candidate",
]
