"""Verified CPU runtimes and source-only signals for prompted MobileSAM."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import io

import cv2
import numpy as np

from openscaner.adapters import docaligner
from openscaner.adapters._model_integrity import verified_model_bytes
from openscaner.adapters.base import AdapterUnavailable
from openscaner.fusion.artifacts import ModelArtifactIdentity
from openscaner.fusion.signals import DocAlignerSignals
from openscaner.geometry import validate_quad
from openscaner.prompted_sam.artifacts import PROMPTED_SAM_DEPENDENCY_ARTIFACTS
from openscaner.prompted_sam.geometry import (
    crop_from_quad,
    mask_quad_in_source,
    prompt_from_quad,
)
from openscaner.prompted_sam.policy import PromptedSamCandidate, PromptedSamPolicy


MAX_PROMPT_CALLS = 1
MAX_RETURNED_MASKS = 3
_STABILITY_OFFSET = 1.0


def _runtime():
    try:
        import torch
        from mobile_sam import SamPredictor, sam_model_registry
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterUnavailable(
            "Prompted MobileSAM requires the mobile_sam and torch runtimes"
        ) from error
    return torch, SamPredictor, sam_model_registry


def _bounded_confidence(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(result, 0.0, 1.0))


def _immutable_quad(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    corners = np.asarray(value)
    if (
        corners.shape != (4, 2)
        or not np.issubdtype(corners.dtype, np.number)
        or not np.isfinite(corners).all()
    ):
        raise ValueError("coarse_corners must be a finite 4x2 array or None")
    result = corners.astype(np.float32, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PromptedSamSignals:
    coarse_corners: np.ndarray | None
    coarse_confidence: float
    candidates: tuple[PromptedSamCandidate, ...]
    backend: str

    def __post_init__(self) -> None:
        coarse = _immutable_quad(self.coarse_corners)
        confidence = _bounded_confidence(
            self.coarse_confidence,
            name="coarse_confidence",
        )
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, PromptedSamCandidate)
            for candidate in self.candidates
        ):
            raise TypeError("candidates must be a tuple of PromptedSamCandidate values")
        if self.backend != "torch:cpu":
            raise ValueError("backend must be exactly 'torch:cpu'")
        object.__setattr__(self, "coarse_corners", coarse)
        object.__setattr__(self, "coarse_confidence", confidence)


def _mask_stability(logits: np.ndarray) -> float:
    values = np.asarray(logits)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        return 0.0
    if not np.isfinite(values).all():
        return 0.0
    permissive = int(np.count_nonzero(values > -_STABILITY_OFFSET))
    if permissive == 0:
        return 0.0
    strict = int(np.count_nonzero(values > _STABILITY_OFFSET))
    return float(np.clip(strict / permissive, 0.0, 1.0))


class PromptedSamSignalPredictor:
    """Reuse one verified DocAligner session and one MobileSAM model."""

    def __init__(
        self,
        *,
        docaligner_session: object,
        sam_predictor: object,
        artifacts: Mapping[str, ModelArtifactIdentity],
        docaligner_infer: Callable[[object, np.ndarray], DocAlignerSignals] | None = None,
    ) -> None:
        self._docaligner_session = docaligner_session
        self._sam_predictor = sam_predictor
        self._docaligner_infer = (
            docaligner._infer_docaligner_signals
            if docaligner_infer is None
            else docaligner_infer
        )
        self.artifacts = dict(artifacts)

    def predict(self, image: object, policy: PromptedSamPolicy) -> PromptedSamSignals:
        return self.predict_many(image, (policy,))[0]

    def predict_many(
        self,
        image: object,
        policies: tuple[PromptedSamPolicy, ...],
    ) -> tuple[PromptedSamSignals, ...]:
        source = docaligner._validate_source(image)
        if not isinstance(policies, tuple) or not policies:
            raise ValueError("policies must be a non-empty tuple")
        if any(not isinstance(policy, PromptedSamPolicy) for policy in policies):
            raise TypeError("policies must contain PromptedSamPolicy values")
        coarse_signals = self._docaligner_infer(self._docaligner_session, source)
        coarse = coarse_signals.corners
        if coarse is None:
            result = PromptedSamSignals(
                coarse_corners=None,
                coarse_confidence=coarse_signals.confidence,
                candidates=(),
                backend="torch:cpu",
            )
            return tuple(result for _ in policies)
        valid, _ = validate_quad(coarse, source.shape, reorder=False)
        if not valid:
            result = PromptedSamSignals(
                coarse_corners=None,
                coarse_confidence=0.0,
                candidates=(),
                backend="torch:cpu",
            )
            return tuple(result for _ in policies)

        cached: dict[tuple[float, float, bool], PromptedSamSignals] = {}
        for margin in sorted({policy.crop_margin_ratio for policy in policies}):
            crop = crop_from_quad(source.shape, coarse, margin_ratio=margin)
            crop_image = source[crop.top : crop.bottom, crop.left : crop.right]
            self._sam_predictor.set_image(
                cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB),
                image_format="RGB",
            )
            geometries = sorted(
                {
                    (policy.foreground_scale, policy.include_background)
                    for policy in policies
                    if policy.crop_margin_ratio == margin
                }
            )
            for foreground_scale, include_background in geometries:
                policy = next(
                    item
                    for item in policies
                    if item.crop_margin_ratio == margin
                    and item.foreground_scale == foreground_scale
                    and item.include_background == include_background
                )
                key = (margin, foreground_scale, include_background)
                cached[key] = self._predict_prompt(
                    source,
                    coarse_signals,
                    crop,
                    policy,
                )
        return tuple(
            cached[
                (
                    policy.crop_margin_ratio,
                    policy.foreground_scale,
                    policy.include_background,
                )
            ]
            for policy in policies
        )

    def _predict_prompt(
        self,
        source: np.ndarray,
        coarse_signals: DocAlignerSignals,
        crop,
        policy: PromptedSamPolicy,
    ) -> PromptedSamSignals:
        coarse = coarse_signals.corners
        assert coarse is not None
        prompt = prompt_from_quad(
            coarse,
            crop,
            foreground_scale=policy.foreground_scale,
            include_background=policy.include_background,
        )
        masks, scores, low_resolution_logits = self._sam_predictor.predict(
            point_coords=prompt.points,
            point_labels=prompt.labels,
            box=prompt.box,
            mask_input=None,
            multimask_output=True,
            return_logits=False,
        )
        mask_array = np.asarray(masks)
        score_array = np.asarray(scores)
        logits_array = np.asarray(low_resolution_logits)
        if (
            mask_array.ndim != 3
            or score_array.ndim != 1
            or logits_array.ndim != 3
            or not 1 <= len(mask_array) <= MAX_RETURNED_MASKS
            or len(score_array) != len(mask_array)
            or len(logits_array) != len(mask_array)
            or not np.issubdtype(mask_array.dtype, np.number)
            and mask_array.dtype != np.bool_
            or not np.issubdtype(score_array.dtype, np.number)
            or not np.isfinite(score_array).all()
        ):
            return PromptedSamSignals(
                coarse_corners=coarse,
                coarse_confidence=coarse_signals.confidence,
                candidates=(),
                backend="torch:cpu",
            )

        candidates: list[PromptedSamCandidate] = []
        for index, mask in enumerate(mask_array):
            try:
                recovered = mask_quad_in_source(
                    mask,
                    crop,
                    source,
                    foreground_points=prompt.points[prompt.labels == 1],
                )
            except (TypeError, ValueError, cv2.error):
                continue
            if recovered is None:
                continue
            try:
                candidates.append(
                    PromptedSamCandidate(
                        corners=recovered.corners,
                        family=prompt.family,
                        mask_index=index,
                        predicted_iou=_bounded_confidence(
                            score_array[index],
                            name="predicted_iou",
                        ),
                        mask_stability=_mask_stability(logits_array[index]),
                        edge_support=float(recovered.edge_confidence or 0.0),
                        geometric_stability=recovered.geometric_confidence,
                    )
                )
            except (TypeError, ValueError):
                continue
        return PromptedSamSignals(
            coarse_corners=coarse,
            coarse_confidence=coarse_signals.confidence,
            candidates=tuple(candidates),
            backend="torch:cpu",
        )


def load_prompted_sam_predictor(
    model_dir: object,
    cpu_threads: int,
) -> PromptedSamSignalPredictor:
    threads = docaligner._validate_cpu_threads(cpu_threads)
    docaligner_bytes, docaligner_identity = docaligner._read_docaligner_artifact(
        model_dir
    )
    expected_docaligner = PROMPTED_SAM_DEPENDENCY_ARTIFACTS["docaligner"]
    if docaligner_identity != expected_docaligner:
        raise AdapterUnavailable("DocAligner identity does not match prompted-SAM dependency")

    mobile_identity = PROMPTED_SAM_DEPENDENCY_ARTIFACTS["mobile_sam"]
    mobile_bytes = verified_model_bytes(
        model_dir,
        filename=mobile_identity.filename,
        expected_sha256=mobile_identity.sha256,
        expected_size_bytes=mobile_identity.size_bytes,
        model_family="MobileSAM",
    )
    torch, predictor_class, registry = _runtime()
    torch.set_num_threads(threads)
    try:
        model_builder = registry["vit_t"]
    except KeyError as error:
        raise AdapterUnavailable("MobileSAM runtime does not provide the vit_t model") from error
    model = model_builder(checkpoint=None)
    try:
        state_dict = torch.load(
            io.BytesIO(mobile_bytes),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device="cpu")
        model.eval()
        sam_predictor = predictor_class(model)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise AdapterUnavailable("MobileSAM checkpoint could not be loaded safely") from error
    session = docaligner._build_docaligner_session(docaligner_bytes, threads)
    return PromptedSamSignalPredictor(
        docaligner_session=session,
        sam_predictor=sam_predictor,
        artifacts={
            docaligner_identity.adapter: docaligner_identity,
            mobile_identity.adapter: mobile_identity,
        },
    )


__all__ = [
    "MAX_PROMPT_CALLS",
    "PromptedSamSignalPredictor",
    "PromptedSamSignals",
    "load_prompted_sam_predictor",
]
