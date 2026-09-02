"""CPU-only DocAligner-prompted MobileSAM document-boundary adapter."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from openscaner.adapters.base import AdapterOutput
from openscaner.prompted_sam.artifacts import PROMPTED_SAM_POLICY_ARTIFACT
from openscaner.prompted_sam.policy import load_verified_policy, select_candidate
from openscaner.prompted_sam.signals import load_prompted_sam_predictor


_ADAPTER_NAME = PROMPTED_SAM_POLICY_ARTIFACT.adapter


def _source(image: object) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        raise ValueError("image must be a non-empty three-channel image")
    if source.dtype != np.uint8:
        raise ValueError("image must use uint8 values")
    return source


def run(image, model_dir, cpu_threads):
    source = _source(image)
    policy, policy_sha = load_verified_policy(model_dir, adapter_name=_ADAPTER_NAME)
    predictor = load_prompted_sam_predictor(model_dir, cpu_threads)
    signals = predictor.predict(source, policy)
    if signals.coarse_corners is None:
        return AdapterOutput(
            corners=None,
            confidence=0.0,
            backend="torch:cpu",
            diagnostics={
                "policy_sha256": policy_sha,
                "reason": "docaligner_not_detected",
            },
        )

    selected = select_candidate(
        signals.candidates,
        signals.coarse_corners,
        source,
        policy,
    )
    confidence = signals.coarse_confidence if selected.fallback_used else selected.score
    return AdapterOutput(
        corners=selected.corners,
        confidence=confidence,
        backend="torch:cpu",
        diagnostics={
            "policy_sha256": policy_sha,
            "selected_family": selected.family,
            "mask_index": selected.mask_index,
            "features": None if selected.features is None else asdict(selected.features),
            "fallback_used": selected.fallback_used,
        },
    )
