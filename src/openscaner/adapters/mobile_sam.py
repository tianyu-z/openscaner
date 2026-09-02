"""MobileSAM automatic-mask CPU adapter (Apache-2.0)."""

from __future__ import annotations

import cv2
import numpy as np

from openscaner.adapters._model_integrity import verified_model_path
from openscaner.adapters.base import AdapterOutput, AdapterUnavailable
from openscaner.postprocess import MaskQuad, quad_from_mask

MODEL_FILENAME = "mobile_sam.pt"
EXPECTED_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
MAX_GENERATION_DIMENSION = 1024
POINTS_PER_BATCH = 8
MIN_ACCEPTANCE_CONFIDENCE = 0.70


def _runtime():
    try:
        import torch
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterUnavailable(
            "MobileSAM requires the mobile_sam and torch runtimes"
        ) from error
    return torch, SamAutomaticMaskGenerator, sam_model_registry


def _paper_likelihood(image: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != image.shape[:2] or not np.any(selected):
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1][selected].astype(np.float32) / 255.0
    brightness = hsv[:, :, 2][selected].astype(np.float32) / 255.0
    low_saturation = 1.0 - float(np.mean(saturation))
    light = float(np.mean(brightness))
    consistency = 1.0 - min(1.0, float(np.std(brightness)) / 0.25)
    return float(
        np.clip(0.40 * light + 0.35 * low_saturation + 0.25 * consistency, 0.0, 1.0)
    )


def _score(annotation: dict, recovered: MaskQuad, image: np.ndarray) -> float:
    """Return a fixed weighted confidence on a documented ``[0, 1]`` scale.

    Geometry contributes 35%, source-border support 25%, interior
    paper-likelihood 25%, and MobileSAM's own quality estimates 15%.
    """
    geometry = recovered.geometric_confidence
    border_support = recovered.edge_confidence or 0.0
    paper = _paper_likelihood(image, np.asarray(annotation["segmentation"]))
    generator_quality = 0.5 * float(annotation.get("predicted_iou", 0.0)) + 0.5 * float(
        annotation.get("stability_score", 0.0)
    )
    return float(
        np.clip(
            0.35 * geometry
            + 0.25 * border_support
            + 0.25 * paper
            + 0.15 * generator_quality,
            0.0,
            1.0,
        )
    )


def _generation_image(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= MAX_GENERATION_DIMENSION:
        return image
    scale = MAX_GENERATION_DIMENSION / float(largest_dimension)
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def run(image, model_dir, cpu_threads):
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        raise ValueError("image must be a non-empty three-channel image")

    model_path = verified_model_path(
        model_dir,
        filename=MODEL_FILENAME,
        expected_sha256=EXPECTED_SHA256,
        model_family="MobileSAM",
    )
    torch, generator_class, registry = _runtime()
    torch.set_num_threads(cpu_threads)
    try:
        model_builder = registry["vit_t"]
    except KeyError as error:
        raise AdapterUnavailable(
            "MobileSAM runtime does not provide the vit_t model"
        ) from error
    model = model_builder(checkpoint=str(model_path))
    model.to(device="cpu")
    model.eval()
    generator = generator_class(
        model,
        points_per_side=16,
        points_per_batch=POINTS_PER_BATCH,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.88,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    generation_image = _generation_image(source)
    annotations = generator.generate(cv2.cvtColor(generation_image, cv2.COLOR_BGR2RGB))

    candidates: list[tuple[float, MaskQuad]] = []
    for annotation in annotations:
        mask = np.asarray(annotation.get("segmentation"))
        recovered = quad_from_mask(
            mask,
            source_image=source,
            source_color_order="BGR",
            min_area_ratio=0.01,
        )
        if recovered is None:
            continue
        candidates.append((_score(annotation, recovered, generation_image), recovered))

    if not candidates:
        return AdapterOutput(corners=None, confidence=0.0, backend="torch:cpu")
    confidence, recovered = max(candidates, key=lambda candidate: candidate[0])
    if confidence < MIN_ACCEPTANCE_CONFIDENCE:
        return AdapterOutput(corners=None, confidence=0.0, backend="torch:cpu")
    return AdapterOutput(
        corners=recovered.corners, confidence=confidence, backend="torch:cpu"
    )
