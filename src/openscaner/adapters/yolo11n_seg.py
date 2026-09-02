"""Official Ultralytics YOLO11n-seg CPU adapter (AGPL-3.0)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from openscaner.adapters._model_integrity import verified_model_path
from openscaner.adapters.base import AdapterOutput, AdapterUnavailable
from openscaner.postprocess import quad_from_mask

MODEL_FILENAME = "yolo11n-seg.pt"
EXPECTED_SHA256 = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
COCO80_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
DOCUMENT_LIKE_CLASS_IDS = frozenset({73})


def _runtime():
    try:
        import torch
        from ultralytics import YOLO
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterUnavailable(
            "YOLO11n-seg requires the ultralytics and torch runtimes"
        ) from error
    return torch, YOLO


def _array(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalized_names(names: object) -> tuple[str, ...] | None:
    if isinstance(names, Mapping):
        if set(names) != set(range(len(COCO80_NAMES))):
            return None
        return tuple(str(names[index]) for index in range(len(COCO80_NAMES)))
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return tuple(str(name) for name in names)
    return None


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
        model_family="YOLO11n-seg",
    )
    torch, yolo_class = _runtime()
    torch.set_num_threads(cpu_threads)
    model = yolo_class(str(model_path), task="segment", verbose=False)
    if _normalized_names(getattr(model, "names", None)) != COCO80_NAMES:
        raise AdapterUnavailable(
            "YOLO11n-seg checkpoint does not expose the exact official COCO class map"
        )
    predictions = model.predict(
        source=source,
        device="cpu",
        verbose=False,
        retina_masks=True,
    )

    candidates: list[tuple[float, AdapterOutput]] = []
    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        masks = getattr(prediction, "masks", None)
        if boxes is None or masks is None:
            continue
        class_ids = _array(boxes.cls).reshape(-1)
        confidences = _array(boxes.conf).reshape(-1)
        mask_arrays = _array(masks.data)
        for index in range(min(len(class_ids), len(confidences), len(mask_arrays))):
            class_id = int(class_ids[index])
            if class_id not in DOCUMENT_LIKE_CLASS_IDS:
                continue
            recovered = quad_from_mask(
                mask_arrays[index],
                source_image=source,
                source_color_order="BGR",
                min_area_ratio=0.01,
            )
            if recovered is None:
                continue
            confidence = float(
                np.clip(float(confidences[index]) * recovered.confidence, 0.0, 1.0)
            )
            candidates.append(
                (
                    confidence,
                    AdapterOutput(
                        corners=recovered.corners,
                        confidence=confidence,
                        backend="torch:cpu",
                    ),
                )
            )

    if not candidates:
        return AdapterOutput(corners=None, confidence=0.0, backend="torch:cpu")
    return max(candidates, key=lambda candidate: candidate[0])[1]
