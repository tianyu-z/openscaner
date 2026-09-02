from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np

from openscaner.adapters._model_integrity import verified_model_bytes
from openscaner.adapters.base import AdapterUnavailable


OrientationScorer = Callable[[np.ndarray], float]
CorrectionAngle = Literal[0, 90, 180, 270]

MODEL_FILENAME = "pp_lcnet_x1_0_doc_ori.onnx"
MODEL_FAMILY = "PaddlePaddle PP-LCNet_x1_0 document orientation"
MODEL_SOURCE = (
    "https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx/"
    "resolve/main/inference.onnx"
)
EXPECTED_SIZE_BYTES = 6_788_069
EXPECTED_SHA256 = "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92"
CPU_PROVIDER = "CPUExecutionProvider"
ORIENTATION_LABELS: tuple[CorrectionAngle, ...] = (0, 90, 180, 270)
DEFAULT_AMBIGUITY_MARGIN = 0.05
MAX_INPUT_ASPECT_RATIO = 32.0
MAX_RESIZED_LONG_EDGE = 8_192
MAX_RESIZED_PIXELS = 2_097_152

_RESIZE_SHORT = 256
_CROP_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OrientationInferenceError(RuntimeError):
    """Raised when the orientation runtime or model output is incompatible."""


@dataclass(frozen=True)
class OrientationPrediction:
    predicted_angle: CorrectionAngle
    correction_angle: CorrectionAngle
    confidence: float
    probabilities: tuple[float, float, float, float]
    ambiguous: bool
    probability_margin: float
    ambiguity_margin: float

    def diagnostics(self) -> dict[str, object]:
        return {
            "angle": self.correction_angle,
            "predicted_angle": self.predicted_angle,
            "correction_angle": self.correction_angle,
            "confidence": self.confidence,
            "probabilities": {
                str(angle): probability
                for angle, probability in zip(ORIENTATION_LABELS, self.probabilities)
            },
            "ambiguous": self.ambiguous,
            "probability_margin": self.probability_margin,
            "ambiguity_margin": self.ambiguity_margin,
            "backend": CPU_PROVIDER,
            "model_family": MODEL_FAMILY,
            "model_filename": MODEL_FILENAME,
            "model_sha256": EXPECTED_SHA256,
        }


@dataclass(frozen=True)
class OrientationResult:
    image: np.ndarray
    prediction: OrientationPrediction


def orient_upright(
    image: np.ndarray,
    scorer: OrientationScorer,
    ambiguity_margin: float = 0.05,
) -> tuple[np.ndarray, int, dict[int, float]]:
    """Choose between 0 and 180 degrees; scorer exceptions propagate."""
    if not np.isfinite(ambiguity_margin) or ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be finite and non-negative")

    original = np.asarray(image)
    rotated = np.ascontiguousarray(np.rot90(original, 2))
    scores = {0: float(scorer(original)), 180: float(scorer(rotated))}
    if not all(np.isfinite(score) for score in scores.values()):
        raise ValueError("orientation scorer returned a non-finite score")

    if scores[180] > scores[0] + ambiguity_margin:
        return rotated, 180, scores
    return original, 0, scores


def _validate_image(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or source.shape[0] < 1
        or source.shape[1] < 1
    ):
        raise ValueError("image must be a non-empty three-channel BGR image")
    return source


def prepare_orientation_input(image: np.ndarray) -> np.ndarray:
    """Apply official preprocessing within a 32:1, 6 MiB resize bound."""
    source = _validate_image(image)
    height, width = source.shape[:2]
    scale = _RESIZE_SHORT / min(height, width)
    resized_height = round(height * scale)
    resized_width = round(width * scale)
    aspect_ratio = max(height, width) / min(height, width)
    resized_long_edge = max(resized_height, resized_width)
    resized_pixels = resized_height * resized_width
    if (
        aspect_ratio > MAX_INPUT_ASPECT_RATIO
        or resized_long_edge > MAX_RESIZED_LONG_EDGE
        or resized_pixels > MAX_RESIZED_PIXELS
    ):
        raise OrientationInferenceError(
            "orientation input aspect ratio or projected resize exceeds the "
            f"preprocessing limit: {aspect_ratio:.3f}:1, "
            f"{resized_width}x{resized_height} pixels; maximum "
            f"{MAX_INPUT_ASPECT_RATIO:g}:1, {MAX_RESIZED_LONG_EDGE} pixels on "
            f"the long edge, and {MAX_RESIZED_PIXELS} pixels total"
        )

    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    top = (resized_height - _CROP_SIZE) // 2
    left = (resized_width - _CROP_SIZE) // 2
    cropped = resized[top : top + _CROP_SIZE, left : left + _CROP_SIZE]
    normalized = cropped.astype(np.float32) / np.float32(255.0)
    normalized = (normalized - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


def decode_orientation_output(
    output: object,
    *,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> OrientationPrediction:
    """Decode the model's four probabilities as a CCW correction angle."""
    if not np.isfinite(ambiguity_margin) or ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be finite and non-negative")

    probabilities = np.asarray(output)
    if probabilities.shape != (1, len(ORIENTATION_LABELS)):
        raise OrientationInferenceError(
            "orientation output must contain probabilities with shape (1, 4)"
        )
    probabilities = probabilities.astype(np.float64, copy=False)[0]
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.isclose(float(probabilities.sum()), 1.0, rtol=0.0, atol=1e-4)
    ):
        raise OrientationInferenceError(
            "orientation output must be finite normalized probabilities"
        )

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])
    runner_up = float(np.max(np.delete(probabilities, class_index)))
    probability_margin = confidence - runner_up
    ambiguous = probability_margin <= ambiguity_margin
    predicted_angle = ORIENTATION_LABELS[class_index]
    return OrientationPrediction(
        predicted_angle=predicted_angle,
        correction_angle=0 if ambiguous else predicted_angle,
        confidence=confidence,
        probabilities=cast(
            tuple[float, float, float, float],
            tuple(float(value) for value in probabilities),
        ),
        ambiguous=ambiguous,
        probability_margin=probability_margin,
        ambiguity_margin=float(ambiguity_margin),
    )


def apply_orientation_correction(
    image: np.ndarray,
    correction_angle: CorrectionAngle,
) -> np.ndarray:
    """Apply the model label as PaddleX does: a counterclockwise rotation."""
    if correction_angle not in ORIENTATION_LABELS:
        raise ValueError("correction_angle must be one of 0, 90, 180, or 270")
    source = np.asarray(image)
    return np.ascontiguousarray(np.rot90(source, correction_angle // 90))


def _runtime():
    try:
        import onnxruntime as ort
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterUnavailable(
            "document orientation requires the onnxruntime runtime"
        ) from error
    return ort


def _validate_session(session) -> tuple[str, str]:
    active_providers = list(session.get_providers())
    if active_providers != [CPU_PROVIDER]:
        raise OrientationInferenceError(
            "orientation session must activate CPUExecutionProvider only"
        )

    inputs = list(session.get_inputs())
    if len(inputs) != 1:
        raise OrientationInferenceError("orientation model must have exactly one input")
    input_shape = getattr(inputs[0], "shape", None)
    input_type = getattr(inputs[0], "type", None)
    if (
        not isinstance(input_shape, (list, tuple))
        or len(input_shape) != 4
        or list(input_shape[1:]) != [3, 224, 224]
        or input_type != "tensor(float)"
    ):
        raise OrientationInferenceError(
            "orientation model input must be float NCHW with shape (N, 3, 224, 224)"
        )

    outputs = list(session.get_outputs())
    if len(outputs) != 1:
        raise OrientationInferenceError("orientation model must have exactly one output")
    output_shape = getattr(outputs[0], "shape", None)
    output_type = getattr(outputs[0], "type", None)
    if (
        not isinstance(output_shape, (list, tuple))
        or len(output_shape) != 2
        or output_shape[1] != 4
        or output_type != "tensor(float)"
    ):
        raise OrientationInferenceError(
            "orientation model output must be float probabilities with shape (N, 4)"
        )
    return inputs[0].name, outputs[0].name


def classify_orientation(
    image: np.ndarray,
    model_dir: str | Path,
    cpu_threads: int,
    *,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> OrientationPrediction:
    """Classify the CCW correction using the pinned ONNX model on CPU only."""
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    model_bytes = verified_model_bytes(
        model_dir,
        filename=MODEL_FILENAME,
        expected_sha256=EXPECTED_SHA256,
        expected_size_bytes=EXPECTED_SIZE_BYTES,
        model_family=MODEL_FAMILY,
    )
    ort = _runtime()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = cpu_threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        session = ort.InferenceSession(
            model_bytes,
            sess_options=session_options,
            providers=[CPU_PROVIDER],
        )
    except Exception as error:
        raise OrientationInferenceError(
            "orientation model is incompatible with onnxruntime"
        ) from error

    input_name, output_name = _validate_session(session)
    try:
        input_tensor = prepare_orientation_input(image)
    except OrientationInferenceError:
        raise
    except Exception as error:
        raise OrientationInferenceError("orientation preprocessing failed") from error
    try:
        outputs = session.run(
            [output_name],
            {input_name: input_tensor},
        )
    except Exception as error:
        raise OrientationInferenceError("orientation inference failed") from error
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise OrientationInferenceError(
            "orientation inference did not return exactly one probability output"
        )
    return decode_orientation_output(outputs[0], ambiguity_margin=ambiguity_margin)


def orient_document(
    image: np.ndarray,
    model_dir: str | Path,
    cpu_threads: int,
    *,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> OrientationResult:
    prediction = classify_orientation(
        image,
        model_dir,
        cpu_threads,
        ambiguity_margin=ambiguity_margin,
    )
    return OrientationResult(
        image=apply_orientation_correction(image, prediction.correction_angle),
        prediction=prediction,
    )
