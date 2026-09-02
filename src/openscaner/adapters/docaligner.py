"""Direct DocAligner LCNet100 heatmap ONNX CPU adapter (Apache-2.0)."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from openscaner.adapters._model_integrity import verified_model_bytes
from openscaner.adapters.base import AdapterOutput, AdapterUnavailable
from openscaner.fusion.artifacts import FUSION_DEPENDENCY_ARTIFACTS, ModelArtifactIdentity
from openscaner.fusion.signals import DocAlignerSignals
from openscaner.geometry import validate_quad

_FROZEN_ARTIFACT = FUSION_DEPENDENCY_ARTIFACTS["docaligner"]
MODEL_FILENAME = _FROZEN_ARTIFACT.filename
EXPECTED_SHA256 = _FROZEN_ARTIFACT.sha256
EXPECTED_SIZE_BYTES = _FROZEN_ARTIFACT.size_bytes
INPUT_SIZE = (256, 256)
HEATMAP_THRESHOLD = 0.3
OUTPUT_NAME = "heatmap"
CPU_PROVIDER = "CPUExecutionProvider"


def _runtime():
    try:
        import onnxruntime as ort
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterUnavailable(
            "DocAligner requires the onnxruntime runtime"
        ) from error
    return ort


def _input_tensor(source: np.ndarray) -> np.ndarray:
    """Match upstream preprocessing: OpenCV BGR, CHW float32, divided by 255."""
    resized = cv2.resize(source, INPUT_SIZE)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32)[None] / np.float32(255.0)


def _largest_centroid(
    heatmap: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray | None:
    """Match DocAligner/capybara thresholding and polygon centroid semantics."""
    width, height = image_size
    resized = cv2.resize(np.asarray(heatmap, dtype=np.float32), (width, height))
    resized[resized < HEATMAP_THRESHOLD] = 0.0
    quantized = np.uint8(resized * 255.0)
    _, binary = cv2.threshold(
        quantized,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = [contour for contour in contours if len(contour) >= 3]
    if not polygons:
        return None

    moments = [cv2.moments(polygon) for polygon in polygons]
    largest_area = max(moment["m00"] for moment in moments)
    largest = [moment for moment in moments if moment["m00"] == largest_area]
    if len(largest) != 1:
        return None
    moment = largest[0]
    return np.array(
        [
            moment["m10"] / (moment["m00"] + 1e-5),
            moment["m01"] / (moment["m00"] + 1e-5),
        ],
        dtype=np.float32,
    )


def _heatmap_output(session) -> object:
    matches = [output for output in session.get_outputs() if output.name == OUTPUT_NAME]
    if len(matches) != 1:
        raise AdapterUnavailable(
            "DocAligner LCNet100 checkpoint has no uniquely named heatmap output"
        )
    output = matches[0]
    shape = getattr(output, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 4 or shape[1] != 4:
        raise AdapterUnavailable(
            "DocAligner LCNet100 heatmap output is not exactly four-channel"
        )
    return output


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
    return source


def accepted_docaligner_corners(
    corners: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Return a copied raw quad exactly when production accepts it."""
    if corners is None:
        return None
    try:
        valid, _ = validate_quad(corners, image_shape, reorder=False)
    except (TypeError, ValueError):
        return None
    if not valid:
        return None
    return np.asarray(corners, dtype=np.float32).copy()


def _load_docaligner_session(
    model_dir,
    cpu_threads,
) -> tuple[object, ModelArtifactIdentity]:
    model_bytes, identity = _read_docaligner_artifact(model_dir)
    return _build_docaligner_session(model_bytes, cpu_threads), identity


def _read_docaligner_artifact(model_dir) -> tuple[bytes, ModelArtifactIdentity]:
    model_bytes = verified_model_bytes(
        model_dir,
        filename=MODEL_FILENAME,
        expected_sha256=EXPECTED_SHA256,
        expected_size_bytes=EXPECTED_SIZE_BYTES,
        model_family="DocAligner",
    )
    return model_bytes, ModelArtifactIdentity(
        adapter="docaligner",
        filename=MODEL_FILENAME,
        sha256=hashlib.sha256(model_bytes).hexdigest(),
        size_bytes=len(model_bytes),
    )


def _build_docaligner_session(model_bytes: bytes, cpu_threads: int) -> object:
    ort = _runtime()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = cpu_threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        model_bytes,
        sess_options=session_options,
        providers=[CPU_PROVIDER],
    )

    active_providers = list(session.get_providers())
    if active_providers != [CPU_PROVIDER]:
        raise AdapterUnavailable(
            "DocAligner session must activate CPUExecutionProvider only"
        )
    _heatmap_output(session)
    return session


def _infer_docaligner_signals(
    session,
    source: np.ndarray,
) -> DocAlignerSignals:
    active_providers = list(session.get_providers())
    if active_providers != [CPU_PROVIDER]:
        raise AdapterUnavailable(
            "DocAligner session must activate CPUExecutionProvider only"
        )
    _heatmap_output(session)

    input_name = session.get_inputs()[0].name
    outputs = session.run([OUTPUT_NAME], {input_name: _input_tensor(source)})
    if len(outputs) != 1:
        raise AdapterUnavailable(
            "DocAligner LCNet100 heatmap inference returned an incompatible output"
        )
    raw_heatmaps = outputs[0]
    device = getattr(raw_heatmaps, "device", None)
    device_type = getattr(device, "type", device)
    if device is not None and str(device_type) != "cpu":
        raise AdapterUnavailable("DocAligner returned non-CPU heatmaps")
    heatmaps = np.asarray(raw_heatmaps)
    if (
        heatmaps.ndim != 4
        or heatmaps.shape[0] != 1
        or heatmaps.shape[1] != 4
        or heatmaps.shape[2] < 1
        or heatmaps.shape[3] < 1
    ):
        raise AdapterUnavailable(
            "DocAligner LCNet100 heatmap output is not exactly four-channel"
        )
    if not np.issubdtype(heatmaps.dtype, np.floating):
        raise AdapterUnavailable(
            "DocAligner LCNet100 heatmaps must use a floating-point dtype"
        )
    if not np.isfinite(heatmaps).all():
        raise AdapterUnavailable(
            "DocAligner LCNet100 heatmaps must contain only finite values"
        )

    height, width = source.shape[:2]
    points = [
        _largest_centroid(heatmaps[0, index], (width, height)) for index in range(4)
    ]
    corners = None if any(point is None for point in points) else np.stack(points)
    confidence = float(
        np.clip(
            np.mean([float(np.max(heatmaps[0, index])) for index in range(4)]),
            0.0,
            1.0,
        )
    )
    return DocAlignerSignals(
        corners=corners,
        confidence=confidence,
        heatmaps=heatmaps,
        backend=active_providers[0],
    )


def _predict_docaligner_signals(
    source: np.ndarray,
    model_dir,
    cpu_threads,
) -> DocAlignerSignals:
    session, _ = _load_docaligner_session(model_dir, cpu_threads)
    return _infer_docaligner_signals(session, source)


def run(image, model_dir, cpu_threads):
    threads = _validate_cpu_threads(cpu_threads)
    source = _validate_source(image)
    signals = _predict_docaligner_signals(source, model_dir, threads)

    corners = accepted_docaligner_corners(signals.corners, source.shape)
    if corners is None:
        return AdapterOutput(corners=None, confidence=0.0, backend=signals.backend)
    return AdapterOutput(
        corners=corners,
        confidence=signals.confidence,
        backend=signals.backend,
    )
