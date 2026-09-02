"""Source-only CPU M-LSD Tiny occlusion experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Sequence
import uuid

import cv2
import numpy as np
import psutil
import torch

try:
    import resource as _resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    _resource = None

from openscaner.adapters import docaligner
from openscaner.geometry import validate_quad
from openscaner.geometry import order_quad
from openscaner.mlsd_tiny_decode import MODEL_INPUT_SIZE, decode_lines
from openscaner.models.mlsd_tiny import MLSDTiny, build_model


MODEL_FILENAME = "mlsd_tiny_512_fp32.pth"
MODEL_SHA256 = "3f2323cfb9faec5a0ef2454c4f68998da21a9f1b7918ce89b85886628b750b94"
MIN_ALIGNMENT = 0.98
MAX_EDGE_OFFSET_RATIO = 0.03
MIN_EDGE_OVERLAP = 0.35
CORNER_ENDPOINT_SUPPORT_RATIO = 0.01
_SEMANTIC_FILENAMES = frozenset({"result.json", "overlay.jpg", "rectified.jpg"})


@dataclass(frozen=True)
class Refinement:
    corners: np.ndarray | None
    reason: str | None
    diagnostics: dict[str, object]


def _cross2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def load_pinned_model(model_dir: Path) -> MLSDTiny:
    """Load the approved Tiny checkpoint on CPU after a byte-level integrity check."""
    path = model_dir / MODEL_FILENAME
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != MODEL_SHA256:
        raise ValueError(f"M-LSD Tiny checkpoint checksum mismatch: {path}")
    model = build_model().cpu()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _point_endpoint_distance(point: np.ndarray, line: np.ndarray) -> float:
    return min(float(np.linalg.norm(point - line[:2])), float(np.linalg.norm(point - line[2:])))


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    first_start, first_end = first[:2], first[2:]
    second_start, second_end = second[:2], second[2:]
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = _cross2d(first_direction, second_direction)
    if abs(denominator) < 1e-6:
        return None
    offset = second_start - first_start
    position = _cross2d(offset, second_direction) / denominator
    return first_start + position * first_direction


def _edge_candidates(
    lines: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    diagonal: float,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    direction = end - start
    edge_length = float(np.linalg.norm(direction))
    unit = direction / edge_length
    candidates: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        line_start, line_end = line[:2], line[2:]
        line_direction = line_end - line_start
        line_length = float(np.linalg.norm(line_direction))
        if line_length == 0.0:
            continue
        line_unit = line_direction / line_length
        alignment = abs(float(np.dot(unit, line_unit)))
        offset = (
            abs(_cross2d(unit, line_start - start))
            + abs(_cross2d(unit, line_end - start))
        ) / 2.0
        projected_start = float(np.dot(line_start - start, unit))
        projected_end = float(np.dot(line_end - start, unit))
        overlap = max(0.0, min(max(projected_start, projected_end), edge_length) - max(min(projected_start, projected_end), 0.0))
        overlap_ratio = overlap / edge_length
        candidate = {
            "index": index,
            "line": [float(value) for value in line],
            "alignment": alignment,
            "offset_px": offset,
            "overlap_ratio": overlap_ratio,
            "selected": False,
        }
        candidate["eligible"] = (
            alignment >= MIN_ALIGNMENT
            and offset <= diagonal * MAX_EDGE_OFFSET_RATIO
            and overlap_ratio >= MIN_EDGE_OVERLAP
        )
        candidate["score"] = alignment * overlap_ratio - offset / diagonal
        candidates.append(candidate)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return None, candidates
    selected = max(eligible, key=lambda item: (float(item["score"]), -int(item["index"])))
    selected["selected"] = True
    return selected, candidates


def refine_quad_from_lines(
    lines: np.ndarray,
    coarse_quad: np.ndarray,
    *,
    source_shape: tuple[int, int, int] | tuple[int, int],
) -> Refinement:
    """Require observed line support at every coarse corner before refinement."""
    detected = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
    coarse = np.asarray(coarse_quad, dtype=np.float32)
    if coarse.shape != (4, 2):
        raise ValueError("coarse quadrilateral must have four points")
    height, width = source_shape[:2]
    diagonal = float(np.hypot(width, height))
    if diagonal == 0.0:
        raise ValueError("source image dimensions must be positive")

    edge_records: list[dict[str, object]] = []
    selected_lines: list[np.ndarray | None] = []
    for edge_index in range(4):
        start, end = coarse[edge_index], coarse[(edge_index + 1) % 4]
        selected, candidates = _edge_candidates(detected, start, end, diagonal)
        selected_lines.append(
            detected[int(selected["index"])] if selected is not None else None
        )
        edge_records.append(
            {
                "edge_index": edge_index,
                "coarse_edge": [start.astype(float).tolist(), end.astype(float).tolist()],
                "candidates": candidates,
                "selected_line": None if selected is None else dict(selected),
            }
        )

    support_distance = diagonal * CORNER_ENDPOINT_SUPPORT_RATIO
    corner_records: list[dict[str, object]] = []
    for corner_index, point in enumerate(coarse):
        previous = selected_lines[(corner_index - 1) % 4]
        following = selected_lines[corner_index]
        distances = [
            None if line is None else _point_endpoint_distance(point, line)
            for line in (previous, following)
        ]
        observed = all(distance is not None and distance <= support_distance for distance in distances)
        corner_records.append(
            {
                "corner_index": corner_index,
                "coarse_corner": point.astype(float).tolist(),
                "support_distance_px": support_distance,
                "adjacent_endpoint_distances_px": distances,
                "observed": observed,
            }
        )

    diagnostics: dict[str, object] = {
        "detected_lines": detected.astype(float).tolist(),
        "edges": edge_records,
        "corners": corner_records,
    }
    if not all(bool(record["observed"]) for record in corner_records):
        return Refinement(None, "incomplete_observed_corner_support", diagnostics)

    intersections: list[np.ndarray] = []
    for edge_index in range(4):
        previous = selected_lines[(edge_index - 1) % 4]
        following = selected_lines[edge_index]
        assert previous is not None and following is not None
        point = _line_intersection(previous, following)
        if point is None:
            return Refinement(None, "parallel_selected_edges", diagnostics)
        intersections.append(point)
    refined = np.asarray(intersections, dtype=np.float32)
    valid, reason = validate_quad(refined, source_shape, reorder=False)
    if not valid:
        return Refinement(None, f"invalid_refined_quad:{reason}", diagnostics)
    diagnostics["refined_corners"] = refined.astype(float).tolist()
    return Refinement(refined, None, diagnostics)


def detect_lines(image: np.ndarray, model: MLSDTiny) -> np.ndarray:
    """Run the pinned network on a BGR image and decode its source-pixel lines."""
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty three-channel image")
    resized = cv2.resize(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    alpha = np.ones((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1), dtype=np.float32)
    tensor = np.concatenate((resized.astype(np.float32), alpha), axis=2)
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    tensor = torch.from_numpy(tensor / np.float32(127.5) - np.float32(1.0))
    with torch.inference_mode():
        output = model(tensor.cpu())
    return decode_lines(
        output.cpu(),
        source_shape=image.shape,
        score_threshold=0.10,
        distance_threshold=20.0,
    )


def _result(
    *,
    status: str,
    corners: np.ndarray | None,
    confidence: float,
    backend: str,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    return {
        "name": "mlsd_tiny",
        "status": status,
        "corners": None if corners is None else corners.astype(float).tolist(),
        "confidence": confidence,
        "backend": backend,
        "error": None,
        "diagnostics": diagnostics,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remove_controlled_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("refusing to remove an unsafe M-LSD output directory")
    entries = list(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise RuntimeError("refusing to remove unsafe M-LSD output contents")
    for entry in entries:
        entry.unlink()
    path.rmdir()


def _prepare_output_target(output_dir: Path) -> tuple[Path, Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise RuntimeError("M-LSD output parent must be a real directory")
    root = output_dir.parent.resolve(strict=True)
    target = root / output_dir.name
    if not output_dir.name or target.parent != root:
        raise RuntimeError("M-LSD output path escapes its parent directory")
    if target.is_symlink():
        raise RuntimeError("M-LSD output directory must not be a symlink")
    if target.exists() and not target.is_dir():
        raise RuntimeError("M-LSD output path is not a directory")
    if target.exists():
        for entry in target.iterdir():
            if entry.is_symlink() or not entry.is_file() or entry.name not in _SEMANTIC_FILENAMES:
                raise RuntimeError("M-LSD output contains an unexpected entry")
    return root, target


def _write_semantic_output(
    output_dir: Path,
    result: dict[str, object],
    image: np.ndarray | None = None,
    corners: np.ndarray | None = None,
) -> None:
    """Transactionally replace an output directory with only current semantic artifacts."""
    root, target = _prepare_output_target(output_dir)
    publication = Path(tempfile.mkdtemp(prefix=".mlsd-tiny-publish-", dir=root))
    backup: Path | None = None
    try:
        _write_json(publication / "result.json", result)
        if result["status"] == "ok":
            assert image is not None and corners is not None
            _write_overlay(publication, image, corners)
        if target.exists():
            backup = root / f".mlsd-tiny-old-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(publication, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None:
            _remove_controlled_directory(backup)
            backup = None
    finally:
        if publication.exists():
            _remove_controlled_directory(publication)


def _write_run_report(
    run_report_path: Path,
    result: dict[str, object],
    elapsed_ms: float,
) -> None:
    report_parent = run_report_path.parent
    report_parent.mkdir(parents=True, exist_ok=True)
    if report_parent.is_symlink() or not report_parent.is_dir():
        raise RuntimeError("M-LSD run-report parent must be a real directory")
    if run_report_path.is_symlink():
        raise RuntimeError("M-LSD run-report path must not be a symlink")
    report = {
        "schema_version": 1,
        "name": "mlsd_tiny",
        "status": result["status"],
        "semantic_result_sha256": hashlib.sha256(
            json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        ).hexdigest(),
        "elapsed_ms": elapsed_ms,
        "peak_rss_mb": _peak_rss_mb(),
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=".mlsd-tiny-run-report-", dir=report_parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, run_report_path)
    finally:
        temporary.unlink(missing_ok=True)


def _peak_rss_mb() -> float:
    if _resource is not None:
        peak_rss = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
        peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024.0
    else:
        memory = psutil.Process().memory_info()
        peak_working_set = getattr(memory, "peak_wset", None)
        if peak_working_set is None:
            raise RuntimeError("OS high-water mark is unavailable on this platform")
        peak_rss_bytes = float(peak_working_set)
    if peak_rss_bytes <= 0:
        raise RuntimeError("OS high-water mark is unavailable on this platform")
    return peak_rss_bytes / (1024.0 * 1024.0)


def _validate_run_report_path(output_dir: Path, run_report_path: Path | None) -> None:
    if run_report_path is None:
        return
    if run_report_path.resolve(strict=False).is_relative_to(output_dir.resolve(strict=False)):
        raise ValueError("M-LSD run-report must be outside the semantic output directory")


def _derive_run_report_path(output_dir: Path) -> Path:
    candidate_parent = output_dir.parent
    if candidate_parent.name.endswith("-candidates"):
        prefix = candidate_parent.name.removesuffix("-candidates")
        report_parent = candidate_parent.with_name(f"{prefix}-run-reports")
        return report_parent / f"{output_dir.name}.json"
    return candidate_parent / f"{output_dir.name}.run-report.json"


def _write_overlay(output_dir: Path, image: np.ndarray, corners: np.ndarray) -> None:
    overlay = image.copy()
    cv2.polylines(overlay, [np.rint(corners).astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    if not cv2.imwrite(str(output_dir / "overlay.jpg"), overlay):
        raise OSError("could not write M-LSD Tiny overlay")


def run_experiment(
    image_path: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    cpu_threads: int,
    run_report_path: Path | None = None,
) -> dict[str, object]:
    """Run the source-only coarse-quad plus line-support experiment on CPU."""
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    _validate_run_report_path(output_dir, run_report_path)
    torch.set_num_threads(cpu_threads)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode source image: {image_path}")
    started = time.perf_counter()
    model = load_pinned_model(model_dir)
    lines = detect_lines(image, model)
    coarse = docaligner.run(image, model_dir, cpu_threads)
    diagnostics: dict[str, object] = {
        "model": {"filename": MODEL_FILENAME, "sha256": MODEL_SHA256},
        "coarse_detection": {
            "backend": coarse.backend,
            "confidence": coarse.confidence,
            "corners": None if coarse.corners is None else np.asarray(coarse.corners, dtype=float).tolist(),
        },
    }
    if coarse.corners is None:
        diagnostics["refinement"] = {
            "detected_lines": lines.astype(float).tolist(),
            "edges": [],
            "corners": [],
        }
        diagnostics["not_detected_reason"] = "coarse_quad_unavailable"
        result = _result(
            status="not_detected",
            corners=None,
            confidence=0.0,
            backend="torch:cpu",
            diagnostics=diagnostics,
        )
        _write_semantic_output(output_dir, result)
        if run_report_path is not None:
            _write_run_report(run_report_path, result, (time.perf_counter() - started) * 1000.0)
        return result

    refinement = refine_quad_from_lines(
        lines,
        order_quad(np.asarray(coarse.corners, dtype=np.float32)),
        source_shape=image.shape,
    )
    diagnostics["refinement"] = refinement.diagnostics
    if refinement.corners is None:
        diagnostics["not_detected_reason"] = refinement.reason
        result = _result(
            status="not_detected",
            corners=None,
            confidence=0.0,
            backend="torch:cpu",
            diagnostics=diagnostics,
        )
        _write_semantic_output(output_dir, result)
        if run_report_path is not None:
            _write_run_report(run_report_path, result, (time.perf_counter() - started) * 1000.0)
        return result

    result = _result(
        status="ok",
        corners=refinement.corners,
        confidence=float(coarse.confidence),
        backend="torch:cpu",
        diagnostics=diagnostics,
    )
    _write_semantic_output(output_dir, result, image, refinement.corners)
    if run_report_path is not None:
        _write_run_report(run_report_path, result, (time.perf_counter() - started) * 1000.0)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the source-only CPU M-LSD Tiny experiment")
    parser.add_argument("--image", dest="image_path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    result = run_experiment(
        options.image_path,
        options.model_dir,
        options.output_dir,
        cpu_threads=options.cpu_threads,
        run_report_path=_derive_run_report_path(options.output_dir),
    )
    print(json.dumps({"status": result["status"], "output_dir": str(options.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
