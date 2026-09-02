"""CPU benchmark orchestration with defense-in-depth candidate isolation.

Environment filtering, source checks, and a private working directory prevent
accidental evaluator-data access. They are not a portable OS security sandbox,
and CPU backend declarations cannot prevent a deliberately dishonest adapter.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import psutil

from openscaner.adapters.base import (
    AdapterContractError,
    AdapterOutput,
    AdapterUnavailable,
    CPU_BACKENDS,
    discover_adapters,
    load_adapter,
)
from openscaner.contracts import CandidateResult, Status
from openscaner.geometry import order_quad, validate_quad, warp_document
from openscaner.orientation import orient_document


_RESULT_FILENAME = "result.json"
_SUCCESS_FILENAMES = ("overlay.jpg", "rectified.jpg")
_RESULT_FIELDS = frozenset(
    {
        "name",
        "status",
        "corners",
        "confidence",
        "backend",
        "elapsed_ms",
        "peak_rss_mb",
        "error",
        "diagnostics",
    }
)
_FORBIDDEN_RUNTIME_IDENTIFIERS = frozenset(
    {
        "reference",
        "reference_path",
        "reference_image",
        "reference_asset",
        "reference_assets",
        "reference_fixture",
        "reference_fixtures",
        "reference_corners",
        "reference_quad",
        "manual_points",
        "manual_corners",
        "manual_quad",
        "manual_roi",
        "manual_geometry",
        "prompt_points",
        "prompt_corners",
        "prompt_quad",
        "prompt_roi",
        "prompt_coordinates",
        "roi",
        "ground_truth",
        "ground_truth_path",
        "ground_truth_image",
        "ground_truth_corners",
        "ground_truth_quad",
        "evaluator_asset",
        "evaluator_assets",
        "evaluator_path",
        "evaluator_data",
        "evaluator_fixture",
        "evaluator_fixtures",
    }
)
_FORBIDDEN_ASSET_TOKENS = _FORBIDDEN_RUNTIME_IDENTIFIERS | frozenset(
    {
        "evaluator",
        "groundtruth",
        "references",
    }
)
_FORBIDDEN_ASSET_TOKEN_SEQUENCES = frozenset(
    tuple(token.replace("-", "_").split("_")) for token in _FORBIDDEN_ASSET_TOKENS
)
_KNOWN_IMAGE_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".bmp",
        ".dib",
        ".doc",
        ".docx",
        ".gif",
        ".heic",
        ".heif",
        ".jpeg",
        ".jpg",
        ".odt",
        ".pdf",
        ".png",
        ".svg",
        ".tif",
        ".tiff",
        ".webp",
    }
)
_MAX_RESULT_BYTES = 1_000_000
_LOG_TAIL_BYTES = 8_192
_PROCESS_POLL_SECONDS = 0.01
_ALLOWED_PARENT_ENVIRONMENT = frozenset(
    {
        # Launching/importing, including temporary adapter packages in tests.
        "PATH",
        "PYTHONPATH",
        "SYSTEMROOT",
        "WINDIR",
        # Standard filesystem and locale behavior.
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        # Public model caches used by supported candidate runtimes.
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "TORCH_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "ONNX_HOME",
        "YOLO_CONFIG_DIR",
        # Certificate and proxy settings needed for optional model downloads.
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)


def build_candidate_command(
    *,
    name: str,
    module: str,
    image_path: Path,
    model_dir: Path,
    output_dir: Path,
    cpu_threads: int,
) -> list[str]:
    """Build the private worker command for one candidate."""
    return [
        sys.executable,
        "-m",
        "openscaner.benchmark",
        "_candidate",
        "--name",
        name,
        "--module",
        module,
        "--image",
        str(image_path),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir),
        "--cpu-threads",
        str(cpu_threads),
    ]


def _cpu_environment(cpu_threads: int) -> dict[str, str]:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")

    environment = {key: os.environ[key] for key in _ALLOWED_PARENT_ENVIRONMENT if key in os.environ}
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "GPU_DEVICE_ORDINAL": "",
            "ONEAPI_DEVICE_SELECTOR": "*:cpu",
            "SYCL_DEVICE_FILTER": "cpu",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "OPENCV_OPENCL_RUNTIME": "disabled",
            "OMP_TARGET_OFFLOAD": "DISABLED",
            "ACC_DEVICE_TYPE": "host",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "OPENSCANER_DEVICE": "cpu",
            "OPENSCANER_ONNX_PROVIDER": "CPUExecutionProvider",
            "OMP_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "VECLIB_MAXIMUM_THREADS": str(cpu_threads),
            "NUMEXPR_NUM_THREADS": str(cpu_threads),
        }
    )
    return environment


@dataclass(frozen=True)
class _ProcessOutcome:
    returncode: int
    elapsed_ms: float
    peak_rss_mb: float
    stdout_tail: str
    stderr_tail: str
    timed_out: bool


def _process_tree_rss(process: psutil.Process) -> int:
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = [process]

    total = 0
    for member in processes:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total


def _remember_descendants(tracked: psutil.Process, descendants: set[psutil.Process]) -> None:
    try:
        descendants.update(tracked.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def _signal_process_group(process: subprocess.Popen[bytes], signal_number: int) -> None:
    if os.name == "nt":
        return
    try:
        os.killpg(process.pid, signal_number)
    except (ProcessLookupError, PermissionError):
        pass


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    tracked: psutil.Process,
    descendants: set[psutil.Process],
) -> None:
    _remember_descendants(tracked, descendants)
    processes = [*descendants, tracked]

    for member in processes:
        try:
            member.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _signal_process_group(process, signal.SIGTERM)
    _, alive = psutil.wait_procs(processes, timeout=0.25)
    for member in alive:
        try:
            member.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _signal_process_group(process, getattr(signal, "SIGKILL", signal.SIGTERM))
    psutil.wait_procs(alive, timeout=0.25)
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class _TailBuffer:
    def __init__(self, limit: int = _LOG_TAIL_BYTES) -> None:
        self._limit = limit
        self._data = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            if len(chunk) >= self._limit:
                self._data[:] = chunk[-self._limit :]
                return
            self._data.extend(chunk)
            overflow = len(self._data) - self._limit
            if overflow > 0:
                del self._data[:overflow]

    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace")


def _drain_stream(stream, destination: _TailBuffer) -> None:
    try:
        while chunk := stream.read(64 * 1024):
            destination.append(chunk)
    except (OSError, ValueError):
        pass
    finally:
        stream.close()


def _run_process(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path | None,
    timeout_seconds: float,
) -> _ProcessOutcome:
    started = time.perf_counter()
    peak_rss_bytes = 0
    timed_out = False
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        ),
        start_new_session=os.name != "nt",
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("candidate process output pipes were not created")

    stdout_tail = _TailBuffer()
    stderr_tail = _TailBuffer()
    readers = [
        threading.Thread(target=_drain_stream, args=(process.stdout, stdout_tail), daemon=True),
        threading.Thread(target=_drain_stream, args=(process.stderr, stderr_tail), daemon=True),
    ]
    for reader in readers:
        reader.start()

    tracked = psutil.Process(process.pid)
    descendants: set[psutil.Process] = set()
    try:
        while process.poll() is None:
            _remember_descendants(tracked, descendants)
            peak_rss_bytes = max(peak_rss_bytes, _process_tree_rss(tracked))
            elapsed_seconds = time.perf_counter() - started
            if elapsed_seconds >= timeout_seconds:
                timed_out = True
                break
            time.sleep(min(_PROCESS_POLL_SECONDS, timeout_seconds - elapsed_seconds))
        _remember_descendants(tracked, descendants)
        peak_rss_bytes = max(peak_rss_bytes, _process_tree_rss(tracked))
    finally:
        _terminate_process_tree(process, tracked, descendants)
    returncode = process.wait()
    for reader in readers:
        reader.join(timeout=0.5)
    for stream, reader in zip((process.stdout, process.stderr), readers):
        if reader.is_alive():
            stream.close()
            reader.join(timeout=0.5)

    return _ProcessOutcome(
        returncode=returncode,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        peak_rss_mb=peak_rss_bytes / (1024.0 * 1024.0),
        stdout_tail=stdout_tail.text(),
        stderr_tail=stderr_tail.text(),
        timed_out=timed_out,
    )


def _write_result(output_dir: Path, result: CandidateResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), allow_nan=False, indent=2, sort_keys=True)
    result_path = output_dir / _RESULT_FILENAME
    temporary_path = output_dir / f".{_RESULT_FILENAME}.tmp"
    try:
        temporary_path.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary_path, result_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_number(value: object, field: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return number


def _validate_worker_output_files(output_dir: Path, status: Status) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("worker output directory must be a real directory")
    entries = list(output_dir.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"worker output contains unsafe entry: {entry.name}")

    expected = {_RESULT_FILENAME, *_SUCCESS_FILENAMES} if status == "ok" else {_RESULT_FILENAME}
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError(f"worker output files do not match status {status}")


def _read_result(output_dir: Path, *, expected_name: str, source_shape: tuple[int, ...]) -> CandidateResult:
    result_path = output_dir / _RESULT_FILENAME
    if result_path.is_symlink() or not result_path.is_file():
        raise ValueError("worker did not produce a regular result.json")
    if result_path.stat().st_size > _MAX_RESULT_BYTES:
        raise ValueError("worker result.json is too large")

    payload = json.loads(
        result_path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict) or set(payload) != _RESULT_FIELDS:
        raise ValueError("worker result.json has an invalid schema")
    if payload["name"] != expected_name:
        raise ValueError(f"candidate result name {payload['name']!r} does not match {expected_name!r}")

    status = payload["status"]
    if status not in {"ok", "not_detected", "unavailable", "error"}:
        raise ValueError(f"invalid candidate status: {status!r}")
    confidence = _finite_number(payload["confidence"], "confidence", maximum=1.0)
    elapsed_ms = _finite_number(payload["elapsed_ms"], "elapsed_ms")
    peak_rss_mb = _finite_number(payload["peak_rss_mb"], "peak_rss_mb")
    backend = payload["backend"]
    error = payload["error"]
    diagnostics = payload["diagnostics"]
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError("candidate diagnostics must be a JSON object or null")

    corners: np.ndarray | None = None
    if status == "ok":
        if backend not in CPU_BACKENDS:
            raise ValueError("successful candidate did not declare a supported CPU backend")
        if error is not None:
            raise ValueError("successful candidate result must not contain an error")
        try:
            corners = np.asarray(payload["corners"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("successful candidate corners are invalid") from exc
        valid, reason = validate_quad(corners, source_shape)
        if not valid:
            raise ValueError(f"successful candidate corners are invalid: {reason}")
    else:
        if payload["corners"] is not None:
            raise ValueError("non-ok candidate result must not contain corners")
        if status == "not_detected":
            if backend not in CPU_BACKENDS or error is not None:
                raise ValueError("not_detected result must declare a CPU backend without an error")
        elif backend is not None or not isinstance(error, str) or not error:
            raise ValueError("failed candidate result has inconsistent backend or error fields")

    _validate_worker_output_files(output_dir, status)
    if status == "ok":
        overlay = cv2.imread(str(output_dir / "overlay.jpg"), cv2.IMREAD_COLOR)
        rectified = cv2.imread(str(output_dir / "rectified.jpg"), cv2.IMREAD_COLOR)
        if overlay is None or overlay.shape[:2] != source_shape[:2]:
            raise ValueError("successful candidate overlay is missing or invalid")
        if rectified is None or rectified.size == 0:
            raise ValueError("successful candidate rectified image is missing or invalid")

    return CandidateResult(
        name=expected_name,
        status=status,
        corners=corners,
        confidence=confidence,
        backend=backend,
        elapsed_ms=elapsed_ms,
        peak_rss_mb=peak_rss_mb,
        error=error,
        diagnostics=diagnostics,
    )


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _candidate_source_path(module: str, environment: Mapping[str, str]) -> Path:
    parts = module.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        raise AdapterContractError(f"invalid candidate module name: {module!r}")

    search_paths = environment.get("PYTHONPATH", "").split(os.pathsep)
    search_paths.extend(str(path) for path in sys.path)
    relative_path = Path(*parts).with_suffix(".py")
    for search_path in search_paths:
        root = Path(search_path or os.getcwd())
        candidate = root / relative_path
        if candidate.is_symlink():
            raise AdapterContractError("candidate module source must not be a symlink")
        if candidate.is_file():
            return candidate.resolve()
    raise AdapterUnavailable(f"could not locate source for candidate module {module}")


def _source_identifiers(tree: ast.AST) -> list[str]:
    identifiers: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.append(node.id)
        elif isinstance(node, ast.arg):
            identifiers.append(node.arg)
        elif isinstance(node, ast.Attribute):
            identifiers.append(node.attr)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            identifiers.append(node.arg)
        elif isinstance(node, ast.alias):
            identifiers.extend(node.name.split("."))
            if node.asname is not None:
                identifiers.append(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            identifiers.extend(node.module.split("."))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.append(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            identifiers.extend(node.names)
    return identifiers


def _is_forbidden_asset_path(value: str) -> bool:
    normalized = value.strip().casefold().replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    suffix = Path(filename).suffix
    is_path_like = (
        "/" in normalized
        or normalized.startswith((".", "~"))
        or suffix in _KNOWN_IMAGE_DOCUMENT_EXTENSIONS
    )
    if not is_path_like:
        return False

    components = [part for part in normalized.split("/") if part not in {"", ".", ".."}]
    if components and suffix:
        components[-1] = components[-1][: -len(suffix)]

    path_tokens: list[str] = []
    for component in components:
        canonical = component.strip(".").replace("-", "_").replace(" ", "_")
        path_tokens.extend(token for token in canonical.split("_") if token)

    return any(
        tuple(path_tokens[index : index + len(forbidden)]) == forbidden
        for forbidden in _FORBIDDEN_ASSET_TOKEN_SEQUENCES
        for index in range(len(path_tokens) - len(forbidden) + 1)
    )


def _validate_candidate_source(module: str, environment: Mapping[str, str]) -> None:
    """Reject accidental evaluator coupling; this is not an OS security sandbox."""
    source_path = _candidate_source_path(module, environment)
    source = source_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as error:
        raise AdapterContractError("candidate source is not valid Python") from error

    if any(
        identifier.casefold() in _FORBIDDEN_RUNTIME_IDENTIFIERS
        for identifier in _source_identifiers(tree)
    ):
        raise AdapterContractError("candidate source uses a forbidden inference input")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_forbidden_asset_path(node.value):
                raise AdapterContractError("candidate source uses a forbidden evaluator asset path")
        if isinstance(node, ast.Subscript):
            key = node.slice
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.casefold() in _FORBIDDEN_RUNTIME_IDENTIFIERS
            ):
                raise AdapterContractError("candidate source accesses a forbidden inference input")


def _prepare_artifact_target(artifacts_dir: Path, name: str) -> tuple[Path, Path]:
    if artifacts_dir.is_symlink():
        raise RuntimeError("artifacts directory must not be a symlink")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if artifacts_dir.is_symlink() or not artifacts_dir.is_dir():
        raise RuntimeError("artifacts path must be a real directory")

    root = artifacts_dir.resolve(strict=True)
    target = root / name
    if target.parent != root:
        raise RuntimeError("candidate artifact path escapes the artifacts directory")
    if target.is_symlink():
        raise RuntimeError("candidate artifact directory must not be a symlink")
    if target.exists() and not target.is_dir():
        raise RuntimeError("candidate artifact path is not a directory")
    if target.exists():
        allowed = {_RESULT_FILENAME, *_SUCCESS_FILENAMES}
        for entry in target.iterdir():
            if entry.is_symlink():
                raise RuntimeError(f"candidate artifact contains a symlink: {entry.name}")
            if not entry.is_file() or entry.name not in allowed:
                raise RuntimeError(f"candidate artifact contains an unexpected entry: {entry.name}")
    if target.resolve(strict=False).parent != root:
        raise RuntimeError("candidate artifact path resolves outside the artifacts directory")
    return root, target


def _quarantine_candidate_directory(artifacts_dir: Path, name: str) -> tuple[Path, Path]:
    if artifacts_dir.is_symlink() or not artifacts_dir.is_dir():
        raise RuntimeError("artifacts path must be a real directory")
    root = artifacts_dir.resolve(strict=True)
    target = root / name
    if target.parent != root or target.is_symlink() or not target.is_dir():
        raise RuntimeError("unsafe candidate artifact cannot be quarantined")
    if target.resolve(strict=True).parent != root:
        raise RuntimeError("candidate artifact path resolves outside the artifacts directory")

    quarantine = root / f".{name}-quarantine-{uuid.uuid4().hex}"
    os.replace(target, quarantine)
    return root, target


def _remove_controlled_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("refusing to remove an unsafe artifact directory")
    entries = list(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise RuntimeError("refusing to remove unsafe artifact contents")
    for entry in entries:
        entry.unlink()
    path.rmdir()


def _publish_result(
    root: Path,
    target: Path,
    result: CandidateResult,
    worker_output: Path | None,
) -> None:
    _, checked_target = _prepare_artifact_target(root, result.name)
    if checked_target != target:
        raise RuntimeError("candidate artifact target changed during execution")

    publication = Path(tempfile.mkdtemp(prefix=f".{result.name}-publish-", dir=root))
    backup: Path | None = None
    try:
        if result.status == "ok":
            if worker_output is None:
                raise RuntimeError("successful result has no staged artifacts")
            for filename in _SUCCESS_FILENAMES:
                source = worker_output / filename
                if source.is_symlink() or not source.is_file():
                    raise RuntimeError(f"staged artifact is unsafe: {filename}")
                shutil.copyfile(source, publication / filename)
        _write_result(publication, result)

        if target.exists():
            backup = root / f".{result.name}-old-{uuid.uuid4().hex}"
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


def _write_error_manifest(root: Path, name: str, result: CandidateResult) -> None:
    manifest = Path(tempfile.mkdtemp(prefix=f".{name}-error-", dir=root))
    _write_result(manifest, result)


def _write_detection_artifacts(
    output_dir: Path,
    image: np.ndarray,
    corners: np.ndarray,
    model_dir: Path,
    cpu_threads: int,
) -> dict[str, object]:
    overlay = image.copy()
    cv2.polylines(overlay, [np.rint(corners).astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    rectified = warp_document(image, corners)
    orientation = orient_document(rectified, model_dir, cpu_threads)
    overlay_temporary = output_dir / ".overlay.tmp.jpg"
    rectified_temporary = output_dir / ".rectified.tmp.jpg"
    try:
        if not cv2.imwrite(str(overlay_temporary), overlay):
            raise OSError("could not write overlay.jpg")
        if not cv2.imwrite(str(rectified_temporary), orientation.image):
            raise OSError("could not write rectified.jpg")
        os.replace(overlay_temporary, output_dir / "overlay.jpg")
        os.replace(rectified_temporary, output_dir / "rectified.jpg")
    finally:
        overlay_temporary.unlink(missing_ok=True)
        rectified_temporary.unlink(missing_ok=True)
    return orientation.prediction.diagnostics()


def _merge_orientation_diagnostics(
    adapter_diagnostics: dict[str, object] | None,
    orientation_diagnostics: dict[str, object],
) -> dict[str, object]:
    merged = dict(adapter_diagnostics or {})
    if "orientation" in merged:
        raise AdapterContractError(
            "adapter diagnostics reserve the 'orientation' key for the shared model"
        )
    merged["orientation"] = orientation_diagnostics
    return merged


def _run_candidate(
    *,
    name: str,
    module: str,
    image_path: Path,
    model_dir: Path,
    output_dir: Path,
    cpu_threads: int,
) -> CandidateResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status: Status
    corners: np.ndarray | None = None
    confidence = 0.0
    backend: str | None = None
    error_text: str | None = None
    diagnostics: dict[str, object] | None = None

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode source image: {image_path}")

        entrypoint = load_adapter(module)
        output = entrypoint(image, model_dir, cpu_threads)
        if not isinstance(output, AdapterOutput):
            raise AdapterContractError("adapter run must return AdapterOutput with an explicit CPU backend")

        confidence = output.confidence
        backend = output.backend
        diagnostics = output.diagnostics
        if output.corners is None:
            status = "not_detected"
        else:
            corners = order_quad(output.corners)
            valid, reason = validate_quad(corners, image.shape)
            if not valid:
                raise ValueError(f"adapter returned an invalid quadrilateral: {reason}")
            orientation_diagnostics = _write_detection_artifacts(
                output_dir,
                image,
                corners,
                model_dir,
                cpu_threads,
            )
            diagnostics = _merge_orientation_diagnostics(
                diagnostics,
                orientation_diagnostics,
            )
            status = "ok"
    except (AdapterUnavailable, ModuleNotFoundError, ImportError) as error:
        status = "unavailable"
        corners = None
        confidence = 0.0
        backend = None
        error_text = _error_text(error)
    except Exception as error:
        status = "error"
        corners = None
        confidence = 0.0
        backend = None
        error_text = _error_text(error)

    result = CandidateResult(
        name=name,
        status=status,
        corners=corners,
        confidence=confidence,
        backend=backend,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        peak_rss_mb=0.0,
        error=error_text,
        diagnostics=diagnostics,
    )
    _write_result(output_dir, result)
    return result


def _fallback_result(
    name: str,
    error: BaseException,
    elapsed_ms: float,
    peak_rss_mb: float,
) -> CandidateResult:
    return CandidateResult(
        name=name,
        status="error",
        corners=None,
        confidence=0.0,
        backend=None,
        elapsed_ms=elapsed_ms,
        peak_rss_mb=peak_rss_mb,
        error=_error_text(error),
    )


def _run_benchmark_candidate(
    *,
    name: str,
    module: str,
    image_path: Path,
    model_dir: Path,
    artifacts_dir: Path,
    cpu_threads: int,
    timeout_seconds: float,
) -> CandidateResult:
    started = time.perf_counter()
    peak_rss_mb = 0.0
    try:
        root, target = _prepare_artifact_target(artifacts_dir, name)
    except Exception as error:
        result = _fallback_result(
            name,
            error,
            (time.perf_counter() - started) * 1000.0,
            peak_rss_mb,
        )
        try:
            root, target = _quarantine_candidate_directory(artifacts_dir, name)
            try:
                _publish_result(root, target, result, None)
            except Exception as publication_error:
                combined = RuntimeError(
                    f"{result.error}; artifact publication failed: {_error_text(publication_error)}"
                )
                result = _fallback_result(
                    name,
                    combined,
                    (time.perf_counter() - started) * 1000.0,
                    peak_rss_mb,
                )
                _write_error_manifest(root, name, result)
        except Exception:
            pass
        return result

    worker_output: Path | None = None
    try:
        environment = _cpu_environment(cpu_threads)
        _validate_candidate_source(module, environment)
        with tempfile.TemporaryDirectory(prefix=f"openscaner-{name}-") as staging_text:
            staging = Path(staging_text)
            suffix = image_path.suffix if image_path.suffix else ".image"
            staged_image = staging / f"source{suffix}"
            shutil.copyfile(image_path, staged_image)
            source_image = cv2.imread(str(staged_image), cv2.IMREAD_COLOR)
            if source_image is None:
                raise ValueError(f"could not decode source image: {image_path}")

            worker_output = staging / "output"
            worker_output.mkdir()
            command = build_candidate_command(
                name=name,
                module=module,
                image_path=staged_image,
                model_dir=model_dir.resolve(strict=False),
                output_dir=worker_output,
                cpu_threads=cpu_threads,
            )
            completed = _run_process(
                command,
                environment=environment,
                cwd=staging,
                timeout_seconds=timeout_seconds,
            )
            peak_rss_mb = completed.peak_rss_mb
            if completed.timed_out:
                raise TimeoutError(f"candidate process timed out after {timeout_seconds:g} seconds")
            if completed.returncode != 0:
                detail = completed.stderr_tail.strip() or completed.stdout_tail.strip() or "no diagnostic output"
                raise RuntimeError(f"candidate process exited {completed.returncode}: {detail}")

            result = _read_result(
                worker_output,
                expected_name=name,
                source_shape=source_image.shape,
            )
            result = replace(
                result,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                peak_rss_mb=peak_rss_mb,
            )
            _publish_result(root, target, result, worker_output)
            return result
    except Exception as error:
        result = _fallback_result(
            name,
            error,
            (time.perf_counter() - started) * 1000.0,
            peak_rss_mb,
        )
        try:
            _publish_result(root, target, result, None)
        except Exception as publication_error:
            combined = RuntimeError(f"{result.error}; artifact publication failed: {_error_text(publication_error)}")
            result = _fallback_result(
                name,
                combined,
                (time.perf_counter() - started) * 1000.0,
                peak_rss_mb,
            )
        return result


def run_benchmark(
    *,
    image_path: Path,
    model_dir: Path,
    artifacts_dir: Path,
    candidates: Mapping[str, str] | None = None,
    cpu_threads: int = 1,
    timeout_seconds: float = 300.0,
) -> dict[str, CandidateResult]:
    """Run every selected candidate in a fresh CPU-only subprocess."""
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and greater than zero")
    selected = dict(discover_adapters() if candidates is None else candidates)
    results: dict[str, CandidateResult] = {}

    for name, module in selected.items():
        if not name or Path(name).name != name or name in {".", ".."}:
            results[name] = _fallback_result(
                name,
                ValueError(f"invalid candidate name: {name!r}"),
                0.0,
                0.0,
            )
            continue
        results[name] = _run_benchmark_candidate(
            name=name,
            module=module,
            image_path=image_path,
            model_dir=model_dir,
            artifacts_dir=artifacts_dir,
            cpu_threads=cpu_threads,
            timeout_seconds=timeout_seconds,
        )
    return results


def _candidate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m openscaner.benchmark _candidate")
    parser.add_argument("--name", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, required=True)
    return parser


def _benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document-boundary candidates in isolated CPU processes")
    parser.add_argument("--input", dest="image_path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifacts", dest="artifacts_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="accepted for explicit reproducibility; always enabled")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["_candidate"]:
        options = _candidate_parser().parse_args(arguments[1:])
        _run_candidate(
            name=options.name,
            module=options.module,
            image_path=options.image,
            model_dir=options.model_dir,
            output_dir=options.output_dir,
            cpu_threads=options.cpu_threads,
        )
        return 0

    parser = _benchmark_parser()
    options = parser.parse_args(arguments)
    discovered = discover_adapters()
    if options.all:
        selected = discovered
    elif options.candidate:
        unknown = sorted(set(options.candidate) - discovered.keys())
        if unknown:
            parser.error(f"unknown candidate(s): {', '.join(unknown)}")
        selected = {name: discovered[name] for name in options.candidate}
    else:
        parser.error("choose --all or at least one --candidate")

    run_benchmark(
        image_path=options.image_path,
        model_dir=options.model_dir,
        artifacts_dir=options.artifacts_dir,
        candidates=selected,
        cpu_threads=options.cpu_threads,
        timeout_seconds=options.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
