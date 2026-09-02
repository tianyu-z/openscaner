from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from openscaner.benchmark import run_benchmark
from openscaner.service.schemas import ScanResult


@dataclass(frozen=True, slots=True)
class ProcessorError(RuntimeError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def _artifact_sha_from_manifest(model_dir: Path, adapter: str) -> tuple[str | None, str | None]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [item for item in manifest.get("models", []) if item.get("adapter") == adapter]
    if not entries:
        return None, None
    entry = entries[0]
    policy = (
        entry.get("fusion_policy")
        or entry.get("prompted_sam_policy")
        or entry.get("local_corner_refiner_policy")
        or {}
    )
    model_sha = entry.get("sha256")
    policy_sha = policy.get("sha256") if isinstance(policy, dict) else None
    return model_sha if isinstance(model_sha, str) else None, policy_sha if isinstance(policy_sha, str) else None


def process_image(
    input_path: Path,
    *,
    model_dir: Path,
    output_dir: Path,
    adapter: str,
    cpu_threads: int,
    timeout_seconds: float,
) -> ScanResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = output_dir / ".benchmark"
    if benchmark_root.exists():
        shutil.rmtree(benchmark_root)
    results = run_benchmark(
        image_path=input_path,
        model_dir=model_dir,
        artifacts_dir=benchmark_root,
        candidates={adapter: f"openscaner.adapters.{adapter}"},
        cpu_threads=cpu_threads,
        timeout_seconds=timeout_seconds,
    )
    candidate = results.get(adapter)
    if candidate is None:
        raise ProcessorError("processing_failed", "adapter did not return a result")
    candidate_dir = benchmark_root / adapter
    result_path = candidate_dir / "result.json"
    if result_path.is_file():
        shutil.copyfile(result_path, output_dir / "result.json")
    if candidate.status != "ok":
        raise ProcessorError("processing_failed", candidate.error or candidate.status)
    if not result_path.is_file():
        raise ProcessorError("processing_failed", "adapter did not write result.json")
    for filename in ("overlay.jpg", "rectified.jpg"):
        source = candidate_dir / filename
        if not source.is_file():
            raise ProcessorError("processing_failed", f"missing {filename}")
        shutil.copyfile(source, output_dir / filename)
    model_sha, policy_sha = _artifact_sha_from_manifest(model_dir, adapter)
    diagnostics = candidate.diagnostics or {}
    policy_sha = policy_sha or diagnostics.get("policy_sha256")
    fallback_used = diagnostics.get("fallback_used")
    return ScanResult(
        status=candidate.status,
        corners=candidate.corners.tolist() if candidate.corners is not None else None,
        confidence=candidate.confidence,
        fallback_used=fallback_used if isinstance(fallback_used, bool) else None,
        adapter=adapter,
        elapsed_ms=candidate.elapsed_ms,
        model_sha256=model_sha,
        policy_sha256=policy_sha if isinstance(policy_sha, str) else None,
        overlay_url=None,
        rectified_url=None,
        result_json_url="result.json",
        warnings=[],
    )
