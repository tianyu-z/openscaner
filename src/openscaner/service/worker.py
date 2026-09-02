from __future__ import annotations

import argparse
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from openscaner.service.processor import ProcessorError, process_image
from openscaner.service.schemas import JobStatus, ScanResult
from openscaner.service.settings import ServiceSettings, settings_from_env
from openscaner.service.store import ClaimedItem, JobRepository
from openscaner.service.storage import JobLayout, ServiceStorage


Processor = Callable[..., ScanResult]


def _layout_for_claim(storage: ServiceStorage, repo: JobRepository, item: ClaimedItem) -> JobLayout:
    detail = repo.get_job(item.job_id)
    if detail.storage_path:
        return storage.layout_from_root(Path(detail.storage_path))
    layout = storage.create_job_layout(item.job_id)
    repo.set_job_storage_path(item.job_id, str(layout.root))
    return layout


def run_one(
    repo: JobRepository,
    storage: ServiceStorage,
    settings: ServiceSettings,
    *,
    worker_id: str,
    processor: Processor = process_image,
) -> bool:
    item = repo.claim_next_item(worker_id=worker_id)
    if item is None:
        return False
    layout = _layout_for_claim(storage, repo, item)
    output_dir = storage.item_result_dir(layout, item.id)
    try:
        if item.input_path is None:
            raise RuntimeError("item input path is missing")
        result = processor(
            Path(item.input_path),
            model_dir=settings.model_dir,
            output_dir=output_dir,
            adapter=item.adapter,
            cpu_threads=1,
            timeout_seconds=settings.sync_timeout_seconds,
        )
        repo.complete_item(
            item.id,
            result_json_path=str(output_dir / "result.json"),
            overlay_path=str(output_dir / "overlay.jpg") if (output_dir / "overlay.jpg").exists() else None,
            rectified_path=str(output_dir / "rectified.jpg") if (output_dir / "rectified.jpg").exists() else None,
            corners_json=result.model_dump_json(include={"corners"}),
            confidence=result.confidence,
            fallback_used=result.fallback_used,
            adapter=result.adapter,
            model_sha256=result.model_sha256,
            policy_sha256=result.policy_sha256,
            elapsed_ms=result.elapsed_ms,
        )
        if not settings.keep_inputs:
            Path(item.input_path).unlink(missing_ok=True)
    except ProcessorError as error:
        repo.fail_item(item.id, code=error.code, message=error.message)
    except Exception as error:
        repo.fail_item(item.id, code="processing_failed", message=str(error))
    detail = repo.get_job(item.job_id)
    if detail.status in {JobStatus.SUCCEEDED, JobStatus.PARTIALLY_FAILED, JobStatus.FAILED}:
        zip_path = storage.create_results_zip(layout)
        repo.set_job_zip_path(item.job_id, str(zip_path))
    return True


def cleanup_expired_jobs(repo: JobRepository, storage: ServiceStorage) -> int:
    removed = 0
    for job in repo.expired_jobs():
        if job.storage_path:
            storage.delete_tree(Path(job.storage_path))
        repo.mark_job_expired(job.id)
        removed += 1
    return removed


def run_forever(settings: ServiceSettings | None = None) -> None:
    settings = settings or settings_from_env()
    repo = JobRepository(settings.storage_root / "openscaner.db")
    repo.initialize()
    repo.recover_stale_items(
        stale_seconds=settings.stale_item_seconds,
        max_attempts=settings.max_attempts,
    )
    storage = ServiceStorage(settings.storage_root)
    worker_id = f"worker_{uuid.uuid4().hex}"
    last_cleanup = 0.0
    while True:
        if time.monotonic() - last_cleanup > 3600:
            cleanup_expired_jobs(repo, storage)
            last_cleanup = time.monotonic()
        if not run_one(repo, storage, settings, worker_id=worker_id):
            time.sleep(settings.worker_poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the OpenScaner service worker")
    parser.parse_args(argv)
    run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
