from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from openscaner.service.processor import ProcessorError, process_image
from openscaner.service.schemas import ItemStatus, ItemSummary, JobDetail, JobSummary, ScanResult
from openscaner.service.settings import ServiceSettings, settings_from_env
from openscaner.service.store import JobRepository
from openscaner.service.storage import ServiceStorage


Processor = Callable[..., ScanResult]


def create_app(
    settings: ServiceSettings | None = None,
    *,
    processor: Processor = process_image,
) -> FastAPI:
    settings = settings or settings_from_env()
    repo = JobRepository(settings.storage_root / "openscaner.db")
    repo.initialize()
    storage = ServiceStorage(settings.storage_root)
    app = FastAPI(title="OpenScaner Service")
    app.state.settings = settings
    app.state.repo = repo
    app.state.storage = storage
    app.state.processor = processor

    def require_auth(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_openscaner_password: str | None = Header(default=None, alias="X-OpenScaner-Password"),
        authorization: str | None = Header(default=None),
    ) -> str:
        if settings.auth_disabled:
            return "dev"
        bearer = None
        if authorization and authorization.startswith("Bearer "):
            bearer = authorization.removeprefix("Bearer ").strip()
        supplied_key = x_api_key or bearer
        if supplied_key and _matches_any(supplied_key, settings.api_keys):
            return "api-key"
        if (
            x_openscaner_password
            and settings.web_password
            and secrets.compare_digest(x_openscaner_password, settings.web_password)
        ):
            return "web"
        raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "invalid credentials"})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/version")
    def version(_: str = Depends(require_auth)) -> dict[str, object]:
        manifest = settings.model_dir / "manifest.json"
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
        return {
            "default_adapter": settings.default_adapter,
            "manifest_sha256": manifest_sha,
        }

    @app.post("/api/v1/jobs", response_model=JobSummary, status_code=201)
    async def create_job(
        images: list[UploadFile] = File(...),
        adapter: str | None = Form(default=None),
        actor: str = Depends(require_auth),
    ) -> JobSummary:
        if not images or len(images) > settings.max_batch_items:
            raise HTTPException(status_code=400, detail={"code": "invalid_image", "message": "invalid batch size"})
        selected_adapter = _selected_adapter(settings, adapter)
        job = repo.create_job(
            adapter=selected_adapter,
            requested_by=actor,
            filenames=[image.filename or "upload.jpg" for image in images],
            retention_days=settings.retention_days,
        )
        layout = storage.create_job_layout(job.id)
        repo.set_job_storage_path(job.id, str(layout.root))
        try:
            detail = repo.get_job(job.id)
            for upload, item in zip(images, detail.items, strict=True):
                path = storage.input_path(layout, item_id=item.id, original_filename=item.original_filename)
                payload = await _read_limited(upload, settings.max_upload_mb * 1024 * 1024)
                path.write_bytes(payload)
                repo.set_item_input_path(item.id, str(path))
        except ValueError as error:
            storage.delete_tree(layout.root)
            repo.delete_job(job.id)
            raise HTTPException(status_code=400, detail={"code": "invalid_image", "message": str(error)}) from error
        except HTTPException:
            storage.delete_tree(layout.root)
            repo.delete_job(job.id)
            raise
        return repo.get_job_summary(job.id)

    @app.get("/api/v1/jobs", response_model=list[JobSummary])
    def list_jobs(_: str = Depends(require_auth)) -> list[JobSummary]:
        return repo.list_jobs()

    @app.get("/api/v1/jobs/{job_id}", response_model=JobDetail)
    def get_job(job_id: str, _: str = Depends(require_auth)) -> JobDetail:
        return _job_with_urls(_repo_job_or_404(repo, job_id))

    @app.get("/api/v1/jobs/{job_id}/items/{item_id}", response_model=ItemSummary)
    def get_item(job_id: str, item_id: str, _: str = Depends(require_auth)) -> ItemSummary:
        item = _repo_item_or_404(repo, job_id, item_id)
        return _item_with_urls(item)

    @app.get("/api/v1/jobs/{job_id}/items/{item_id}/overlay.jpg")
    def download_overlay(job_id: str, item_id: str, _: str = Depends(require_auth)) -> FileResponse:
        return _artifact_response(repo, storage, job_id, item_id, "overlay")

    @app.get("/api/v1/jobs/{job_id}/items/{item_id}/rectified.jpg")
    def download_rectified(job_id: str, item_id: str, _: str = Depends(require_auth)) -> FileResponse:
        return _artifact_response(repo, storage, job_id, item_id, "rectified")

    @app.get("/api/v1/jobs/{job_id}/items/{item_id}/result.json")
    def download_result_json(job_id: str, item_id: str, _: str = Depends(require_auth)) -> FileResponse:
        return _artifact_response(repo, storage, job_id, item_id, "result_json")

    @app.post("/api/v1/scan", response_model=ScanResult)
    async def scan(
        image: UploadFile = File(...),
        adapter: str | None = Form(default=None),
        actor: str = Depends(require_auth),
    ) -> ScanResult:
        selected_adapter = _selected_adapter(settings, adapter)
        job = repo.create_job(
            adapter=selected_adapter,
            requested_by=actor,
            filenames=[image.filename or "upload.jpg"],
            retention_days=settings.retention_days,
            initial_item_status=ItemStatus.RUNNING,
            worker_id="sync",
        )
        layout = storage.create_job_layout(job.id)
        repo.set_job_storage_path(job.id, str(layout.root))
        item = repo.get_job(job.id).items[0]
        try:
            input_path = storage.input_path(layout, item_id=item.id, original_filename=item.original_filename)
            input_path.write_bytes(await _read_limited(image, settings.max_upload_mb * 1024 * 1024))
            repo.set_item_input_path(item.id, str(input_path))
        except ValueError as error:
            storage.delete_tree(layout.root)
            repo.delete_job(job.id)
            raise HTTPException(status_code=400, detail={"code": "invalid_image", "message": str(error)}) from error
        output_dir = storage.item_result_dir(layout, item.id)
        try:
            result = processor(
                input_path,
                model_dir=settings.model_dir,
                output_dir=output_dir,
                adapter=selected_adapter,
                cpu_threads=1,
                timeout_seconds=settings.sync_timeout_seconds,
            )
        except ProcessorError as error:
            repo.fail_item(item.id, code=error.code, message=error.message)
            raise HTTPException(status_code=500, detail={"code": error.code, "message": error.message}) from error
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
        zip_path = storage.create_results_zip(layout)
        repo.set_job_zip_path(job.id, str(zip_path))
        if not settings.keep_inputs:
            input_path.unlink(missing_ok=True)
        return _with_urls(result, job.id, item.id)

    @app.get("/api/v1/jobs/{job_id}/download.zip")
    def download_job(job_id: str, _: str = Depends(require_auth)) -> FileResponse:
        detail = _repo_job_or_404(repo, job_id)
        if detail.zip_path is None:
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": "zip not ready"})
        path = storage.resolve_path(Path(detail.zip_path))
        if not path.is_file():
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": "zip not found"})
        return FileResponse(path, media_type="application/zip", filename=f"{job_id}-results.zip")

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, _: str = Depends(require_auth)) -> None:
        detail = _repo_job_or_404(repo, job_id)
        if detail.storage_path:
            storage.delete_tree(Path(detail.storage_path))
        repo.delete_job(job_id)

    static_dir = Path(__file__).with_name("static")
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def _matches_any(value: str, candidates: tuple[str, ...]) -> bool:
    return any(secrets.compare_digest(value, candidate) for candidate in candidates)


def _selected_adapter(settings: ServiceSettings, requested: str | None) -> str:
    if requested and requested != settings.default_adapter and not settings.allow_adapter_override:
        raise HTTPException(status_code=400, detail={"code": "invalid_image", "message": "adapter override disabled"})
    return requested or settings.default_adapter


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    payload = await upload.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail={"code": "invalid_image", "message": "upload too large"})
    return payload


def _repo_job_or_404(repo: JobRepository, job_id: str) -> JobDetail:
    try:
        return repo.get_job(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "job not found"}) from error


def _repo_item_or_404(repo: JobRepository, job_id: str, item_id: str) -> ItemSummary:
    try:
        item = repo.get_item(item_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "item not found"}) from error
    if item.job_id != job_id:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "item not found"})
    return item


def _job_with_urls(detail: JobDetail) -> JobDetail:
    items = [_item_with_urls(item) for item in detail.items]
    return detail.model_copy(update={"items": items})


def _item_with_urls(item: ItemSummary) -> ItemSummary:
    if item.result is None:
        return item
    return item.model_copy(update={"result": _with_urls(item.result, item.job_id, item.id)})


def _with_urls(result: ScanResult, job_id: str, item_id: str) -> ScanResult:
    base = f"/api/v1/jobs/{job_id}/items/{item_id}"
    return result.model_copy(
        update={
            "overlay_url": f"{base}/overlay.jpg",
            "rectified_url": f"{base}/rectified.jpg",
            "result_json_url": f"{base}/result.json",
        }
    )


def _artifact_response(
    repo: JobRepository,
    storage: ServiceStorage,
    job_id: str,
    item_id: str,
    artifact: str,
) -> FileResponse:
    item = _repo_item_or_404(repo, job_id, item_id)
    if item.result is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "artifact not ready"})
    path_text = {
        "overlay": item.result.overlay_url,
        "rectified": item.result.rectified_url,
        "result_json": item.result.result_json_url,
    }[artifact]
    if path_text is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "artifact not found"})
    path = storage.resolve_path(Path(path_text))
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "artifact not found"})
    media_type = "application/json" if artifact == "result_json" else "image/jpeg"
    return FileResponse(path, media_type=media_type)
