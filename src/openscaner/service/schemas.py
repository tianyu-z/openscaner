from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIALLY_FAILED = "partially_failed"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"


class ItemStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorCode(StrEnum):
    INVALID_IMAGE = "invalid_image"
    MODEL_UNAVAILABLE = "model_unavailable"
    PROCESSING_FAILED = "processing_failed"
    TIMEOUT = "timeout"
    STORAGE_ERROR = "storage_error"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"


class ErrorPayload(BaseModel):
    code: ErrorCode
    message: str
    item_id: str | None = None
    recoverable: bool = False


class ScanResult(BaseModel):
    status: Literal["ok", "not_detected", "unavailable", "error"]
    corners: list[list[float]] | None
    confidence: float = Field(ge=0.0, le=1.0)
    fallback_used: bool | None = None
    adapter: str
    elapsed_ms: float = Field(ge=0.0)
    model_sha256: str | None = None
    policy_sha256: str | None = None
    overlay_url: str | None = None
    rectified_url: str | None = None
    result_json_url: str
    warnings: list[str] = Field(default_factory=list)
    error: ErrorPayload | None = None


class JobSummary(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    total_count: int
    done_count: int
    failed_count: int
    storage_path: str | None = Field(default=None, exclude=True)
    zip_path: str | None = Field(default=None, exclude=True)
    download_url: str | None = None


class ItemSummary(BaseModel):
    id: str
    job_id: str
    ordinal: int
    original_filename: str
    status: ItemStatus
    result: ScanResult | None = None
    error: ErrorPayload | None = None


class JobDetail(JobSummary):
    items: list[ItemSummary]
