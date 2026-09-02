from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openscaner.service.schemas import (
    ErrorPayload,
    ItemStatus,
    ItemSummary,
    JobDetail,
    JobStatus,
    JobSummary,
    ScanResult,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class ClaimedItem:
    id: str
    job_id: str
    ordinal: int
    original_filename: str
    adapter: str
    status: ItemStatus
    attempt_count: int
    worker_id: str
    input_path: str | None


class JobRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version(version)
                SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    requested_by TEXT,
                    adapter TEXT NOT NULL,
                    total_count INTEGER NOT NULL,
                    done_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    storage_path TEXT,
                    zip_path TEXT,
                    retention_expires_at TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS items(
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    original_filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    heartbeat_at TEXT,
                    input_path TEXT,
                    overlay_path TEXT,
                    rectified_path TEXT,
                    result_json_path TEXT,
                    corners_json TEXT,
                    confidence REAL,
                    fallback_used INTEGER,
                    adapter TEXT NOT NULL,
                    model_sha256 TEXT,
                    policy_sha256 TEXT,
                    elapsed_ms REAL,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_items_status ON items(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_items_job ON items(job_id, ordinal);
                """
            )

    def create_job(
        self,
        *,
        adapter: str,
        requested_by: str | None,
        filenames: list[str],
        retention_days: int,
        initial_item_status: ItemStatus = ItemStatus.QUEUED,
        worker_id: str | None = None,
    ) -> JobSummary:
        if not filenames:
            raise ValueError("filenames must not be empty")
        if initial_item_status not in {ItemStatus.QUEUED, ItemStatus.RUNNING}:
            raise ValueError("initial_item_status must be queued or running")
        now = utc_now()
        expires = (datetime.now(UTC) + timedelta(days=retention_days)).isoformat()
        job_id = _id("job")
        job_status = JobStatus.RUNNING if initial_item_status == ItemStatus.RUNNING else JobStatus.QUEUED
        started_at = now if initial_item_status == ItemStatus.RUNNING else None
        attempt_count = 1 if initial_item_status == ItemStatus.RUNNING else 0
        heartbeat_at = now if initial_item_status == ItemStatus.RUNNING else None
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO jobs(
                    id,status,created_at,updated_at,started_at,requested_by,adapter,total_count,retention_expires_at
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (job_id, job_status.value, now, now, started_at, requested_by, adapter, len(filenames), expires),
            )
            for ordinal, filename in enumerate(filenames):
                db.execute(
                    """
                    INSERT INTO items(
                        id,job_id,ordinal,original_filename,status,created_at,started_at,
                        attempt_count,worker_id,heartbeat_at,adapter
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        _id("item"),
                        job_id,
                        ordinal,
                        filename,
                        initial_item_status.value,
                        now,
                        started_at,
                        attempt_count,
                        worker_id,
                        heartbeat_at,
                        adapter,
                    ),
                )
        return self.get_job_summary(job_id)

    def set_job_storage_path(self, job_id: str, storage_path: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET storage_path = ?, updated_at = ? WHERE id = ?",
                (storage_path, utc_now(), job_id),
            )

    def set_item_input_path(self, item_id: str, input_path: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE items SET input_path = ? WHERE id = ?",
                (input_path, item_id),
            )

    def list_jobs(self, *, limit: int = 100) -> list[JobSummary]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_job_summary_from_row(row) for row in rows]

    def get_job_summary(self, job_id: str) -> JobSummary:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _job_summary_from_row(row)

    def get_job(self, job_id: str) -> JobDetail:
        with self._connect() as db:
            job_row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            item_rows = db.execute(
                "SELECT * FROM items WHERE job_id = ? ORDER BY ordinal",
                (job_id,),
            ).fetchall()
        if job_row is None:
            raise KeyError(job_id)
        summary = _job_summary_from_row(job_row)
        return JobDetail(
            id=summary.id,
            status=summary.status,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            total_count=summary.total_count,
            done_count=summary.done_count,
            failed_count=summary.failed_count,
            storage_path=summary.storage_path,
            zip_path=summary.zip_path,
            download_url=summary.download_url,
            items=[_item_summary_from_row(row) for row in item_rows],
        )

    def get_item(self, item_id: str) -> ItemSummary:
        with self._connect() as db:
            row = db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(item_id)
        return _item_summary_from_row(row)

    def claim_next_item(self, *, worker_id: str) -> ClaimedItem | None:
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT items.*
                FROM items
                JOIN jobs ON jobs.id = items.job_id
                WHERE items.status = ?
                  AND jobs.status IN (?, ?)
                ORDER BY items.created_at, items.ordinal
                LIMIT 1
                """,
                (ItemStatus.QUEUED.value, JobStatus.QUEUED.value, JobStatus.RUNNING.value),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            db.execute(
                """
                UPDATE items
                SET status = ?, started_at = COALESCE(started_at, ?),
                    attempt_count = attempt_count + 1, worker_id = ?, heartbeat_at = ?
                WHERE id = ?
                """,
                (ItemStatus.RUNNING.value, now, worker_id, now, row["id"]),
            )
            db.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ?
                """,
                (JobStatus.RUNNING.value, now, now, row["job_id"]),
            )
            claimed = db.execute("SELECT * FROM items WHERE id = ?", (row["id"],)).fetchone()
            db.commit()
        return _claimed_item_from_row(claimed)

    def heartbeat(self, item_id: str, *, worker_id: str) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE items
                SET heartbeat_at = ?
                WHERE id = ? AND worker_id = ? AND status = ?
                """,
                (utc_now(), item_id, worker_id, ItemStatus.RUNNING.value),
            )

    def complete_item(
        self,
        item_id: str,
        *,
        result_json_path: str,
        overlay_path: str | None,
        rectified_path: str | None,
        corners_json: str | None,
        confidence: float,
        fallback_used: bool | None,
        adapter: str,
        model_sha256: str | None,
        policy_sha256: str | None,
        elapsed_ms: float,
    ) -> None:
        now = utc_now()
        with self._connect() as db:
            row = db.execute("SELECT job_id FROM items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise KeyError(item_id)
            db.execute(
                """
                UPDATE items
                SET status = ?, finished_at = ?, result_json_path = ?, overlay_path = ?,
                    rectified_path = ?, corners_json = ?, confidence = ?, fallback_used = ?,
                    adapter = ?, model_sha256 = ?, policy_sha256 = ?, elapsed_ms = ?,
                    error_code = NULL, error_message = NULL
                WHERE id = ?
                """,
                (
                    ItemStatus.SUCCEEDED.value,
                    now,
                    result_json_path,
                    overlay_path,
                    rectified_path,
                    corners_json,
                    confidence,
                    _bool_to_db(fallback_used),
                    adapter,
                    model_sha256,
                    policy_sha256,
                    elapsed_ms,
                    item_id,
                ),
            )
            self._refresh_job_aggregate(db, row["job_id"])

    def fail_item(self, item_id: str, *, code: str, message: str) -> None:
        now = utc_now()
        with self._connect() as db:
            row = db.execute("SELECT job_id FROM items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise KeyError(item_id)
            db.execute(
                """
                UPDATE items
                SET status = ?, finished_at = ?, error_code = ?, error_message = ?
                WHERE id = ?
                """,
                (ItemStatus.FAILED.value, now, code, message, item_id),
            )
            self._refresh_job_aggregate(db, row["job_id"])

    def set_job_zip_path(self, job_id: str, zip_path: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET zip_path = ?, updated_at = ? WHERE id = ?",
                (zip_path, utc_now(), job_id),
            )

    def recover_stale_items(self, *, stale_seconds: int, max_attempts: int) -> int:
        cutoff = (datetime.now(UTC) - timedelta(seconds=stale_seconds)).isoformat()
        recovered = 0
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT id, job_id, attempt_count
                FROM items
                WHERE status = ?
                  AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                """,
                (ItemStatus.RUNNING.value, cutoff),
            ).fetchall()
            for row in rows:
                if row["attempt_count"] < max_attempts:
                    db.execute(
                        """
                        UPDATE items
                        SET status = ?, worker_id = NULL, heartbeat_at = NULL
                        WHERE id = ?
                        """,
                        (ItemStatus.QUEUED.value, row["id"]),
                    )
                else:
                    db.execute(
                        """
                        UPDATE items
                        SET status = ?, finished_at = ?, error_code = ?, error_message = ?
                        WHERE id = ?
                        """,
                        (
                            ItemStatus.FAILED.value,
                            utc_now(),
                            "processing_failed",
                            "worker heartbeat expired",
                            row["id"],
                        ),
                    )
                    self._refresh_job_aggregate(db, row["job_id"])
                recovered += 1
        return recovered

    def expired_jobs(self, *, now: datetime | None = None) -> list[JobSummary]:
        value = (now or datetime.now(UTC)).isoformat()
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE retention_expires_at <= ? AND status != ? ORDER BY retention_expires_at",
                (value, JobStatus.EXPIRED.value),
            ).fetchall()
        return [_job_summary_from_row(row) for row in rows]

    def mark_job_expired(self, job_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, finished_at = COALESCE(finished_at, ?) WHERE id = ?",
                (JobStatus.EXPIRED.value, utc_now(), utc_now(), job_id),
            )

    def delete_job(self, job_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def _refresh_job_aggregate(self, db: sqlite3.Connection, job_id: str) -> None:
        counts = db.execute(
            """
            SELECT
                COUNT(*) AS total_count,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS done_count,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS failed_count
            FROM items
            WHERE job_id = ?
            """,
            (ItemStatus.SUCCEEDED.value, ItemStatus.FAILED.value, job_id),
        ).fetchone()
        total_count = int(counts["total_count"])
        done_count = int(counts["done_count"] or 0)
        failed_count = int(counts["failed_count"] or 0)
        finished_at = None
        if failed_count == 0 and done_count == total_count:
            status = JobStatus.SUCCEEDED
            finished_at = utc_now()
        elif done_count + failed_count == total_count and done_count > 0:
            status = JobStatus.PARTIALLY_FAILED
            finished_at = utc_now()
        elif done_count + failed_count == total_count:
            status = JobStatus.FAILED
            finished_at = utc_now()
        else:
            status = JobStatus.RUNNING
        db.execute(
            """
            UPDATE jobs
            SET status = ?, updated_at = ?, finished_at = COALESCE(?, finished_at),
                done_count = ?, failed_count = ?
            WHERE id = ?
            """,
            (status.value, utc_now(), finished_at, done_count, failed_count, job_id),
        )


def _job_summary_from_row(row: sqlite3.Row) -> JobSummary:
    zip_path = row["zip_path"]
    job_id = row["id"]
    return JobSummary(
        id=job_id,
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        total_count=row["total_count"],
        done_count=row["done_count"],
        failed_count=row["failed_count"],
        storage_path=row["storage_path"],
        zip_path=zip_path,
        download_url=f"/api/v1/jobs/{job_id}/download.zip" if zip_path else None,
    )


def _item_summary_from_row(row: sqlite3.Row) -> ItemSummary:
    result = None
    if row["result_json_path"]:
        result = ScanResult(
            status="ok",
            corners=_parse_corners(row["corners_json"]),
            confidence=row["confidence"] or 0.0,
            fallback_used=_bool_from_db(row["fallback_used"]),
            adapter=row["adapter"],
            elapsed_ms=row["elapsed_ms"] or 0.0,
            model_sha256=row["model_sha256"],
            policy_sha256=row["policy_sha256"],
            overlay_url=row["overlay_path"],
            rectified_url=row["rectified_path"],
            result_json_url=row["result_json_path"],
            warnings=[],
        )
    error = None
    if row["error_code"]:
        error = ErrorPayload(code=row["error_code"], message=row["error_message"] or "", item_id=row["id"])
    return ItemSummary(
        id=row["id"],
        job_id=row["job_id"],
        ordinal=row["ordinal"],
        original_filename=row["original_filename"],
        status=row["status"],
        result=result,
        error=error,
    )


def _claimed_item_from_row(row: sqlite3.Row) -> ClaimedItem:
    return ClaimedItem(
        id=row["id"],
        job_id=row["job_id"],
        ordinal=row["ordinal"],
        original_filename=row["original_filename"],
        adapter=row["adapter"],
        status=ItemStatus(row["status"]),
        attempt_count=row["attempt_count"],
        worker_id=row["worker_id"],
        input_path=row["input_path"],
    )


def _parse_corners(value: str | None) -> list[list[float]] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if isinstance(parsed, dict):
        parsed = parsed.get("corners")
    if parsed is None:
        return None
    return [[float(point[0]), float(point[1])] for point in parsed]


def _bool_to_db(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _bool_from_db(value: int | None) -> bool | None:
    if value is None:
        return None
    return bool(value)
