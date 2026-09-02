from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def parse_api_keys(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _positive_float(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    storage_root: Path = Path("data")
    model_dir: Path = Path("models")
    default_adapter: str = "docaligner_pp_lcnet_fusion"
    allow_adapter_override: bool = False
    retention_days: int = 30
    keep_inputs: bool = True
    max_upload_mb: int = 50
    max_batch_items: int = 500
    sync_timeout_seconds: float = 60.0
    worker_poll_seconds: float = 1.0
    stale_item_seconds: int = 900
    max_attempts: int = 2
    web_password: str | None = None
    api_keys: tuple[str, ...] = ()
    auth_disabled: bool = False

    def __post_init__(self) -> None:
        _positive_int("retention_days", self.retention_days)
        _positive_int("max_upload_mb", self.max_upload_mb)
        _positive_int("max_batch_items", self.max_batch_items)
        _positive_float("sync_timeout_seconds", self.sync_timeout_seconds)
        _positive_float("worker_poll_seconds", self.worker_poll_seconds)
        _positive_int("stale_item_seconds", self.stale_item_seconds)
        _positive_int("max_attempts", self.max_attempts)


def settings_from_env(environ: dict[str, str] | None = None) -> ServiceSettings:
    env = os.environ if environ is None else environ
    return ServiceSettings(
        storage_root=Path(env.get("OPENSCANER_STORAGE_ROOT", "data")),
        model_dir=Path(env.get("OPENSCANER_MODEL_DIR", "models")),
        default_adapter=env.get("OPENSCANER_DEFAULT_ADAPTER", "docaligner_pp_lcnet_fusion"),
        allow_adapter_override=_bool_env(env.get("OPENSCANER_ALLOW_ADAPTER_OVERRIDE"), default=False),
        retention_days=int(env.get("OPENSCANER_RETENTION_DAYS", "30")),
        keep_inputs=_bool_env(env.get("OPENSCANER_KEEP_INPUTS"), default=True),
        max_upload_mb=int(env.get("OPENSCANER_MAX_UPLOAD_MB", "50")),
        max_batch_items=int(env.get("OPENSCANER_MAX_BATCH_ITEMS", "500")),
        sync_timeout_seconds=float(env.get("OPENSCANER_SYNC_TIMEOUT_SECONDS", "60")),
        worker_poll_seconds=float(env.get("OPENSCANER_WORKER_POLL_SECONDS", "1")),
        stale_item_seconds=int(env.get("OPENSCANER_STALE_ITEM_SECONDS", "900")),
        max_attempts=int(env.get("OPENSCANER_MAX_ATTEMPTS", "2")),
        web_password=env.get("OPENSCANER_WEB_PASSWORD"),
        api_keys=parse_api_keys(env.get("OPENSCANER_API_KEYS")),
        auth_disabled=_bool_env(env.get("OPENSCANER_AUTH_DISABLED"), default=False),
    )
