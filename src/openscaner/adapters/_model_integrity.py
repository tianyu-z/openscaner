"""Model-file integrity checks shared by pretrained adapters."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path

from openscaner.adapters.base import AdapterUnavailable


def verified_model_path(
    model_dir: str | Path,
    *,
    filename: str,
    expected_sha256: str,
    model_family: str,
) -> Path:
    """Return a regular model file only when its pinned SHA-256 matches."""
    model_path = Path(model_dir) / filename
    if model_path.is_symlink() or not model_path.is_file():
        raise AdapterUnavailable(f"{model_family} weight is absent: {filename}")

    digest = hashlib.sha256()
    try:
        with model_path.open("rb") as model_file:
            while chunk := model_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise AdapterUnavailable(
            f"{model_family} weight could not be read: {filename}"
        ) from error

    actual_sha256 = digest.hexdigest()
    if not secrets.compare_digest(actual_sha256, expected_sha256):
        raise AdapterUnavailable(
            f"{model_family} weight SHA-256 checksum mismatch: {filename}"
        )
    return model_path


def verified_model_bytes(
    model_dir: str | Path,
    *,
    filename: str,
    expected_sha256: str,
    expected_size_bytes: int,
    model_family: str,
) -> bytes:
    """Return the exact immutable bytes whose pinned SHA-256 was verified."""
    model_path = Path(model_dir) / filename
    descriptor = None
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        if no_follow:
            flags |= no_follow
        elif model_path.is_symlink():
            raise AdapterUnavailable(f"{model_family} weight is absent: {filename}")
        descriptor = os.open(model_path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AdapterUnavailable(
                f"{model_family} weight is not a regular file: {filename}"
            )
        if metadata.st_size != expected_size_bytes:
            raise AdapterUnavailable(
                f"{model_family} weight size mismatch: {filename}; "
                f"expected {expected_size_bytes} bytes, found {metadata.st_size}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as model_file:
            payload = model_file.read(expected_size_bytes + 1)
        if len(payload) != expected_size_bytes:
            raise AdapterUnavailable(
                f"{model_family} weight size changed while reading: {filename}"
            )
    except AdapterUnavailable:
        raise
    except FileNotFoundError as error:
        raise AdapterUnavailable(
            f"{model_family} weight is absent: {filename}"
        ) from error
    except OSError as error:
        raise AdapterUnavailable(
            f"{model_family} weight could not be read: {filename}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if not secrets.compare_digest(actual_sha256, expected_sha256):
        raise AdapterUnavailable(
            f"{model_family} weight SHA-256 checksum mismatch: {filename}"
        )
    return payload
