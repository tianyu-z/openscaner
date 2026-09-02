#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple


MANIFEST_SECTIONS = ("experiments", "shared_models", "models")
DEFAULT_MANIFEST = Path("models") / "manifest.json"
DEFAULT_MODEL_DIR = Path("models")
USER_AGENT = "OpenScaner model downloader"


class DownloadError(RuntimeError):
    """Raised when a model cannot be downloaded or verified."""


class ModelDownload(NamedTuple):
    section: str
    name: str
    filename: str
    url: str
    sha256: str | None
    size_bytes: int | None


Downloader = Callable[[str, Path], object]


def _entry_name(entry: dict[str, object]) -> str:
    for key in ("adapter", "name", "model_family", "local_filename"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _entry_size(entry: dict[str, object]) -> int | None:
    for key in ("size_bytes", "checkpoint_size_bytes"):
        value = entry.get(key)
        if isinstance(value, int):
            return value
    return None


def load_downloads(manifest_path: Path = DEFAULT_MANIFEST) -> list[ModelDownload]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloads: list[ModelDownload] = []
    for section in MANIFEST_SECTIONS:
        for entry in manifest.get(section, []):
            if not isinstance(entry, dict):
                continue
            url = entry.get("download_url")
            filename = entry.get("local_filename")
            if not isinstance(url, str) or not isinstance(filename, str):
                continue
            sha256 = entry.get("sha256")
            if sha256 is not None and not isinstance(sha256, str):
                raise DownloadError(f"{filename}: sha256 must be a string")
            downloads.append(
                ModelDownload(
                    section=section,
                    name=_entry_name(entry),
                    filename=filename,
                    url=url,
                    sha256=sha256,
                    size_bytes=_entry_size(entry),
                )
            )
    return downloads


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model(path: Path, item: ModelDownload) -> None:
    if item.size_bytes is not None and path.stat().st_size != item.size_bytes:
        raise DownloadError(
            f"{item.filename}: size mismatch "
            f"(expected {item.size_bytes}, got {path.stat().st_size})"
        )
    if item.sha256 is not None:
        actual = sha256_file(path)
        if actual.lower() != item.sha256.lower():
            raise DownloadError(
                f"{item.filename}: checksum mismatch "
                f"(expected {item.sha256}, got {actual})"
            )


def download_url(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def ensure_model(
    item: ModelDownload,
    model_dir: Path = DEFAULT_MODEL_DIR,
    *,
    downloader: Downloader = download_url,
    force: bool = False,
) -> str:
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / item.filename
    if target.exists() and not force:
        _verify_model(target, item)
        return "skipped"

    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        downloader(item.url, temporary)
        _verify_model(temporary, item)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def iter_selected(downloads: Iterable[ModelDownload], selected: set[str]) -> list[ModelDownload]:
    items = list(downloads)
    if not selected:
        return items
    return [
        item
        for item in items
        if item.filename in selected or item.name in selected
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download third-party OpenScaner model weights")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--force", action="store_true", help="replace existing downloaded files")
    parser.add_argument("--dry-run", action="store_true", help="show downloads without fetching files")
    parser.add_argument(
        "models",
        nargs="*",
        help="optional adapter names or filenames to download",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    downloads = iter_selected(load_downloads(args.manifest), set(args.models))
    if not downloads:
        raise DownloadError("no matching downloadable models found")

    for item in downloads:
        target = args.model_dir / item.filename
        if args.dry_run:
            print(f"would download {item.name}: {item.url} -> {target}")
            continue
        result = ensure_model(item, args.model_dir, force=args.force)
        print(f"{result} {item.name}: {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
