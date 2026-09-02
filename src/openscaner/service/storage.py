from __future__ import annotations

import shutil
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_RESULT_FILENAMES = ("overlay.jpg", "rectified.jpg", "result.json")


@dataclass(frozen=True, slots=True)
class JobLayout:
    root: Path
    inputs: Path
    results: Path
    logs: Path
    downloads: Path


class ServiceStorage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _inside_root(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("path escapes storage root")
        return resolved

    def resolve_path(self, path: Path) -> Path:
        return self._inside_root(path)

    def create_job_layout(self, job_id: str) -> JobLayout:
        if Path(job_id).name != job_id or not job_id.startswith("job_"):
            raise ValueError("invalid job id")
        date = datetime.now(UTC).date().isoformat()
        root = self._inside_root(self.root / "jobs" / date / job_id)
        layout = self.layout_from_root(root)
        for path in (layout.inputs, layout.results, layout.logs, layout.downloads):
            path.mkdir(parents=True, exist_ok=True)
        return layout

    def layout_from_root(self, root: Path) -> JobLayout:
        root = self._inside_root(root)
        return JobLayout(
            root=root,
            inputs=root / "inputs",
            results=root / "results",
            logs=root / "logs",
            downloads=root / "downloads",
        )

    def input_path(self, layout: JobLayout, *, item_id: str, original_filename: str) -> Path:
        suffix = Path(original_filename).suffix.lower()
        if suffix not in _IMAGE_EXTENSIONS:
            raise ValueError("unsupported image extension")
        if Path(item_id).name != item_id or not item_id.startswith("item_"):
            raise ValueError("invalid item id")
        return self._inside_root(layout.inputs / f"{item_id}{suffix}")

    def item_result_dir(self, layout: JobLayout, item_id: str) -> Path:
        if Path(item_id).name != item_id or not item_id.startswith("item_"):
            raise ValueError("invalid item id")
        result_dir = self._inside_root(layout.results / item_id)
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def create_results_zip(self, layout: JobLayout) -> Path:
        layout.downloads.mkdir(parents=True, exist_ok=True)
        zip_path = self._inside_root(layout.downloads / "results.zip")
        temporary = self._inside_root(layout.downloads / f"results.{uuid.uuid4().hex}.tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for item_dir in sorted(layout.results.iterdir()):
                    if not item_dir.is_dir() or item_dir.is_symlink():
                        continue
                    for filename in _RESULT_FILENAMES:
                        path = item_dir / filename
                        if path.is_file() and not path.is_symlink():
                            archive.write(path, Path("results") / item_dir.name / filename)
            temporary.replace(zip_path)
        finally:
            temporary.unlink(missing_ok=True)
        return zip_path

    def delete_tree(self, path: Path) -> None:
        resolved = self._inside_root(path)
        if resolved.exists():
            shutil.rmtree(resolved)
