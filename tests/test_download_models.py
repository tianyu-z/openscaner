from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "experiments": [],
                "shared_models": [],
                "models": entries,
                "schema_version": 3,
            }
        ),
        encoding="utf-8",
    )


def test_manifest_downloads_only_entries_with_download_urls(tmp_path: Path) -> None:
    module = load_module()
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [
            {
                "adapter": "third_party",
                "availability": "download_required",
                "local_filename": "third-party.pt",
                "download_url": "https://example.test/third-party.pt",
                "sha256": "0" * 64,
            },
            {
                "adapter": "locally_trained",
                "availability": "locally_trained",
                "local_filename": "own-model.pth",
                "sha256": "1" * 64,
            },
        ],
    )

    downloads = module.load_downloads(manifest)

    assert [item.filename for item in downloads] == ["third-party.pt"]
    assert downloads[0].url == "https://example.test/third-party.pt"


def test_project_manifest_downloads_only_third_party_weights() -> None:
    module = load_module()

    downloads = module.load_downloads(Path(__file__).resolve().parents[1] / "models" / "manifest.json")

    assert [item.filename for item in downloads] == [
        "mlsd_tiny_512_fp32.pth",
        "pp_lcnet_x1_0_doc_ori.onnx",
        "yolo11n-seg.pt",
        "mobile_sam.pt",
        "lcnet100_h_e_bifpn_256_fp32.onnx",
    ]


def test_existing_file_with_matching_sha_is_skipped(tmp_path: Path) -> None:
    module = load_module()
    payload = b"already downloaded"
    expected_sha = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "model.pt"
    target.write_bytes(payload)
    item = module.ModelDownload(
        section="models",
        name="model",
        filename="model.pt",
        url="https://example.test/model.pt",
        sha256=expected_sha,
        size_bytes=len(payload),
    )

    calls: list[str] = []
    result = module.ensure_model(item, tmp_path, downloader=lambda *_: calls.append("download"))

    assert result == "skipped"
    assert calls == []


def test_existing_file_with_wrong_sha_fails_without_force(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "model.pt").write_bytes(b"wrong bytes")
    item = module.ModelDownload(
        section="models",
        name="model",
        filename="model.pt",
        url="https://example.test/model.pt",
        sha256=hashlib.sha256(b"expected bytes").hexdigest(),
        size_bytes=None,
    )

    with pytest.raises(module.DownloadError, match="checksum mismatch"):
        module.ensure_model(item, tmp_path, downloader=lambda *_: None)
