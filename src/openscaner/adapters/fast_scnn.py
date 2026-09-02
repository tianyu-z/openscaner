"""CPU adapter for the locally trained Fast-SCNN model."""

from __future__ import annotations

from openscaner.adapters._model_integrity import verified_model_bytes
from openscaner.adapters._trained_segmentation import (
    load_checkpoint,
    run_binary_segmenter,
)
from openscaner.models.fast_scnn import build_model

MODEL_FILENAME = "fast_scnn.pth"
EXPECTED_SHA256 = "073abd28291aabd1599d13d1704108aa13dbe41c84be90b73db471b18b60b175"
EXPECTED_SIZE_BYTES = 4_725_211
MODEL_FAMILY = "Fast-SCNN"


def _load_model(model_path):
    return load_checkpoint(
        model_path,
        model_name="fast_scnn",
        builder=lambda: build_model(pretrained=False),
    )


def run(image, model_dir, cpu_threads):
    model_bytes = verified_model_bytes(
        model_dir,
        filename=MODEL_FILENAME,
        expected_sha256=EXPECTED_SHA256,
        expected_size_bytes=EXPECTED_SIZE_BYTES,
        model_family=MODEL_FAMILY,
    )
    return run_binary_segmenter(
        image,
        model_path=model_bytes,
        cpu_threads=cpu_threads,
        loader=_load_model,
    )
