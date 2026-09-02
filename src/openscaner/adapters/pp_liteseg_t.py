"""CPU adapter for the locally trained PP-LiteSeg-T model."""

from __future__ import annotations

from openscaner.adapters._model_integrity import verified_model_bytes
from openscaner.adapters._trained_segmentation import (
    load_checkpoint,
    run_binary_segmenter,
)
from openscaner.models.pp_liteseg_t import build_model

MODEL_FILENAME = "pp_liteseg_t.pth"
EXPECTED_SHA256 = "61559e900dbc6c061cf49a6115f38a2377663851260f79da308fd6aa6eae3d7f"
EXPECTED_SIZE_BYTES = 32_243_427
MODEL_FAMILY = "PP-LiteSeg-T"
HORIZONTAL_FLIP_TTA = True


def _load_model(model_path):
    return load_checkpoint(
        model_path,
        model_name="pp_liteseg_t",
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
        horizontal_flip_tta=HORIZONTAL_FLIP_TTA,
    )
