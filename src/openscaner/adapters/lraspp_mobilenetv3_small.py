"""CPU adapter for the locally trained LR-ASPP MobileNetV3-Small model."""

from __future__ import annotations

from openscaner.adapters._model_integrity import verified_model_bytes
from openscaner.adapters._trained_segmentation import (
    load_checkpoint,
    run_binary_segmenter,
)
from openscaner.models.lraspp_mobilenetv3_small import build_model

MODEL_FILENAME = "lraspp_mobilenetv3_small.pth"
EXPECTED_SHA256 = "62023053a48ce26567fb9ff40eaedbcd6955dab2bb9d0a64564307f5f50e3e98"
EXPECTED_SIZE_BYTES = 4_431_967
MODEL_FAMILY = "LR-ASPP MobileNetV3-Small"


def _load_model(model_path):
    return load_checkpoint(
        model_path,
        model_name="lraspp_mobilenetv3_small",
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
