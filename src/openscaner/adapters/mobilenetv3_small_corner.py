"""CPU adapter for the frozen MobileNetV3-Small corner heatmap model."""

from __future__ import annotations

from openscaner.adapters._corner_heatmap import run_corner_heatmap


def run(image, model_dir, cpu_threads):
    return run_corner_heatmap(
        image,
        model_dir,
        cpu_threads,
        adapter_name="mobilenetv3_small_corner",
        architecture_alias="mobilenetv3_small",
        model_family="MobileNetV3-Small corner heatmap model",
        checkpoint_optional=False,
    )
