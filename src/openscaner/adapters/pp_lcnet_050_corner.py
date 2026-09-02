"""CPU adapter for the frozen PP-LCNet-0.5 corner heatmap model."""

from __future__ import annotations

from openscaner.adapters._corner_heatmap import run_corner_heatmap


def run(image, model_dir, cpu_threads):
    return run_corner_heatmap(
        image,
        model_dir,
        cpu_threads,
        adapter_name="pp_lcnet_050_corner",
        architecture_alias="pp_lcnet_050",
        model_family="PP-LCNet-0.5 corner heatmap model",
        checkpoint_optional=False,
    )
