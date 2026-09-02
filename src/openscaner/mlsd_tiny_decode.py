"""CPU decoder for the pinned M-LSD Tiny output tensor.

Derived from NAVER's Apache-2.0 M-LSD PyTorch implementation at commit
2312205254e66911703decf775f626995d260f17.  See the bundled notice.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


MODEL_INPUT_SIZE = 512
MODEL_OUTPUT_SIZE = 256


def decode_lines(
    output: torch.Tensor,
    *,
    source_shape: tuple[int, int, int] | tuple[int, int],
    score_threshold: float,
    distance_threshold: float,
    topk: int = 200,
) -> np.ndarray:
    """Decode the center and displacement maps into source-pixel segments."""
    if output.device.type != "cpu":
        raise ValueError("M-LSD Tiny decoder accepts CPU tensors only")
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] < 5:
        raise ValueError("M-LSD Tiny output must have one batch and at least five channels")
    if output.shape[2:] != (MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE):
        raise ValueError("M-LSD Tiny output spatial shape must be 256 by 256")
    if topk < 1:
        raise ValueError("topk must be positive")
    height, width = source_shape[:2]
    if height < 1 or width < 1:
        raise ValueError("source image must have positive dimensions")

    centers = output[:, 0:1]
    heat = torch.sigmoid(centers)
    local_maximum = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
    heat = (heat * (local_maximum == heat)).reshape(-1)
    scores, indices = torch.topk(heat, min(topk, heat.numel()), largest=True)
    y = torch.div(indices, MODEL_OUTPUT_SIZE, rounding_mode="floor")
    x = torch.remainder(indices, MODEL_OUTPUT_SIZE)

    displacement = output[0, 1:5].permute(1, 2, 0)
    result: list[list[float]] = []
    source_x_scale = width / MODEL_OUTPUT_SIZE
    source_y_scale = height / MODEL_OUTPUT_SIZE
    for score, row, column in zip(scores.tolist(), y.tolist(), x.tolist(), strict=True):
        vector = displacement[row, column]
        distance = float(torch.linalg.vector_norm(vector[:2] - vector[2:]).item())
        if score <= score_threshold or distance <= distance_threshold:
            continue
        start_x, start_y, end_x, end_y = vector.tolist()
        result.append(
            [
                (column + start_x) * source_x_scale,
                (row + start_y) * source_y_scale,
                (column + end_x) * source_x_scale,
                (row + end_y) * source_y_scale,
            ]
        )
    return np.asarray(result, dtype=np.float32).reshape(-1, 4)
