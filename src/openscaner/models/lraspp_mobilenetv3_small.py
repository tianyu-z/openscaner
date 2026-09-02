"""LR-ASPP binary segmenter with an ImageNet MobileNetV3-Small backbone.

The head is torchvision's implementation of Lite R-ASPP.  MobileNetV3-Small
features at output strides 8 and 32 replace the Large backbone used by
torchvision's packaged segmentation model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.segmentation.lraspp import LRASPPHead


class LRASPPMobileNetV3Small(nn.Module):
    """Lite R-ASPP decoder over MobileNetV3-Small stride-8/32 features."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        self.backbone.classifier = nn.Identity()
        # Torchvision's classification default (0.01) needs hundreds of
        # updates before evaluation statistics reflect a small segmentation
        # dataset.  The standard PyTorch momentum keeps the official layers
        # and weights while making the exported evaluation model calibrated.
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.1
        self.classifier = LRASPPHead(
            low_channels=24,
            high_channels=576,
            num_classes=1,
            inter_channels=128,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_size = inputs.shape[-2:]
        features = inputs
        low: torch.Tensor | None = None
        for index, layer in enumerate(self.backbone.features):
            features = layer(features)
            if index == 3:
                low = features
        if low is None:  # pragma: no cover - fixed torchvision topology guard
            raise RuntimeError("MobileNetV3-Small did not produce its stride-8 feature")
        logits = self.classifier({"low": low, "high": features})
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


def build_model(*, pretrained: bool = False) -> LRASPPMobileNetV3Small:
    """Build the binary LR-ASPP model, optionally with ImageNet initialization."""
    return LRASPPMobileNetV3Small(pretrained=pretrained)
