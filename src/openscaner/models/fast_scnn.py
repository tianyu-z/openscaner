"""Fast-SCNN binary segmentation architecture.

This follows the paper's learning-to-downsample, global feature extractor,
feature fusion, and lightweight classifier stages.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            ConvBNReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
        )


class LearningToDownsample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(32, 48, stride=2),
            DepthwiseSeparableConv(48, 64, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LinearBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion: int = 6,
        stride: int = 1,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.layers = nn.Sequential(
            ConvBNReLU(in_channels, hidden_channels, kernel_size=1),
            ConvBNReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
            ),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layers(inputs)
        return inputs + output if self.use_residual else output


class PyramidPooling(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        pooled_channels = channels // 4
        self.projections = nn.ModuleList(
            [ConvBNReLU(channels, pooled_channels, kernel_size=1) for _ in range(4)]
        )
        self.output = ConvBNReLU(channels * 2, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        size = inputs.shape[-2:]
        pooled = [inputs]
        for bin_size, projection in zip((1, 2, 3, 6), self.projections):
            feature = F.adaptive_avg_pool2d(inputs, bin_size)
            feature = projection(feature)
            pooled.append(
                F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            )
        return self.output(torch.cat(pooled, dim=1))


def _bottleneck_stage(
    in_channels: int,
    out_channels: int,
    *,
    blocks: int,
    stride: int,
) -> nn.Sequential:
    layers = [LinearBottleneck(in_channels, out_channels, stride=stride)]
    layers.extend(
        LinearBottleneck(out_channels, out_channels) for _ in range(blocks - 1)
    )
    return nn.Sequential(*layers)


class GlobalFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.Sequential(
            _bottleneck_stage(64, 64, blocks=3, stride=2),
            _bottleneck_stage(64, 96, blocks=3, stride=2),
            _bottleneck_stage(96, 128, blocks=3, stride=1),
            PyramidPooling(128),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.stages(inputs)


class FeatureFusionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.low_depthwise = ConvBNReLU(128, 128, kernel_size=3, padding=1, groups=128)
        self.low_projection = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128)
        )
        self.high_projection = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        low = F.interpolate(
            low, size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        low = self.low_projection(self.low_depthwise(low))
        high = self.high_projection(high)
        return self.activation(high + low)


class Classifier(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 1, kernel_size=1),
        )


class FastSCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusionModule()
        self.classifier = Classifier()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_size = inputs.shape[-2:]
        high = self.learning_to_downsample(inputs)
        low = self.global_feature_extractor(high)
        fused = self.feature_fusion(high, low)
        logits = self.classifier(fused)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


def build_model(*, pretrained: bool = False) -> FastSCNN:
    """Build Fast-SCNN; the architecture has no official ImageNet checkpoint."""
    if pretrained:
        raise ValueError("Fast-SCNN has no official ImageNet checkpoint")
    return FastSCNN()
