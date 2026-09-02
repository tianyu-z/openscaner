"""Minimal M-LSD Tiny network port for CPU inference.

Derived from NAVER's Apache-2.0 M-LSD PyTorch implementation at commit
2312205254e66911703decf775f626995d260f17.  See the bundled notice.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BlockTypeA(nn.Module):
    def __init__(self, in_c1: int, in_c2: int, out_c1: int, out_c2: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b = F.interpolate(self.conv1(b), scale_factor=2.0, mode="bilinear", align_corners=True)
        return torch.cat((self.conv2(a), b), dim=1)


class BlockTypeB(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(tensor) + tensor)


class BlockTypeC(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(self.conv1(tensor)))


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        self.stride = stride
        padding = 0 if stride == 2 else (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            tensor = F.pad(tensor, (0, 1, 0, 1), "constant", 0)
        for module in self:
            tensor = module(tensor)
        return tensor


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("stride must be one or two")
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.conv(tensor)
        return tensor + output if self.use_res_connect else output


class MobileNetV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_channel = 32
        setting = ([1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2])
        features: list[nn.Module] = [ConvBNReLU(4, input_channel, stride=2)]
        for expand, channels, count, stride in setting:
            for index in range(count):
                features.append(
                    InvertedResidual(input_channel, channels, stride if index == 0 else 1, expand)
                )
                input_channel = channels
        self.features = nn.Sequential(*features)
        self.fpn_selected = (3, 6, 10)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected: list[torch.Tensor] = []
        for index, feature in enumerate(self.features):
            tensor = feature(tensor)
            if index in self.fpn_selected:
                selected.append(tensor)
            if index == self.fpn_selected[-1]:
                break
        return selected[0], selected[1], selected[2]


class MLSDTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = MobileNetV2()
        self.block12 = BlockTypeA(32, 64, 64, 64)
        self.block13 = BlockTypeB(128, 64)
        self.block14 = BlockTypeA(24, 64, 32, 32)
        self.block15 = BlockTypeB(64, 64)
        self.block16 = BlockTypeC(64, 16)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        c2, c3, c4 = self.backbone(tensor)
        tensor = self.block13(self.block12(c3, c4))
        tensor = self.block15(self.block14(c2, tensor))
        tensor = self.block16(tensor)[:, 7:, :, :]
        return F.interpolate(tensor, scale_factor=2.0, mode="bilinear", align_corners=True)


def build_model() -> MLSDTiny:
    """Build the exact Tiny topology expected by the pinned checkpoint."""
    return MLSDTiny()
