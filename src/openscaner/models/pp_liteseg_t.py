"""PyTorch port of the Apache-2.0 PaddleSeg PP-LiteSeg-T family.

PP-LiteSeg-T is the published lightweight configuration: an STDC1 encoder,
PPContextModule, and spatial Unified Attention Fusion Modules (UAFM).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tarfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_STDC1_URL = "https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz"
OFFICIAL_STDC1_ARCHIVE_SHA256 = (
    "245fe3c2e029c7ff271b4bd4e229f3849a50566fb6356ea0e56494146c6a9187"
)
OFFICIAL_STDC1_PARAMETERS_SHA256 = (
    "8de78e6996a74eaf84d2b561d62a4f9d0be0ace6936ec15f1ab0f91adb208e01"
)


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CatBottleneck(nn.Module):
    """Short-Term Dense Concatenate block used by STDCNet."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(in_channels, out_channels // 2, 1),
                ConvBNReLU(out_channels // 2, out_channels // 4, 3),
                ConvBNReLU(out_channels // 4, out_channels // 8, 3),
                ConvBNReLU(out_channels // 8, out_channels // 8, 3),
            ]
        )
        self.avd_layer = (
            nn.Sequential(
                nn.Conv2d(
                    out_channels // 2,
                    out_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_channels // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels // 2),
            )
            if stride == 2
            else nn.Identity()
        )
        self.skip = (
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            if stride == 2
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.branches[0](inputs)
        branch_input = self.avd_layer(first) if self.stride == 2 else first
        outputs = [self.skip(first)]
        for branch in self.branches[1:]:
            branch_input = branch(branch_input)
            outputs.append(branch_input)
        return torch.cat(outputs, dim=1)


class STDC1(nn.Module):
    """STDC1 encoder with the official [2, 2, 2] stage depths."""

    feature_channels = (32, 64, 256, 512, 1024)

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.ModuleList(
            [ConvBNReLU(3, 32, stride=2), ConvBNReLU(32, 64, stride=2)]
        )
        channels = ((64, 256), (256, 512), (512, 1024))
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    CatBottleneck(in_channels, out_channels, stride=2),
                    CatBottleneck(out_channels, out_channels, stride=1),
                )
                for in_channels, out_channels in channels
            ]
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        output = inputs
        for layer in self.stem:
            output = layer(output)
            features.append(output)
        for stage in self.stages:
            output = stage(output)
            features.append(output)
        return features


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _torch_key(paddle_key: str) -> str:
    parts = paddle_key.split(".", 2)
    if len(parts) != 3 or parts[0] != "features":
        raise ValueError(f"unexpected official STDC1 parameter: {paddle_key}")
    feature_index = int(parts[1])
    suffix = parts[2]
    if feature_index < 2:
        prefix = f"stem.{feature_index}"
        suffix = suffix.replace("conv.", "0.").replace("bn.", "1.")
    else:
        stage, block = divmod(feature_index - 2, 2)
        prefix = f"stages.{stage}.{block}"
        suffix = suffix.replace("conv_list.", "branches.")
        suffix = suffix.replace(".conv.", ".0.").replace(".bn.", ".1.")
    suffix = suffix.replace("._mean", ".running_mean")
    suffix = suffix.replace("._variance", ".running_var")
    return f"{prefix}.{suffix}"


def _official_stdc1_parameters() -> dict[str, object]:
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = checkpoint_dir / "PP_STDCNet1-model.pdparams"
    if parameters_path.is_file():
        payload = parameters_path.read_bytes()
        if _sha256(payload) == OFFICIAL_STDC1_PARAMETERS_SHA256:
            return pickle.loads(payload)

    archive_path = checkpoint_dir / "PP_STDCNet1.tar.gz"
    if not archive_path.is_file() or _sha256(archive_path.read_bytes()) != (
        OFFICIAL_STDC1_ARCHIVE_SHA256
    ):
        temporary = archive_path.with_name(f".{archive_path.name}.tmp")
        torch.hub.download_url_to_file(
            OFFICIAL_STDC1_URL,
            temporary,
            hash_prefix=OFFICIAL_STDC1_ARCHIVE_SHA256,
            progress=True,
        )
        os.replace(temporary, archive_path)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = archive.getmember("PP_STDCNet1/model.pdparams")
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("official STDC1 archive has no parameter payload")
        payload = stream.read()
    if _sha256(payload) != OFFICIAL_STDC1_PARAMETERS_SHA256:
        raise RuntimeError("official STDC1 parameter checksum mismatch")
    temporary = parameters_path.with_name(f".{parameters_path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, parameters_path)
    return pickle.loads(payload)


def _load_official_stdc1_weights(backbone: STDC1) -> None:
    paddle_state = _official_stdc1_parameters()
    converted = {
        _torch_key(key): torch.from_numpy(value)
        for key, value in paddle_state.items()
        if key.startswith("features.")
    }
    incompatible = backbone.load_state_dict(converted, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.endswith("num_batches_tracked")
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"official STDC1 parameters are incompatible: missing={missing}, "
            f"unexpected={unexpected}"
        )


class PPContextModule(nn.Module):
    """PP-LiteSeg's sum-fused pyramid pooling context module."""

    def __init__(self, in_channels: int = 1024, channels: int = 128) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    ConvBNReLU(in_channels, channels, kernel_size=1),
                )
                for size in (1, 2, 4)
            ]
        )
        self.output = ConvBNReLU(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        size = inputs.shape[-2:]
        pooled = [
            F.interpolate(
                stage(inputs), size=size, mode="bilinear", align_corners=False
            )
            for stage in self.stages
        ]
        return self.output(torch.stack(pooled).sum(dim=0))


class UAFM(nn.Module):
    """PP-LiteSeg spatial Unified Attention Fusion Module."""

    def __init__(
        self, low_channels: int, high_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.low_projection = ConvBNReLU(low_channels, high_channels)
        self.attention = nn.Sequential(
            ConvBNReLU(4, 2),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.output = ConvBNReLU(high_channels, out_channels)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        low = self.low_projection(low)
        high = F.interpolate(
            high, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        summary = torch.cat(
            (
                low.mean(dim=1, keepdim=True),
                low.amax(dim=1, keepdim=True),
                high.mean(dim=1, keepdim=True),
                high.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        attention = torch.sigmoid(self.attention(summary))
        return self.output(low * attention + high * (1.0 - attention))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__(
            ConvBNReLU(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
        )


class PPLiteSegT(nn.Module):
    """Binary PP-LiteSeg-T using the official STDC1 decoder widths."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = STDC1()
        if pretrained:
            _load_official_stdc1_weights(self.backbone)
        self.context = PPContextModule(1024, 128)
        self.fusion = nn.ModuleList(
            [UAFM(1024, 128, 128), UAFM(512, 128, 64), UAFM(256, 64, 32)]
        )
        self.segmentation_head = SegmentationHead(32, 32)
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        backbone_modules = set(self.backbone.modules())
        for module in self.modules():
            if module in backbone_modules:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_size = inputs.shape[-2:]
        features = self.backbone(inputs)[2:]
        high = self.context(features[-1])
        for low, fusion in zip(reversed(features), self.fusion):
            high = fusion(low, high)
        logits = self.segmentation_head(high)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


def build_model(*, pretrained: bool = False) -> PPLiteSegT:
    """Build PP-LiteSeg-T, optionally converting official STDC1 ImageNet weights."""
    return PPLiteSegT(pretrained=pretrained)
