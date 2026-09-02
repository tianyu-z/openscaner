"""Lightweight corner-heatmap document boundary models."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from pathlib import Path

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load as load_safetensors
from torch import nn


BACKBONE_ALIASES = {
    "mobilenetv3_small": "mobilenetv3_small_100",
    "pp_lcnet_050": "lcnet_050",
}
PRETRAINED_WEIGHT_SPECS = {
    "mobilenetv3_small": {
        "repo_id": "timm/mobilenetv3_small_100.lamb_in1k",
        "revision": "1824797e7887cbec1990e4adbd6675960a36c589",
        "filename": "model.safetensors",
        "sha256": "46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697",
        "size_bytes": 10_241_912,
        "model_tag": "lamb_in1k",
    },
    "pp_lcnet_050": {
        "repo_id": "timm/lcnet_050.ra2_in1k",
        "revision": "7d724d70bd001f453d812e733aa9b1931a69f6c2",
        "filename": "model.safetensors",
        "sha256": "d4f14612fea88f058cbd73123ebc8b5ed17d12a66d158dc93668af16d62fd3a9",
        "size_bytes": 7_560_712,
        "model_tag": "ra2_in1k",
    },
}
LOCAL_SOFT_ARGMAX_RADIUS = 2
LOCAL_SOFTMAX_LOGIT_RANGE = 80.0


def _download_verified_pretrained_state_dict(
    backbone_alias: str,
    *,
    downloader: Callable[..., str] | None = None,
) -> dict[str, torch.Tensor]:
    """Download, read once, and deserialize one exact immutable weight file."""
    spec = PRETRAINED_WEIGHT_SPECS[backbone_alias]
    download = hf_hub_download if downloader is None else downloader
    downloaded_path = Path(
        download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            filename=spec["filename"],
        )
    )
    try:
        # Hugging Face snapshot entries may be symlinks into its content-addressed
        # blob store. Resolve that expected cache indirection, then reject a
        # symlink at the actual object opened for verification.
        path = downloaded_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"pretrained weight is not a readable regular file: {downloaded_path}"
        ) from error

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"pretrained weight is not a regular file: {path}")
        if before.st_size != spec["size_bytes"]:
            raise ValueError(
                f"pretrained weight size mismatch: expected {spec['size_bytes']}, "
                f"got {before.st_size}"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            payload = stream.read(int(spec["size_bytes"]) + 1)
            after = os.fstat(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise ValueError("pretrained weight changed while it was being read")
    if len(payload) != spec["size_bytes"]:
        raise ValueError(
            f"pretrained weight size mismatch: expected {spec['size_bytes']}, "
            f"got {len(payload)}"
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != spec["sha256"]:
        raise ValueError(
            f"pretrained weight SHA-256 mismatch: expected {spec['sha256']}, "
            f"got {actual_sha256}"
        )
    return load_safetensors(payload)


def _deterministic_bilinear_upsample_2x(inputs: torch.Tensor) -> torch.Tensor:
    """Upsample an NCHW tensor 2x with separable align_corners=False bilinear."""
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch tensor")
    if inputs.ndim != 4:
        raise ValueError("inputs must be a 4D NCHW tensor")
    if inputs.shape[-2] == 0 or inputs.shape[-1] == 0:
        raise ValueError("inputs must have non-empty spatial dimensions")

    left = torch.cat((inputs[..., :1], inputs[..., :-1]), dim=-1)
    right = torch.cat((inputs[..., 1:], inputs[..., -1:]), dim=-1)
    horizontal = torch.stack(
        (0.25 * left + 0.75 * inputs, 0.75 * inputs + 0.25 * right),
        dim=-1,
    ).reshape(*inputs.shape[:-1], 2 * inputs.shape[-1])

    top = torch.cat((horizontal[..., :1, :], horizontal[..., :-1, :]), dim=-2)
    bottom = torch.cat(
        (horizontal[..., 1:, :], horizontal[..., -1:, :]), dim=-2
    )
    return torch.stack(
        (0.25 * top + 0.75 * horizontal, 0.75 * horizontal + 0.25 * bottom),
        dim=-2,
    ).reshape(
        *horizontal.shape[:-2],
        2 * horizontal.shape[-2],
        horizontal.shape[-1],
    )


def _normalize_decode_logits(values: torch.Tensor) -> torch.Tensor:
    """Promote and bound irrelevant softmax tails before extrapolation."""
    compute_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    promoted = values.to(dtype=compute_dtype)
    # Softmax is invariant to this shift. Values below -80 contribute less than
    # 2e-35 relative mass, so clipping them preserves coordinates while bounding
    # the subsequent quadratic linear combinations.
    shifted = promoted - promoted.amax()
    shifted = torch.where(
        torch.isneginf(shifted),
        shifted.new_full((), -LOCAL_SOFTMAX_LOGIT_RANGE),
        shifted,
    )
    if not torch.isfinite(shifted).all():
        raise RuntimeError("corner heatmap normalization produced non-finite values")
    return shifted.clamp(min=-LOCAL_SOFTMAX_LOGIT_RANGE, max=0.0)


def _quadratic_border_pad(values: torch.Tensor) -> torch.Tensor:
    """Extend a 2D logit surface by two samples using edge quadratics."""
    left_near = 3.0 * values[:, :1] - 3.0 * values[:, 1:2] + values[:, 2:3]
    left_far = 6.0 * values[:, :1] - 8.0 * values[:, 1:2] + 3.0 * values[:, 2:3]
    right_near = 3.0 * values[:, -1:] - 3.0 * values[:, -2:-1] + values[:, -3:-2]
    right_far = (
        6.0 * values[:, -1:] - 8.0 * values[:, -2:-1] + 3.0 * values[:, -3:-2]
    )
    padded = torch.cat((left_far, left_near, values, right_near, right_far), dim=1)

    top_near = 3.0 * padded[:1] - 3.0 * padded[1:2] + padded[2:3]
    top_far = 6.0 * padded[:1] - 8.0 * padded[1:2] + 3.0 * padded[2:3]
    bottom_near = 3.0 * padded[-1:] - 3.0 * padded[-2:-1] + padded[-3:-2]
    bottom_far = (
        6.0 * padded[-1:] - 8.0 * padded[-2:-1] + 3.0 * padded[-3:-2]
    )
    return torch.cat(
        (top_far, top_near, padded, bottom_near, bottom_far), dim=0
    )


def decode_corner_heatmaps(
    heatmaps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode TL, TR, BR, BL logits with fixed 5x5 local soft-argmax windows."""
    if not isinstance(heatmaps, torch.Tensor):
        raise TypeError("heatmaps must be a torch.Tensor")
    if (
        heatmaps.ndim != 4
        or heatmaps.shape[0] < 1
        or heatmaps.shape[1] != 4
        or heatmaps.shape[2] < 3
        or heatmaps.shape[3] < 3
    ):
        raise ValueError(
            "heatmaps must have shape N x 4 x H x W with N > 0 and H,W > 2"
        )
    if not heatmaps.is_floating_point():
        raise TypeError("heatmaps must use a floating-point dtype")
    if not torch.isfinite(heatmaps).all():
        raise ValueError("heatmaps must contain only finite values")

    batch_size, channels, height, width = heatmaps.shape
    flat = heatmaps.flatten(2)
    peak_indices = flat.argmax(dim=-1)
    peak_logits = flat.gather(dim=-1, index=peak_indices.unsqueeze(-1)).squeeze(-1)
    decoded_batches: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        decoded_channels: list[torch.Tensor] = []
        for channel_index in range(channels):
            peak_index = int(peak_indices[batch_index, channel_index].item())
            peak_y, peak_x = divmod(peak_index, width)
            normalized = _normalize_decode_logits(
                heatmaps[batch_index, channel_index]
            )
            padded = _quadratic_border_pad(normalized)
            window_size = 2 * LOCAL_SOFT_ARGMAX_RADIUS + 1
            local_logits = padded[
                peak_y : peak_y + window_size,
                peak_x : peak_x + window_size,
            ]
            weights = torch.softmax(local_logits.flatten(), dim=0)
            yy, xx = torch.meshgrid(
                torch.arange(
                    peak_y - LOCAL_SOFT_ARGMAX_RADIUS,
                    peak_y + LOCAL_SOFT_ARGMAX_RADIUS + 1,
                    device=heatmaps.device,
                    dtype=heatmaps.dtype,
                ),
                torch.arange(
                    peak_x - LOCAL_SOFT_ARGMAX_RADIUS,
                    peak_x + LOCAL_SOFT_ARGMAX_RADIUS + 1,
                    device=heatmaps.device,
                    dtype=heatmaps.dtype,
                ),
                indexing="ij",
            )
            x = (weights * xx.flatten()).sum() / (width - 1)
            y = (weights * yy.flatten()).sum() / (height - 1)
            decoded_channels.append(torch.stack((x, y)))
        decoded_batches.append(torch.stack(decoded_channels))

    corners = torch.stack(decoded_batches).clamp(0.0, 1.0)
    confidence = torch.sigmoid(peak_logits.to(dtype=corners.dtype)).mean(dim=1)
    confidence = confidence.clamp(0.0, 1.0)
    if not torch.isfinite(corners).all() or not torch.isfinite(confidence).all():
        raise RuntimeError("corner heatmap decoding produced non-finite output")
    return corners, confidence


class FPNDecoder(nn.Module):
    """Shared 32-channel top-down feature pyramid with two prediction heads."""

    def __init__(self, feature_channels: list[int]) -> None:
        super().__init__()
        self.lateral_projections = nn.ModuleList(
            nn.Conv2d(channels, 32, kernel_size=1) for channels in feature_channels
        )
        self.corner_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        pyramid = self.lateral_projections[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            source_size = tuple(pyramid.shape[-2:])
            target_size = tuple(features[index].shape[-2:])
            expected_size = (2 * source_size[0], 2 * source_size[1])
            if target_size != expected_size:
                raise ValueError(
                    "FPN top-down fusion requires adjacent feature maps to be "
                    f"exactly 2x spatial sizes; got {source_size} -> {target_size}"
                )
            pyramid = _deterministic_bilinear_upsample_2x(pyramid)
            pyramid = pyramid + self.lateral_projections[index](features[index])
        return {
            "corner_heatmaps": self.corner_head(pyramid),
            "mask_logits": self.mask_head(pyramid),
        }


class CornerHeatmapModel(nn.Module):
    """Corner and mask predictor backed by a timm feature extractor."""

    def __init__(
        self,
        backbone_alias: str,
        *,
        pretrained: bool,
        pretrained_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.backbone_alias = backbone_alias
        create_options: dict[str, object] = {
            "pretrained": pretrained,
            "features_only": True,
            "out_indices": (1, 2, 3, 4),
        }
        if pretrained:
            if pretrained_state_dict is None:
                raise ValueError("pretrained model requires verified weight bytes")
            create_options["pretrained_cfg_overlay"] = {
                "state_dict": pretrained_state_dict,
                "source": "",
            }
        self.backbone = timm.create_model(
            BACKBONE_ALIASES[backbone_alias], **create_options
        )
        if pretrained:
            for config_name in ("pretrained_cfg", "default_cfg"):
                config = getattr(self.backbone, config_name, None)
                if isinstance(config, dict):
                    config.pop("state_dict", None)
        self.decoder = FPNDecoder(list(self.backbone.feature_info.channels()))

    @property
    def checkpoint_metadata(self) -> dict[str, str]:
        """Return stable architecture metadata for serialized checkpoints."""
        return {"backbone_alias": self.backbone_alias}

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decoder(self.backbone(inputs))


def build_model(backbone: str, *, pretrained: bool = True) -> CornerHeatmapModel:
    """Build a corner-heatmap model for a supported public backbone alias."""
    if backbone not in BACKBONE_ALIASES:
        choices = ", ".join(sorted(BACKBONE_ALIASES))
        raise ValueError(
            f"unknown corner heatmap backbone {backbone!r}; expected one of: {choices}"
        )
    pretrained_state_dict = (
        _download_verified_pretrained_state_dict(backbone) if pretrained else None
    )
    return CornerHeatmapModel(
        backbone,
        pretrained=pretrained,
        pretrained_state_dict=pretrained_state_dict,
    )
