"""PP-LCNet-0.5 model for refining one local document corner."""

from __future__ import annotations

import os
import stat
import tempfile
import warnings
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import onnxruntime as ort
import timm
import torch
from torch import nn

from openscaner.models.corner_heatmap import (
    LOCAL_SOFT_ARGMAX_RADIUS,
    _deterministic_bilinear_upsample_2x,
    _download_verified_pretrained_state_dict,
    _normalize_decode_logits,
    _quadratic_border_pad,
)


_BACKBONE_ALIAS = "pp_lcnet_050"
_TIMM_MODEL_NAME = "lcnet_050"
_FEATURE_INDICES = (1, 2, 3, 4)
_EXPORT_BATCH_SIZE = 4
_EXPORT_INPUT_SHAPE = (_EXPORT_BATCH_SIZE, 3, 256, 256)
_EXPORT_INPUT_NAME = "patches"
_EXPORT_OUTPUT_NAMES = ("corner_logits", "edge_logits", "confidence")
_MAX_ONNX_BYTES = 32 * 1024 * 1024
_LOGIT_PARITY_RTOL = 5e-4
_LOGIT_PARITY_ATOL = 5e-4
_CONFIDENCE_PARITY_RTOL = 1e-4
_CONFIDENCE_PARITY_ATOL = 1e-5
_DECODED_CORNER_PARITY_ATOL = 2e-5


def differentiable_local_soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Decode one local corner logit map with an unclamped 5x5 soft argmax."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if (
        logits.ndim != 4
        or logits.shape[0] < 1
        or logits.shape[1] != 1
        or logits.shape[2] < 3
        or logits.shape[3] < 3
    ):
        raise ValueError(
            "logits must have shape N x 1 x H x W with N > 0 and H,W > 2"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")

    batch_size, _, height, width = logits.shape
    peak_indices = logits.flatten(2).argmax(dim=-1)
    decoded: list[torch.Tensor] = []
    window_size = 2 * LOCAL_SOFT_ARGMAX_RADIUS + 1
    for batch_index in range(batch_size):
        peak_index = int(peak_indices[batch_index, 0].item())
        peak_y, peak_x = divmod(peak_index, width)
        normalized = _normalize_decode_logits(logits[batch_index, 0])
        padded = _quadratic_border_pad(normalized)
        local_logits = padded[
            peak_y : peak_y + window_size,
            peak_x : peak_x + window_size,
        ]
        weights = torch.softmax(local_logits.flatten(), dim=0)
        yy, xx = torch.meshgrid(
            torch.arange(
                peak_y - LOCAL_SOFT_ARGMAX_RADIUS,
                peak_y + LOCAL_SOFT_ARGMAX_RADIUS + 1,
                device=logits.device,
                dtype=logits.dtype,
            ),
            torch.arange(
                peak_x - LOCAL_SOFT_ARGMAX_RADIUS,
                peak_x + LOCAL_SOFT_ARGMAX_RADIUS + 1,
                device=logits.device,
                dtype=logits.dtype,
            ),
            indexing="ij",
        )
        x = (weights * xx.flatten()).sum() / (width - 1)
        y = (weights * yy.flatten()).sum() / (height - 1)
        decoded.append(torch.stack((x, y)))

    coordinates = torch.stack(decoded)
    if not torch.isfinite(coordinates).all():
        raise RuntimeError("local soft argmax produced non-finite coordinates")
    return coordinates


class _RefinerDecoder(nn.Module):
    """Shared 32-channel top-down decoder with corner and incident-edge heads."""

    def __init__(self, feature_channels: list[int]) -> None:
        super().__init__()
        self.lateral_projections = nn.ModuleList(
            nn.Conv2d(channels, 32, kernel_size=1) for channels in feature_channels
        )
        self.corner_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
        )

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pyramid = self.lateral_projections[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            source_size = tuple(pyramid.shape[-2:])
            target_size = tuple(features[index].shape[-2:])
            if target_size != (2 * source_size[0], 2 * source_size[1]):
                raise ValueError(
                    "refiner top-down fusion requires adjacent feature maps to be "
                    f"exactly 2x spatial sizes; got {source_size} -> {target_size}"
                )
            pyramid = _deterministic_bilinear_upsample_2x(pyramid)
            pyramid = pyramid + self.lateral_projections[index](features[index])
        return self.corner_head(pyramid), self.edge_head(pyramid)


class LocalCornerRefiner(nn.Module):
    """Trainable PP-LCNet-0.5 local-corner refinement model."""

    def __init__(
        self,
        *,
        pretrained: bool,
        pretrained_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        create_options: dict[str, object] = {
            "pretrained": pretrained,
            "features_only": True,
            "out_indices": _FEATURE_INDICES,
        }
        if pretrained:
            if pretrained_state_dict is None:
                raise ValueError("pretrained model requires verified weight bytes")
            create_options["pretrained_cfg_overlay"] = {
                "state_dict": pretrained_state_dict,
                "source": "",
            }
        self.backbone = timm.create_model(_TIMM_MODEL_NAME, **create_options)
        if pretrained:
            for config_name in ("pretrained_cfg", "default_cfg"):
                config = getattr(self.backbone, config_name, None)
                if isinstance(config, dict):
                    config.pop("state_dict", None)
        self.decoder = _RefinerDecoder(list(self.backbone.feature_info.channels()))

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch tensor")
        if (
            inputs.ndim != 4
            or inputs.shape[0] < 1
            or tuple(inputs.shape[1:]) != (3, 256, 256)
        ):
            raise ValueError("inputs must have shape N x 3 x 256 x 256 with N > 0")
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype")
        floating_parameters = tuple(
            parameter
            for parameter in self.parameters()
            if parameter.is_floating_point()
        )
        if any(parameter.device != inputs.device for parameter in floating_parameters):
            raise ValueError(
                "inputs must use the same device as floating model parameters"
            )
        if any(parameter.dtype != inputs.dtype for parameter in floating_parameters):
            expected_dtype = floating_parameters[0].dtype
            raise TypeError(
                f"inputs dtype {inputs.dtype} must match model floating parameter "
                f"dtype {expected_dtype}"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("inputs must contain only finite values")

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if not torch.onnx.is_in_onnx_export():
            self._validate_inputs(inputs)
        corner_logits, edge_logits = self.decoder(self.backbone(inputs))
        confidence = torch.sigmoid(corner_logits.flatten(2).amax(dim=-1))
        if not torch.onnx.is_in_onnx_export() and (
            not torch.isfinite(corner_logits).all()
            or not torch.isfinite(edge_logits).all()
            or not torch.isfinite(confidence).all()
        ):
            raise RuntimeError("local corner refiner produced non-finite output")
        return {
            "corner_logits": corner_logits,
            "edge_logits": edge_logits,
            "confidence": confidence,
        }


class LocalCornerRefinerExportWrapper(nn.Module):
    """Convert the trainable mapping output to the fixed ONNX tuple contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(inputs)
        return (
            outputs["corner_logits"],
            outputs["edge_logits"],
            outputs["confidence"],
        )


def build_local_corner_refiner(pretrained: bool = True) -> LocalCornerRefiner:
    """Build the fixed PP-LCNet-0.5 local-corner refiner."""
    pretrained_state_dict = (
        _download_verified_pretrained_state_dict(_BACKBONE_ALIAS)
        if pretrained
        else None
    )
    return LocalCornerRefiner(
        pretrained=pretrained,
        pretrained_state_dict=pretrained_state_dict,
    )


def _read_bounded_regular_file(stream: BinaryIO) -> bytes:
    stream.flush()
    metadata = os.fstat(stream.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("exported ONNX stream is not a regular file")
    stream.seek(0)
    payload = stream.read(_MAX_ONNX_BYTES + 1)
    if not payload:
        raise RuntimeError("exported ONNX payload is empty")
    if len(payload) > _MAX_ONNX_BYTES:
        raise RuntimeError("exported ONNX payload exceeds the size limit")
    return payload


def _import_onnx() -> Any:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "install the ml dependency group to enable ONNX export"
        ) from error
    return onnx


def _validate_output_parity(
    outputs: list[np.ndarray],
    expected_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for name, output, expected in zip(
        _EXPORT_OUTPUT_NAMES, outputs, expected_outputs, strict=True
    ):
        rtol, atol = (
            (_CONFIDENCE_PARITY_RTOL, _CONFIDENCE_PARITY_ATOL)
            if name == "confidence"
            else (_LOGIT_PARITY_RTOL, _LOGIT_PARITY_ATOL)
        )
        try:
            np.testing.assert_allclose(
                output,
                expected.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as error:
            raise RuntimeError(
                f"ONNX Runtime output parity failed for {name}"
            ) from error

    actual_decoded = differentiable_local_soft_argmax(
        torch.from_numpy(np.asarray(outputs[0], dtype=np.float32))
    )
    expected_decoded = differentiable_local_soft_argmax(
        expected_outputs[0].detach().cpu().to(dtype=torch.float32)
    )
    try:
        torch.testing.assert_close(
            actual_decoded,
            expected_decoded,
            rtol=0.0,
            atol=_DECODED_CORNER_PARITY_ATOL,
        )
    except AssertionError as error:
        raise RuntimeError("ONNX Runtime decoded corner parity failed") from error


def _validate_onnx_contract(
    payload: bytes,
    fixed_input: torch.Tensor,
    expected_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    onnx = _import_onnx()
    try:
        document = onnx.load_model_from_string(payload)
        onnx.checker.check_model(document)
    except Exception as error:
        raise RuntimeError(
            "exported ONNX model failed structural validation"
        ) from error

    try:
        session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    except Exception as error:
        raise RuntimeError("ONNX Runtime session creation failed") from error
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX Runtime must activate CPUExecutionProvider only")

    expected_inputs = [(_EXPORT_INPUT_NAME, list(_EXPORT_INPUT_SHAPE), "tensor(float)")]
    expected_output_metadata = [
        ("corner_logits", [4, 1, 64, 64], "tensor(float)"),
        ("edge_logits", [4, 2, 64, 64], "tensor(float)"),
        ("confidence", [4, 1], "tensor(float)"),
    ]
    actual_inputs = [
        (value.name, value.shape, value.type) for value in session.get_inputs()
    ]
    actual_outputs = [
        (value.name, value.shape, value.type) for value in session.get_outputs()
    ]
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"exported ONNX input contract mismatch: {actual_inputs}")
    if actual_outputs != expected_output_metadata:
        raise RuntimeError(f"exported ONNX output contract mismatch: {actual_outputs}")

    try:
        outputs = session.run(None, {_EXPORT_INPUT_NAME: fixed_input.numpy()})
    except Exception as error:
        raise RuntimeError("ONNX Runtime fixed-input execution failed") from error
    if len(outputs) != len(_EXPORT_OUTPUT_NAMES) or any(
        not np.issubdtype(output.dtype, np.floating)
        or not np.isfinite(output).all()
        for output in outputs
    ):
        raise RuntimeError("ONNX Runtime produced invalid fixed-input outputs")
    _validate_output_parity(outputs, expected_outputs)


def export_refiner_onnx(model: nn.Module) -> bytes:
    """Export and validate the refiner's static batch-four ONNX contract."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch nn.Module")
    if any(module.training for module in model.modules()):
        raise ValueError("model and every submodule must be in eval mode")
    state_tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(value.device.type != "cpu" for value in state_tensors):
        raise ValueError("model parameters and buffers must be on CPU")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in state_tensors
    ):
        raise ValueError("floating model state must use torch.float32")
    _import_onnx()

    wrapper = LocalCornerRefinerExportWrapper(model)
    wrapper.training = False
    fixed_input = torch.linspace(
        -1.0,
        1.0,
        steps=_EXPORT_BATCH_SIZE * 3 * 256 * 256,
        dtype=torch.float32,
    ).reshape(_EXPORT_INPUT_SHAPE)
    with torch.no_grad():
        expected_outputs = wrapper(fixed_input)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".local-corner-refiner-", suffix=".onnx"
    )
    temporary_path = Path(temporary_name)
    try:
        initial_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(initial_metadata.st_mode):
            raise RuntimeError("ONNX temporary path is not a regular file")
        with os.fdopen(descriptor, "w+b", closefd=True) as stream:
            descriptor = -1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                torch.onnx.export(
                    wrapper,
                    fixed_input,
                    stream,
                    input_names=[_EXPORT_INPUT_NAME],
                    output_names=list(_EXPORT_OUTPUT_NAMES),
                    opset_version=17,
                    dynamo=False,
                    dynamic_axes=None,
                    training=torch.onnx.TrainingMode.EVAL,
                    do_constant_folding=True,
                )
            payload = _read_bounded_regular_file(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)

    _validate_onnx_contract(payload, fixed_input, expected_outputs)
    return payload


__all__ = [
    "LocalCornerRefiner",
    "LocalCornerRefinerExportWrapper",
    "build_local_corner_refiner",
    "differentiable_local_soft_argmax",
    "export_refiner_onnx",
]
