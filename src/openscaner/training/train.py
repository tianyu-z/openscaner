"""Shared reproducible training entrypoint for lightweight segmenters."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from openscaner.training.data import (
    DATASET_SEED,
    IMAGE_SIZE,
    SYNTHETIC_DATA_LIMITATION,
    TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
    build_datasets,
)


@dataclass(frozen=True)
class ModelSpec:
    module: str
    filename: str
    family: str
    architecture_source: str
    license: str
    imagenet_initialization: str | None
    initialization_note: str
    notice_path: str
    license_text_path: str


MODEL_SPECS = {
    "lraspp_mobilenetv3_small": ModelSpec(
        module="openscaner.models.lraspp_mobilenetv3_small",
        filename="lraspp_mobilenetv3_small.pth",
        family="LR-ASPP with MobileNetV3-Small",
        architecture_source="https://github.com/pytorch/vision/tree/v0.28.0",
        license="BSD-3-Clause",
        imagenet_initialization=(
            "torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1 "
            "(mobilenet_v3_small-047dcff4.pth)"
        ),
        initialization_note="Official torchvision ImageNet-1K weights loaded before training.",
        notice_path="src/openscaner/third_party/LRASPP_MOBILENETV3_SMALL_NOTICE.txt",
        license_text_path="src/openscaner/third_party/licenses/BSD-3-Clause-Torchvision.txt",
    ),
    "pp_liteseg_t": ModelSpec(
        module="openscaner.models.pp_liteseg_t",
        filename="pp_liteseg_t.pth",
        family="PP-LiteSeg-T with STDC1",
        architecture_source=(
            "https://github.com/PaddlePaddle/PaddleSeg/tree/"
            "3c4db66de1d9d59d0628ed87590b6308a2f4aa2a"
        ),
        license="Apache-2.0",
        imagenet_initialization=(
            "PaddleSeg PP_STDCNet1.tar.gz ImageNet weights, SHA-256 "
            "245fe3c2e029c7ff271b4bd4e229f3849a50566fb6356ea0e56494146c6a9187"
        ),
        initialization_note=(
            "Official Paddle STDC1 ImageNet parameters are checksum-verified and "
            "deterministically mapped into the PyTorch port before training."
        ),
        notice_path="src/openscaner/third_party/PP_LITESEG_T_NOTICE.txt",
        license_text_path="src/openscaner/third_party/licenses/Apache-2.0.txt",
    ),
    "fast_scnn": ModelSpec(
        module="openscaner.models.fast_scnn",
        filename="fast_scnn.pth",
        family="Fast-SCNN",
        architecture_source=(
            "https://github.com/Tramac/Fast-SCNN-pytorch/tree/"
            "0638517d359ae1664a27dfb2cd1780a40a06c465"
        ),
        license="Apache-2.0",
        imagenet_initialization=None,
        initialization_note="The upstream architecture provides no ImageNet initialization.",
        notice_path="src/openscaner/third_party/FAST_SCNN_NOTICE.txt",
        license_text_path="src/openscaner/third_party/licenses/Apache-2.0.txt",
    ),
}

TRAINING_PROTOCOL = {
    "data_source": "deterministic_generated_documents",
    "dataset_seed": DATASET_SEED,
    "image_size": IMAGE_SIZE,
    "train_samples": TRAIN_SAMPLES,
    "validation_samples": VALIDATION_SAMPLES,
    "batch_size": 8,
    "epochs": 16,
    "early_stopping_patience": 5,
    "optimizer": "AdamW",
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "loss": "BCEWithLogits+soft Dice",
    "augmentation": "fixed per-sample deterministic synthetic stressors",
}


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return equally weighted binary cross-entropy and soft Dice losses."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return binary_cross_entropy + dice_loss


def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean hard-mask Dice with smoothing for empty masks."""
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    dimensions = tuple(range(1, predictions.ndim))
    intersection = (predictions & truth).sum(dim=dimensions, dtype=torch.float32)
    denominator = predictions.sum(dim=dimensions) + truth.sum(dim=dimensions)
    return ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one lightweight binary document segmenter"
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts/segmentation-training")
    )
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=VALIDATION_SAMPLES)
    parser.add_argument(
        "--batch-size", type=int, default=TRAINING_PROTOCOL["batch_size"]
    )
    parser.add_argument("--epochs", type=int, default=TRAINING_PROTOCOL["epochs"])
    parser.add_argument(
        "--patience", type=int, default=TRAINING_PROTOCOL["early_stopping_patience"]
    )
    parser.add_argument(
        "--learning-rate", type=float, default=TRAINING_PROTOCOL["learning_rate"]
    )
    parser.add_argument("--seed", type=int, default=DATASET_SEED)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(name: str, *, use_imagenet: bool) -> torch.nn.Module:
    spec = MODEL_SPECS[name]
    module = importlib.import_module(spec.module)
    return module.build_model(pretrained=use_imagenet)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            losses.append(float(bce_dice_loss(logits, targets).item()))
            scores.append(float(dice_score(logits, targets).item()))
    return statistics.fmean(losses), statistics.fmean(scores)


def _cpu_latency_ms(
    model: torch.nn.Module, image: torch.Tensor, cpu_threads: int
) -> float:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    torch.set_num_threads(cpu_threads)
    model = model.to("cpu").eval()
    sample = image.unsqueeze(0).to("cpu")
    with torch.inference_mode():
        for _ in range(2):
            model(sample)
        timings = []
        for _ in range(7):
            started = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - started) * 1000.0)
    return float(statistics.median(timings))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _update_manifest(path: Path, record: dict[str, object]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    models = [
        entry
        for entry in manifest["models"]
        if entry.get("adapter") != record["adapter"]
    ]
    models.append(record)
    manifest["models"] = models
    _write_json(path, manifest)


def train(options: argparse.Namespace) -> dict[str, object]:
    """Train one model, save its best validation checkpoint, and record evidence."""
    if (
        min(
            options.image_size,
            options.train_samples,
            options.validation_samples,
            options.batch_size,
            options.epochs,
            options.patience,
            options.cpu_threads,
        )
        < 1
    ):
        raise ValueError(
            "training counts, dimensions, and thread count must be positive"
        )
    _seed_everything(options.seed)
    device = _select_device(options.device)
    spec = MODEL_SPECS[options.model]
    use_imagenet = spec.imagenet_initialization is not None
    model = _build_model(options.model, use_imagenet=use_imagenet).to(device)
    train_dataset, validation_dataset = build_datasets(
        image_size=options.image_size,
        train_samples=options.train_samples,
        validation_samples=options.validation_samples,
        seed=options.seed,
    )
    loader_generator = torch.Generator().manual_seed(options.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=TRAINING_PROTOCOL["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=options.epochs
    )

    options.model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = options.model_dir / spec.filename
    best_dice = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    training_started = time.perf_counter()
    for epoch in range(1, options.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = bce_dice_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
        validation_loss, validation_dice = _evaluate(model, validation_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": statistics.fmean(train_losses),
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if validation_dice > best_dice + 1e-6:
            best_dice = validation_dice
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "schema_version": 1,
                "model_name": options.model,
                "image_size": options.image_size,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, checkpoint_path)
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= options.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cpu_model = _build_model(options.model, use_imagenet=False)
    cpu_model.load_state_dict(checkpoint["state_dict"], strict=True)
    latency_ms = _cpu_latency_ms(
        cpu_model, validation_dataset[0][0], options.cpu_threads
    )
    checkpoint_size = checkpoint_path.stat().st_size
    checkpoint_sha256 = _sha256(checkpoint_path)
    elapsed_seconds = time.perf_counter() - training_started
    protocol = {
        **TRAINING_PROTOCOL,
        "image_size": options.image_size,
        "train_samples": options.train_samples,
        "validation_samples": options.validation_samples,
        "batch_size": options.batch_size,
        "epochs_requested": options.epochs,
        "early_stopping_patience": options.patience,
        "learning_rate": options.learning_rate,
        "dataset_seed": options.seed,
    }
    report: dict[str, object] = {
        "model": options.model,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_size,
        "parameter_count": sum(
            parameter.numel() for parameter in cpu_model.parameters()
        ),
        "best_epoch": best_epoch,
        "validation_dice": best_dice,
        "cpu_latency_ms_median": latency_ms,
        "cpu_threads": options.cpu_threads,
        "training_device": str(device),
        "training_elapsed_seconds": elapsed_seconds,
        "history": history,
        "protocol": protocol,
        "data_limitation": SYNTHETIC_DATA_LIMITATION,
        "model_spec": asdict(spec),
    }
    _write_json(options.artifacts_dir / f"{options.model}.json", report)
    manifest_record: dict[str, object] = {
        "adapter": options.model,
        "model_family": spec.family,
        "local_filename": spec.filename,
        "upstream": spec.architecture_source,
        "source": (
            "Locally trained best-validation checkpoint from deterministic generated "
            f"documents; seed {options.seed}; command: python -m "
            f"openscaner.training.train --model {options.model} --device auto"
        ),
        "license": spec.license,
        "required_runtime": "PyTorch CPU",
        "sha256": checkpoint_sha256,
        "availability": "locally_trained",
        "runtime_detection": "exact filename and hardcoded SHA-256 verified before loading",
        "imagenet_initialization": spec.imagenet_initialization,
        "validation_dice": best_dice,
        "checkpoint_size_bytes": checkpoint_size,
        "parameter_count": report["parameter_count"],
        "cpu_latency_ms_median": latency_ms,
        "cpu_latency_threads": options.cpu_threads,
        "training_device": str(device),
        "best_epoch": best_epoch,
        "training_protocol": protocol,
        "data_limitation": SYNTHETIC_DATA_LIMITATION,
        "initialization_note": spec.initialization_note,
        "notice_path": spec.notice_path,
        "license_text_path": spec.license_text_path,
    }
    _update_manifest(options.manifest, manifest_record)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    report = train(options)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
