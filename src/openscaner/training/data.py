"""Deterministic, target-independent synthetic document training data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 384
DATASET_SEED = 20260825
TRAIN_SAMPLES = 256
VALIDATION_SAMPLES = 64
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
FORBIDDEN_FIXTURE_SHA256 = frozenset(
    {
        "ea7cc4e7255051730710ebaef345973eb39fb5e22df2cd363595dda8b66ae83b",
        "f077cecae296dd2b095735414af99f2b676f3b723c99bd63d6d56a35e5e1b02d",
    }
)
DATASET_SOURCE_HASHES: frozenset[str] = frozenset()
SYNTHETIC_DATA_LIMITATION = (
    "No public document-boundary dataset was available locally; all samples are "
    "deterministically generated and may not represent real-camera domain variation."
)
AUGMENTATION_STRESSORS = frozenset(
    {
        "clutter",
        "projective_distortion",
        "small_document",
        "occlusion",
        "shadow",
        "blur",
        "glare",
        "folds",
        "low_contrast",
    }
)


def assert_fixture_hashes_absent(source_hashes: Iterable[str]) -> None:
    """Reject any dataset source whose digest matches a protected benchmark fixture."""
    normalized = {str(digest).lower() for digest in source_hashes}
    overlap = FORBIDDEN_FIXTURE_SHA256.intersection(normalized)
    if overlap:
        raise ValueError(
            f"protected fixture hash present in training data: {sorted(overlap)}"
        )


assert_fixture_hashes_absent(DATASET_SOURCE_HASHES)


@dataclass(frozen=True)
class DatasetSplit:
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]


def deterministic_split(
    *,
    train_samples: int = TRAIN_SAMPLES,
    validation_samples: int = VALIDATION_SAMPLES,
    seed: int = DATASET_SEED,
) -> DatasetSplit:
    """Return one stable, disjoint split shared by every architecture."""
    if train_samples < 1 or validation_samples < 1:
        raise ValueError("train_samples and validation_samples must be positive")
    total = train_samples + validation_samples
    generator = np.random.default_rng(seed)
    identifiers = generator.permutation(total).tolist()
    split = DatasetSplit(
        train_ids=tuple(int(value) for value in identifiers[:train_samples]),
        validation_ids=tuple(int(value) for value in identifiers[train_samples:]),
    )
    assert set(split.train_ids).isdisjoint(split.validation_ids)
    return split


def _paper_quad(generator: np.random.Generator, size: int) -> np.ndarray:
    target_area = float(np.exp(generator.uniform(np.log(0.05), np.log(0.90))))
    aspect = float(np.exp(generator.uniform(np.log(0.58), np.log(1.45))))
    width = min(0.94, np.sqrt(target_area * aspect))
    height = min(0.94, target_area / width)
    if height > 0.94:
        height = 0.94
        width = min(0.94, target_area / height)

    center_x = generator.uniform(width / 2 + 0.025, 1.0 - width / 2 - 0.025)
    center_y = generator.uniform(height / 2 + 0.025, 1.0 - height / 2 - 0.025)
    half_width = width * size / 2
    half_height = height * size / 2
    center = np.array((center_x * size, center_y * size), dtype=np.float32)
    corners = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float32,
    )
    angle = generator.uniform(-np.pi, np.pi)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    corners = corners @ rotation.T + center
    perspective = generator.uniform(-0.28, 0.28, size=(4, 2)).astype(np.float32)
    corners += perspective * np.array((width * size, height * size), dtype=np.float32)
    corners[:, 0] = np.clip(corners[:, 0], 3, size - 4)
    corners[:, 1] = np.clip(corners[:, 1], 3, size - 4)
    hull = cv2.convexHull(corners).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < size * size * 0.025:
        return _paper_quad(generator, size)
    center_of_hull = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - center_of_hull[1], hull[:, 0] - center_of_hull[0])
    return hull[np.argsort(angles)].astype(np.float32)


def _background(generator: np.random.Generator, size: int) -> np.ndarray:
    base = generator.integers(15, 205, size=3).astype(np.float32)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    gradient = (
        generator.uniform(-55, 55) * xx / max(size - 1, 1)
        + generator.uniform(-55, 55) * yy / max(size - 1, 1)
    )[..., None]
    image = np.broadcast_to(base, (size, size, 3)).copy() + gradient
    image += generator.normal(0, generator.uniform(2, 16), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    for _ in range(int(generator.integers(7, 22))):
        color = tuple(int(value) for value in generator.integers(5, 245, size=3))
        if generator.random() < 0.55:
            first = tuple(int(value) for value in generator.integers(0, size, size=2))
            second = tuple(int(value) for value in generator.integers(0, size, size=2))
            thickness = int(generator.integers(2, max(3, size // 24)))
            cv2.line(image, first, second, color, thickness, cv2.LINE_AA)
        else:
            first = tuple(int(value) for value in generator.integers(0, size, size=2))
            extent = generator.integers(max(3, size // 30), max(4, size // 4), size=2)
            second = tuple(
                int(value) for value in np.minimum(np.array(first) + extent, size - 1)
            )
            cv2.rectangle(image, first, second, color, -1, cv2.LINE_AA)

    if generator.random() < 0.55:
        key_size = max(4, size // int(generator.integers(16, 26)))
        origin = generator.integers(-key_size, size // 2, size=2)
        color = tuple(int(value) for value in generator.integers(8, 75, size=3))
        for row in range(7):
            for column in range(11):
                x = int(origin[0] + column * key_size * 1.15)
                y = int(origin[1] + row * key_size * 1.15)
                cv2.rectangle(image, (x, y), (x + key_size, y + key_size), color, 1)
    return image


def _document_layer(
    generator: np.random.Generator,
    size: int,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((size, size), dtype=np.uint8)
    polygon = np.rint(corners).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, 255, cv2.LINE_AA)

    tone = int(generator.integers(168, 252))
    warm = int(generator.integers(-16, 18))
    paper_color = np.array(
        [np.clip(tone + warm, 0, 255), tone, np.clip(tone - warm, 0, 255)],
        dtype=np.uint8,
    )
    paper = np.broadcast_to(paper_color, (size, size, 3)).copy()
    paper_noise = generator.normal(0, generator.uniform(1.0, 7.0), paper.shape)
    paper = np.clip(paper.astype(np.float32) + paper_noise, 0, 255).astype(np.uint8)

    ink = tuple(int(value) for value in generator.integers(15, 115, size=3))
    x_min, y_min = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    x_max, y_max = np.minimum(np.ceil(corners.max(axis=0)).astype(int), size - 1)
    for _ in range(int(generator.integers(7, 30))):
        y = int(generator.integers(y_min, max(y_min + 1, y_max + 1)))
        x = int(generator.integers(x_min, max(x_min + 1, x_max + 1)))
        length = int(generator.integers(max(4, size // 25), max(5, size // 3)))
        cv2.line(
            paper,
            (x, y),
            (
                min(size - 1, x + length),
                int(np.clip(y + generator.integers(-8, 9), 0, size - 1)),
            ),
            ink,
            int(generator.integers(1, max(2, size // 96 + 1))),
            cv2.LINE_AA,
        )

    for _ in range(int(generator.integers(0, 4))):
        start = corners[int(generator.integers(0, 4))]
        end = corners[(int(generator.integers(0, 4)) + 2) % 4]
        cv2.line(
            paper,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            (145, 145, 145),
            int(generator.integers(1, 3)),
            cv2.LINE_AA,
        )
    return paper, mask


def _reduce_document_edge_contrast(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend every paper-side boundary pixel toward its nearest background.

    This deliberately creates a difficult paper/background transition without
    changing the complete document ground-truth mask.
    """
    binary_mask = (mask >= 128).astype(np.uint8)
    inside_distance, nearest_outside_labels = cv2.distanceTransformWithLabels(
        binary_mask,
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    edge_width = max(3.0, min(image.shape[:2]) / 32.0)
    paper_edge = (inside_distance >= 1.0) & (inside_distance <= edge_width)
    background = binary_mask == 0
    background_colors = np.zeros(
        (int(nearest_outside_labels.max()) + 1, 3),
        dtype=np.float32,
    )
    background_colors[nearest_outside_labels[background]] = image[background]
    paired_background = background_colors[nearest_outside_labels[paper_edge]]
    reduced = image.astype(np.float32)
    reduced[paper_edge] = reduced[paper_edge] * 0.1 + paired_background * 0.9
    return np.rint(reduced).astype(np.uint8)


def _compose_sample(
    sample_id: int,
    *,
    image_size: int,
    augment: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(np.random.SeedSequence((seed, int(sample_id))))
    image = _background(generator, image_size)
    corners = _paper_quad(generator, image_size)
    paper, mask = _document_layer(generator, image_size, corners)

    shadow = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=max(1.0, image_size * generator.uniform(0.008, 0.04)),
    )
    shadow = np.roll(
        shadow,
        shift=(int(generator.integers(2, max(3, image_size // 15))),) * 2,
        axis=(0, 1),
    )
    image = np.clip(
        image.astype(np.float32)
        * (
            1.0
            - shadow[..., None].astype(np.float32)
            / 255.0
            * generator.uniform(0.08, 0.38)
        ),
        0,
        255,
    ).astype(np.uint8)
    alpha = mask[..., None].astype(np.float32) / 255.0
    image = np.rint(image * (1.0 - alpha) + paper * alpha).astype(np.uint8)

    if augment:
        if generator.random() < 0.8:
            glare = np.zeros((image_size, image_size), dtype=np.uint8)
            center = tuple(int(value) for value in corners.mean(axis=0))
            axes = (
                int(
                    generator.integers(
                        max(2, image_size // 30), max(3, image_size // 5)
                    )
                ),
                int(
                    generator.integers(
                        max(2, image_size // 50), max(3, image_size // 12)
                    )
                ),
            )
            cv2.ellipse(
                glare,
                center,
                axes,
                float(generator.uniform(0, 180)),
                0,
                360,
                255,
                -1,
                cv2.LINE_AA,
            )
            glare = cv2.GaussianBlur(glare, (0, 0), sigmaX=max(1.0, image_size / 50))
            glare_alpha = (glare & mask).astype(np.float32)[..., None] / 255.0
            image = np.clip(
                image.astype(np.float32) + glare_alpha * generator.uniform(20, 95),
                0,
                255,
            ).astype(np.uint8)

        for _ in range(int(generator.integers(0, 4))):
            first = tuple(
                int(value) for value in generator.integers(0, image_size, size=2)
            )
            second = tuple(
                int(value) for value in generator.integers(0, image_size, size=2)
            )
            color = tuple(int(value) for value in generator.integers(5, 220, size=3))
            cv2.line(
                image,
                first,
                second,
                color,
                int(
                    generator.integers(
                        max(2, image_size // 60), max(3, image_size // 10)
                    )
                ),
                cv2.LINE_AA,
            )

        if generator.random() < 0.65:
            sigma = float(generator.uniform(0.3, 2.2))
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        if generator.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        image = _reduce_document_edge_contrast(image, mask)

    return image, (mask >= 128).astype(np.float32)


class SyntheticDocumentDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A deterministic generated dataset containing no external image assets."""

    source_hashes = DATASET_SOURCE_HASHES

    def __init__(
        self,
        *,
        sample_ids: Sequence[int] | range,
        image_size: int = IMAGE_SIZE,
        augment: bool,
        seed: int = DATASET_SEED,
    ) -> None:
        if image_size < 64:
            raise ValueError("image_size must be at least 64")
        identifiers = tuple(int(value) for value in sample_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("sample_ids must be non-empty and unique")
        self.sample_ids = identifiers
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        assert_fixture_hashes_absent(self.source_hashes)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[index]
        image, mask = _compose_sample(
            sample_id,
            image_size=self.image_size,
            augment=self.augment,
            seed=self.seed,
        )
        encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1]
        digest = hashlib.sha256(encoded.tobytes()).hexdigest()
        assert_fixture_hashes_absent((digest,))

        normalized = image.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1)).copy())
        mask_tensor = torch.from_numpy(mask[None].copy())
        return image_tensor, mask_tensor


def build_datasets(
    *,
    image_size: int = IMAGE_SIZE,
    train_samples: int = TRAIN_SAMPLES,
    validation_samples: int = VALIDATION_SAMPLES,
    seed: int = DATASET_SEED,
) -> tuple[SyntheticDocumentDataset, SyntheticDocumentDataset]:
    """Build the one shared deterministic split and augmentation protocol."""
    split = deterministic_split(
        train_samples=train_samples,
        validation_samples=validation_samples,
        seed=seed,
    )
    return (
        SyntheticDocumentDataset(
            sample_ids=split.train_ids,
            image_size=image_size,
            augment=True,
            seed=seed,
        ),
        SyntheticDocumentDataset(
            sample_ids=split.validation_ids,
            image_size=image_size,
            augment=False,
            seed=seed,
        ),
    )
