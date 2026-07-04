"""Dataset exploration helpers for the Stanford Cars dataset."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from src.data.dataset_paths import get_stanford_cars_dir


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    """Simple image metadata used for exploratory checks."""

    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class DatasetSummary:
    """High-level dataset statistics."""

    root_dir: Path
    split_counts: dict[str, int]
    class_counts: dict[str, int]
    image_sizes: dict[tuple[int, int], int]
    corrupt_files: list[Path]
    annotations: dict[str, list[dict[str, str]]]


def get_metadata_dir(dataset_dir: Path) -> Path:
    """Return the directory that stores Stanford Cars metadata files."""
    if (dataset_dir / "names.csv").exists():
        return dataset_dir

    if len(dataset_dir.parents) >= 2:
        candidate = dataset_dir.parents[1]
        if (candidate / "names.csv").exists():
            return candidate

    return dataset_dir


def load_names(dataset_dir: Path) -> list[str]:
    """Load Stanford Cars class names if available."""
    names_file = get_metadata_dir(dataset_dir) / "names.csv"
    if not names_file.exists():
        return []

    with names_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_annotations(dataset_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Load Stanford Cars annotations for train and test splits."""
    annotations: dict[str, list[dict[str, str]]] = {}
    metadata_dir = get_metadata_dir(dataset_dir)
    for split_name in ("train", "test"):
        annotation_file = metadata_dir / f"anno_{split_name}.csv"
        if not annotation_file.exists():
            annotations[split_name] = []
            continue

        with annotation_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = [
                {
                    "file": row[0],
                    "x1": row[1],
                    "y1": row[2],
                    "x2": row[3],
                    "y2": row[4],
                    "class_id": row[5],
                }
                for row in reader
                if len(row) >= 6
            ]
        annotations[split_name] = rows

    return annotations


def iter_image_files(dataset_dir: Path) -> Iterable[Path]:
    """Yield image files under the dataset directory."""
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def get_split_counts(dataset_dir: Path) -> dict[str, int]:
    """Count images per top-level split folder."""
    counts: Counter[str] = Counter()
    for image_path in iter_image_files(dataset_dir):
        try:
            split_name = image_path.relative_to(dataset_dir).parts[0]
        except IndexError:
            continue
        counts[split_name] += 1
    return dict(sorted(counts.items()))


def get_class_counts(dataset_dir: Path) -> dict[str, int]:
    """Count images per class folder.

    Args:
        dataset_dir: Root Stanford Cars directory.

    Returns:
        Mapping of class folder name to image count.
    """
    counts: Counter[str] = Counter()
    for image_path in iter_image_files(dataset_dir):
        class_name = image_path.parent.name
        counts[class_name] += 1
    return dict(sorted(counts.items()))


def get_image_size_counts(dataset_dir: Path) -> dict[tuple[int, int], int]:
    """Count image width/height combinations."""
    sizes: Counter[tuple[int, int]] = Counter()
    for image_path in iter_image_files(dataset_dir):
        try:
            with Image.open(image_path) as image:
                sizes[(image.width, image.height)] += 1
        except (UnidentifiedImageError, OSError):
            continue
    return dict(sorted(sizes.items()))


def find_corrupt_files(dataset_dir: Path) -> list[Path]:
    """Return paths that cannot be opened as images."""
    corrupt_files: list[Path] = []
    for image_path in iter_image_files(dataset_dir):
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            corrupt_files.append(image_path)
    return corrupt_files


def summarize_dataset(dataset_dir: Path | None = None) -> DatasetSummary:
    """Build a compact summary for dataset EDA."""
    root_dir = dataset_dir if dataset_dir is not None else get_stanford_cars_dir()
    return DatasetSummary(
        root_dir=root_dir,
        split_counts=get_split_counts(root_dir),
        class_counts=get_class_counts(root_dir),
        image_sizes=get_image_size_counts(root_dir),
        corrupt_files=find_corrupt_files(root_dir),
        annotations=load_annotations(root_dir),
    )
