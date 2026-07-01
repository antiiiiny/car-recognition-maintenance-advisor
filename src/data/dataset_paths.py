"""Utilities for locating the Stanford Cars dataset on disk."""

from __future__ import annotations

from pathlib import Path


DEFAULT_DATA_DIR = Path("data")
RAW_DATA_SUBDIR = "raw"
STANFORD_CARS_SUBDIR = "stanford-cars"
EXTRACTED_DATASET_SUBDIR = Path("car_data") / "car_data"


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the base data directory.

    Args:
        data_dir: Optional override for the base data directory.

    Returns:
        The resolved data directory path.
    """
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def get_stanford_cars_dir(data_dir: str | Path | None = None) -> Path:
    """Return the Stanford Cars dataset directory.

    Args:
        data_dir: Optional override for the base data directory.

    Returns:
        The Stanford Cars dataset path, preferring the extracted layout when present.
    """
    base_dir = get_data_dir(data_dir)
    extracted_dir = base_dir / EXTRACTED_DATASET_SUBDIR
    if extracted_dir.exists():
        return extracted_dir

    return base_dir / RAW_DATA_SUBDIR / STANFORD_CARS_SUBDIR
