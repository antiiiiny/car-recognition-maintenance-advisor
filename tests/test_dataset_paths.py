"""Smoke tests for dataset path helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.dataset_paths import EXTRACTED_DATASET_SUBDIR, get_stanford_cars_dir


def test_dataset_path_falls_back_to_raw_location(tmp_path: Path) -> None:
    """Verify the fallback Stanford Cars path is rooted under data/raw."""
    expected = tmp_path / "raw" / "stanford-cars"

    assert get_stanford_cars_dir(tmp_path) == expected


def test_dataset_path_prefers_extracted_layout(tmp_path: Path) -> None:
    """Verify the extracted Kaggle layout is preferred when present."""
    expected = tmp_path / EXTRACTED_DATASET_SUBDIR
    expected.mkdir(parents=True)

    assert get_stanford_cars_dir(tmp_path) == expected
