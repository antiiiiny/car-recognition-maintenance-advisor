"""Smoke tests for dataset path helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.dataset_paths import get_stanford_cars_dir


def test_dataset_path_uses_default_location() -> None:
    """Verify the default Stanford Cars path is rooted under data/raw."""
    expected = Path("data") / "raw" / "stanford-cars"
    assert get_stanford_cars_dir() == expected
