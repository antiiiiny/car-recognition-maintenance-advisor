"""Tests for CNN dataset helper utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.cnn_dataset import ClassMapping, discover_class_names, load_class_mapping, save_class_mapping


def test_class_mapping_round_trip(tmp_path: Path) -> None:
    """Verify class mappings save and load deterministically."""
    mapping = ClassMapping(class_names=["Acura TL Sedan 2012", "Audi R8 Coupe 2012"])
    output_path = tmp_path / "class_mapping.json"

    save_class_mapping(mapping, output_path)
    loaded = load_class_mapping(output_path)

    assert loaded == mapping
    assert loaded.class_to_index["Audi R8 Coupe 2012"] == 1
    assert loaded.index_to_class[0] == "Acura TL Sedan 2012"


def test_discover_class_names_uses_sorted_train_folders(tmp_path: Path) -> None:
    """Verify class discovery is alphabetical and based on train folders."""
    train_dir = tmp_path / "train"
    (tmp_path / "test").mkdir()
    (train_dir / "BMW M3 Coupe 2012").mkdir(parents=True)
    (train_dir / "Acura TL Sedan 2012").mkdir(parents=True)

    assert discover_class_names(tmp_path) == ["Acura TL Sedan 2012", "BMW M3 Coupe 2012"]


def test_discover_class_names_fails_when_train_missing(tmp_path: Path) -> None:
    """Verify a clear error is raised for missing train split."""
    with pytest.raises(FileNotFoundError):
        discover_class_names(tmp_path)