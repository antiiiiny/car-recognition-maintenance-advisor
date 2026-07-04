"""Tests for model architecture selection and training utilities."""

from __future__ import annotations

import pytest

from src.model.compare_runs import compare_run_directories
from src.model.architectures import normalize_architecture_name
from src.model.training import ModelRunResult, select_winner


def test_normalize_architecture_name_accepts_supported_aliases() -> None:
    """Verify supported architecture names normalize to canonical values."""
    assert normalize_architecture_name("EfficientNet-B0") == "efficientnetb0"
    assert normalize_architecture_name("resnet") == "resnet50"


def test_normalize_architecture_name_rejects_unknown_model() -> None:
    """Verify unsupported architecture names fail clearly."""
    with pytest.raises(ValueError):
        normalize_architecture_name("mobilenet")


def test_select_winner_uses_validation_top_1_then_top_5() -> None:
    """Verify validation top-1 wins, with validation top-5 as tie-breaker."""
    efficientnet = ModelRunResult(
        architecture="efficientnetb0",
        validation_top_1_accuracy=0.70,
        validation_top_5_accuracy=0.91,
    )
    resnet = ModelRunResult(
        architecture="resnet50",
        validation_top_1_accuracy=0.70,
        validation_top_5_accuracy=0.93,
    )

    assert select_winner([efficientnet, resnet]) == resnet


def test_select_winner_requires_results() -> None:
    """Verify winner selection requires at least one model result."""
    with pytest.raises(ValueError):
        select_winner([])


def test_compare_run_directories_writes_summary_and_winner(tmp_path) -> None:
    """Verify run comparison writes reproducible winner artifacts."""
    efficientnet_dir = tmp_path / "efficientnetb0"
    resnet_dir = tmp_path / "resnet50"
    output_dir = tmp_path / "comparison"
    efficientnet_dir.mkdir()
    resnet_dir.mkdir()
    (efficientnet_dir / "training_log.csv").write_text(
        "epoch,val_top_1_accuracy,val_top_5_accuracy\n1,0.2,0.4\n2,0.3,0.5\n",
        encoding="utf-8",
    )
    (resnet_dir / "training_log.csv").write_text(
        "epoch,val_top_1_accuracy,val_top_5_accuracy\n1,0.3,0.6\n",
        encoding="utf-8",
    )

    winner = compare_run_directories(
        {"efficientnetb0": efficientnet_dir, "resnet50": resnet_dir},
        output_dir,
    )

    assert winner.architecture == "resnet50"
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "selected_model.json").exists()