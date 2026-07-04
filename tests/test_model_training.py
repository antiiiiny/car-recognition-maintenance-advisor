"""Tests for model architecture selection and training utilities."""

from __future__ import annotations

import pytest

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