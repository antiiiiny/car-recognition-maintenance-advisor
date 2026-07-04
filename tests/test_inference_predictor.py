"""Tests for single-image prediction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.predictor import prediction_from_probabilities


def test_prediction_from_probabilities_returns_ranked_labels() -> None:
    """Verify class probabilities are converted to ranked predictions."""
    prediction = prediction_from_probabilities(
        probabilities=np.array([0.1, 0.7, 0.2]),
        class_names=["A", "B", "C"],
        top_k=2,
    )

    assert prediction.label == "B"
    assert prediction.confidence == pytest.approx(0.7)
    assert [item.label for item in prediction.top_predictions] == ["B", "C"]


def test_prediction_from_probabilities_validates_class_count() -> None:
    """Verify mismatched model outputs fail clearly."""
    with pytest.raises(ValueError, match="Expected 2 probabilities"):
        prediction_from_probabilities(
            probabilities=np.array([0.1, 0.2, 0.7]),
            class_names=["A", "B"],
        )