"""Tests for CNN evaluation helper functions."""

from __future__ import annotations

import numpy as np

from src.model.evaluation import calculate_classification_metrics, calculate_classification_metrics_from_probabilities


def test_calculate_classification_metrics_returns_macro_scores() -> None:
    """Verify evaluation metrics are computed as finite values."""
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 2, 2])

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics.accuracy == 0.75
    assert 0.0 <= metrics.macro_precision <= 1.0
    assert 0.0 <= metrics.macro_recall <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_calculate_classification_metrics_from_probabilities_includes_top_k() -> None:
    """Verify top-k accuracy is calculated from probability predictions."""
    y_true = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.7, 0.2],
        ]
    )

    metrics = calculate_classification_metrics_from_probabilities(y_true, probabilities, top_k=2)

    assert metrics.accuracy == 1 / 3
    assert metrics.top_5_accuracy == 1.0