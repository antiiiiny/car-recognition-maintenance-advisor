"""Tests for CNN evaluation helper functions."""

from __future__ import annotations

import numpy as np

from src.model.evaluation import calculate_classification_metrics


def test_calculate_classification_metrics_returns_macro_scores() -> None:
    """Verify evaluation metrics are computed as finite values."""
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 2, 2])

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics.accuracy == 0.75
    assert 0.0 <= metrics.macro_precision <= 1.0
    assert 0.0 <= metrics.macro_recall <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0