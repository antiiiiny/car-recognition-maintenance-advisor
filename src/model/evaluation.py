"""Evaluation helpers for CNN classification metrics and reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    """Compact classification metrics for model comparison and write-up."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Calculate macro-averaged classification metrics.

    Args:
        y_true: Ground-truth integer class indices.
        y_pred: Predicted integer class indices.

    Returns:
        Accuracy and macro precision/recall/F1 metrics.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
    )


def save_metrics(metrics: ClassificationMetrics, output_path: str | Path) -> None:
    """Save classification metrics as JSON.

    Args:
        metrics: Metrics to save.
        output_path: JSON output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2, sort_keys=True), encoding="utf-8")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    """Save a confusion matrix as CSV.

    Args:
        y_true: Ground-truth integer class indices.
        y_pred: Predicted integer class indices.
        output_path: CSV output path.
    """
    from sklearn.metrics import confusion_matrix

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    np.savetxt(path, matrix, delimiter=",", fmt="%d")