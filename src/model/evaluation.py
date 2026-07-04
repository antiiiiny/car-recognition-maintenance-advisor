"""Evaluation helpers for CNN classification metrics and reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    """Compact classification metrics for model comparison and write-up."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    top_5_accuracy: float | None = None


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
        top_5_accuracy=None,
    )


def calculate_top_k_accuracy(y_true: np.ndarray, y_probability: np.ndarray, k: int = 5) -> float:
    """Calculate top-k categorical accuracy from class probabilities.

    Args:
        y_true: Ground-truth integer class indices.
        y_probability: Model probability/logit array shaped ``(n_examples, n_classes)``.
        k: Number of highest-scoring classes to consider.

    Returns:
        Top-k accuracy as a float between 0 and 1.
    """
    if y_probability.ndim != 2:
        raise ValueError("y_probability must be a 2D array of shape (examples, classes).")
    if y_probability.shape[0] != y_true.shape[0]:
        raise ValueError("y_true and y_probability must contain the same number of examples.")
    if y_true.size == 0:
        raise ValueError("At least one example is required to calculate top-k accuracy.")

    effective_k = min(k, y_probability.shape[1])
    top_k_indices = np.argpartition(y_probability, -effective_k, axis=1)[:, -effective_k:]
    matches = np.any(top_k_indices == y_true[:, None], axis=1)
    return float(np.mean(matches))


def calculate_classification_metrics_from_probabilities(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    top_k: int = 5,
) -> ClassificationMetrics:
    """Calculate top-1, top-k, and macro metrics from model probabilities.

    Args:
        y_true: Ground-truth integer class indices.
        y_probability: Model probabilities/logits.
        top_k: Top-k value, defaulting to 5 for the Stanford Cars 196-class task.

    Returns:
        Classification metrics including top-k accuracy.
    """
    y_pred = np.argmax(y_probability, axis=1)
    metrics = calculate_classification_metrics(y_true, y_pred)
    return ClassificationMetrics(
        accuracy=metrics.accuracy,
        macro_precision=metrics.macro_precision,
        macro_recall=metrics.macro_recall,
        macro_f1=metrics.macro_f1,
        top_5_accuracy=calculate_top_k_accuracy(y_true, y_probability, k=top_k),
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


def extract_labels_from_categorical_dataset(dataset: Any) -> np.ndarray:
    """Extract integer labels from a TensorFlow dataset with categorical labels."""
    labels: list[np.ndarray] = []
    for _, batch_labels in dataset:
        labels.append(np.argmax(batch_labels.numpy(), axis=1))
    if not labels:
        return np.array([], dtype=np.int64)
    return np.concatenate(labels).astype(np.int64)


def evaluate_probability_predictions(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    output_dir: str | Path,
    prefix: str = "metrics",
) -> ClassificationMetrics:
    """Calculate and save metrics, predictions, and confusion matrix.

    Args:
        y_true: Ground-truth integer labels.
        y_probability: Predicted probabilities.
        output_dir: Directory where artifacts should be written.
        prefix: File prefix, such as ``validation`` or ``test``.

    Returns:
        Calculated classification metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(y_probability, axis=1).astype(np.int64)
    metrics = calculate_classification_metrics_from_probabilities(y_true, y_probability, top_k=5)

    save_metrics(metrics, output_path / f"{prefix}_metrics.json")
    save_confusion_matrix(y_true, y_pred, output_path / f"{prefix}_confusion_matrix.csv")
    save_predictions(y_true, y_pred, y_probability, output_path / f"{prefix}_predictions.csv")
    return metrics


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save predicted labels and confidence scores as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    confidences = np.max(y_probability, axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_index", "y_true", "y_pred", "confidence"])
        writer.writeheader()
        for index, (actual, predicted, confidence) in enumerate(zip(y_true, y_pred, confidences, strict=True)):
            writer.writerow(
                {
                    "example_index": index,
                    "y_true": int(actual),
                    "y_pred": int(predicted),
                    "confidence": float(confidence),
                }
            )