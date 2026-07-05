"""Confusion matrix analysis for the Stage 6 evaluation write-up.

This module reads the saved test predictions/confusion matrix artifacts from
Stage 2 evaluation and produces:

- A ranked list of the most-confused class pairs (true -> predicted)
- A filtered confusion-matrix heatmap PNG for the top-N most-confused pairs
- A JSON summary artifact with the most-confused pairs and their counts

It is intentionally lightweight (NumPy + matplotlib only) so it can run on
Windows CPU without TensorFlow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConfusedPair:
    """A single true -> predicted confusion pair with count and indices."""

    true_index: int
    predicted_index: int
    true_label: str
    predicted_label: str
    count: int


@dataclass(frozen=True)
class ConfusionAnalysis:
    """Summary of the most-confused class pairs."""

    total_examples: int
    total_misclassified: int
    overall_accuracy: float
    top_confused_pairs: list[ConfusedPair]


def load_class_labels(class_mapping_path: str | Path) -> list[str]:
    """Load class labels from a Stage 2 class mapping JSON.

    Args:
        class_mapping_path: Path to ``class_mapping.json``.

    Returns:
        Ordered list of class label strings.
    """
    path = Path(class_mapping_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["class_names"])


def load_predictions_csv(predictions_path: str | Path) -> dict[str, np.ndarray]:
    """Load y_true, y_pred, and confidence from a Stage 2 predictions CSV.

    Args:
        predictions_path: Path to ``<prefix>_predictions.csv``.

    Returns:
        Dictionary with ``y_true``, ``y_pred``, and ``confidence`` arrays.
    """
    path = Path(predictions_path)
    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        expected = "example_index,y_true,y_pred,confidence"
        if header.replace(" ", "") != expected.replace(" ", ""):
            raise ValueError(f"Unexpected predictions CSV header: {header}")
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            y_true.append(int(parts[1]))
            y_pred.append(int(parts[2]))
            confidence.append(float(parts[3]))
    return {
        "y_true": np.array(y_true, dtype=np.int64),
        "y_pred": np.array(y_pred, dtype=np.int64),
        "confidence": np.array(confidence, dtype=np.float64),
    }


def analyze_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    top_k: int = 15,
) -> ConfusionAnalysis:
    """Compute the most-confused class pairs.

    Args:
        y_true: Ground-truth integer class indices.
        y_pred: Predicted integer class indices.
        class_labels: Ordered class label strings.
        top_k: Number of most-confused pairs to return.

    Returns:
        Confusion analysis with the top-k confused pairs (excluding the
        diagonal, i.e. correct predictions are not counted as confusions).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        raise ValueError("At least one prediction is required.")

    total = int(y_true.size)
    correct = int(np.sum(y_true == y_pred))
    misclassified = total - correct

    pairs: list[ConfusedPair] = []
    for true_index in range(len(class_labels)):
        for predicted_index in range(len(class_labels)):
            if true_index == predicted_index:
                continue
            count = int(np.sum((y_true == true_index) & (y_pred == predicted_index)))
            if count == 0:
                continue
            pairs.append(
                ConfusedPair(
                    true_index=true_index,
                    predicted_index=predicted_index,
                    true_label=class_labels[true_index],
                    predicted_label=class_labels[predicted_index],
                    count=count,
                )
            )

    pairs.sort(key=lambda p: p.count, reverse=True)
    return ConfusionAnalysis(
        total_examples=total,
        total_misclassified=misclassified,
        overall_accuracy=correct / total if total else 0.0,
        top_confused_pairs=pairs[:top_k],
    )


def save_confusion_analysis(analysis: ConfusionAnalysis, output_path: str | Path) -> None:
    """Save the confusion analysis as JSON.

    Args:
        analysis: Confusion analysis to save.
        output_path: JSON output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "total_examples": analysis.total_examples,
        "total_misclassified": analysis.total_misclassified,
        "overall_accuracy": analysis.overall_accuracy,
        "top_confused_pairs": [asdict(p) for p in analysis.top_confused_pairs],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    output_path: str | Path,
    top_k_classes: int = 20,
    title: str = "Confusion Matrix (Top-K Most-Confused Classes)",
) -> None:
    """Plot a filtered confusion matrix heatmap for the top-K most-confused classes.

    Args:
        y_true: Ground-truth integer class indices.
        y_pred: Predicted integer class indices.
        class_labels: Ordered class label strings.
        output_path: PNG output path.
        top_k_classes: Number of classes to include in the heatmap.
        title: Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Identify the classes with the most off-diagonal confusion.
    full_matrix = confusion_matrix(y_true, y_pred)
    off_diagonal = full_matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    confusion_per_class = off_diagonal.sum(axis=0) + off_diagonal.sum(axis=1)
    top_indices = np.argsort(confusion_per_class)[::-1][:top_k_classes]
    top_indices = np.sort(top_indices)

    sub_matrix = full_matrix[np.ix_(top_indices, top_indices)]
    short_labels = [class_labels[i][:25] for i in top_indices]

    fig, ax = plt.subplots(figsize=(max(10, top_k_classes * 0.5), max(8, top_k_classes * 0.5)))
    im = ax.imshow(sub_matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(top_indices)))
    ax.set_yticks(range(len(top_indices)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves_from_log(
    log_path: str | Path,
    output_path: str | Path,
    title: str = "EfficientNetB0 Training Curves",
) -> None:
    """Plot training and validation accuracy/loss curves from a training log CSV.

    Args:
        log_path: Path to ``training_log.csv``.
        output_path: PNG output path.
        title: Plot title.
    """
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(log_path)
    epochs: list[int] = []
    train_acc: list[float] = []
    val_acc: list[float] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_acc.append(float(row["top_1_accuracy"]))
            val_acc.append(float(row["val_top_1_accuracy"]))
            train_loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_acc, marker="o", label="Train Top-1")
    axes[0].plot(epochs, val_acc, marker="s", label="Val Top-1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[1].plot(epochs, val_loss, marker="s", label="Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
