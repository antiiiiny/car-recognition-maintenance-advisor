"""Tests for the Stage 6 confusion matrix analysis module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.model.confusion_analysis import (
    ConfusedPair,
    analyze_confusion,
    load_class_labels,
    load_predictions_csv,
    plot_confusion_heatmap,
    plot_training_curves_from_log,
    save_confusion_analysis,
)


# --- load_class_labels -----------------------------------------------------


def test_load_class_labels(tmp_path: Path) -> None:
    """Class labels load in order from the class mapping JSON."""
    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(json.dumps({"class_names": ["Acura TL Sedan 2012", "BMW M3 Coupe 2012"]}), encoding="utf-8")
    labels = load_class_labels(mapping_path)
    assert labels == ["Acura TL Sedan 2012", "BMW M3 Coupe 2012"]


# --- load_predictions_csv --------------------------------------------------


def test_load_predictions_csv(tmp_path: Path) -> None:
    """Predictions CSV loads into y_true, y_pred, confidence arrays."""
    path = tmp_path / "test_predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_index", "y_true", "y_pred", "confidence"])
        writer.writeheader()
        writer.writerow({"example_index": 0, "y_true": 0, "y_pred": 1, "confidence": 0.7})
        writer.writerow({"example_index": 1, "y_true": 1, "y_pred": 1, "confidence": 0.9})

    data = load_predictions_csv(path)
    assert data["y_true"].tolist() == [0, 1]
    assert data["y_pred"].tolist() == [1, 1]
    assert data["confidence"].tolist() == [0.7, 0.9]


def test_load_predictions_csv_bad_header(tmp_path: Path) -> None:
    """Bad header raises ValueError."""
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c,d\n1,2,3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected predictions CSV header"):
        load_predictions_csv(path)


# --- analyze_confusion -----------------------------------------------------


def test_analyze_confusion_basic() -> None:
    """Most-confused pairs are ranked by count, excluding the diagonal."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 1])  # 0->1 once, 1->0 once, 2->1 once
    labels = ["A", "B", "C"]
    analysis = analyze_confusion(y_true, y_pred, labels, top_k=10)
    assert analysis.total_examples == 6
    assert analysis.total_misclassified == 3
    assert analysis.overall_accuracy == pytest.approx(0.5)
    # All three off-diagonal pairs have count 1
    assert len(analysis.top_confused_pairs) == 3
    assert all(pair.count == 1 for pair in analysis.top_confused_pairs)


def test_analyze_confusion_ranking() -> None:
    """Pairs are sorted by count descending."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0])  # 0->1 thrice, 1->0 twice
    labels = ["A", "B"]
    analysis = analyze_confusion(y_true, y_pred, labels, top_k=5)
    assert analysis.top_confused_pairs[0].count == 3
    assert analysis.top_confused_pairs[0].true_label == "A"
    assert analysis.top_confused_pairs[0].predicted_label == "B"
    assert analysis.top_confused_pairs[1].count == 2


def test_analyze_confusion_empty_raises() -> None:
    """Empty arrays raise ValueError."""
    with pytest.raises(ValueError, match="At least one prediction"):
        analyze_confusion(np.array([]), np.array([]), ["A"])


def test_analyze_confusion_shape_mismatch_raises() -> None:
    """Mismatched y_true/y_pred shapes raise ValueError."""
    with pytest.raises(ValueError, match="same shape"):
        analyze_confusion(np.array([0, 1]), np.array([0]), ["A", "B"])


# --- save_confusion_analysis -----------------------------------------------


def test_save_confusion_analysis(tmp_path: Path) -> None:
    """Analysis saves as JSON with the expected structure."""
    pair = ConfusedPair(true_index=0, predicted_index=1, true_label="A", predicted_label="B", count=5)
    from src.model.confusion_analysis import ConfusionAnalysis

    analysis = ConfusionAnalysis(
        total_examples=10,
        total_misclassified=5,
        overall_accuracy=0.5,
        top_confused_pairs=[pair],
    )
    out = tmp_path / "analysis.json"
    save_confusion_analysis(analysis, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total_examples"] == 10
    assert data["total_misclassified"] == 5
    assert data["top_confused_pairs"][0]["count"] == 5


# --- plot functions (smoke tests) ------------------------------------------


def test_plot_confusion_heatmap_creates_png(tmp_path: Path) -> None:
    """Heatmap PNG is created."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 1])
    labels = ["A", "B", "C"]
    out = tmp_path / "heatmap.png"
    plot_confusion_heatmap(y_true, y_pred, labels, out, top_k_classes=3)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_training_curves_from_log(tmp_path: Path) -> None:
    """Training curves PNG is created from a log CSV."""
    log_path = tmp_path / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "loss", "top_1_accuracy", "top_5_accuracy", "val_loss", "val_top_1_accuracy", "val_top_5_accuracy"],
        )
        writer.writeheader()
        writer.writerow({"epoch": 0, "loss": 2.0, "top_1_accuracy": 0.3, "top_5_accuracy": 0.6, "val_loss": 2.5, "val_top_1_accuracy": 0.25, "val_top_5_accuracy": 0.55})
        writer.writerow({"epoch": 1, "loss": 1.5, "top_1_accuracy": 0.5, "top_5_accuracy": 0.75, "val_loss": 1.8, "val_top_1_accuracy": 0.45, "val_top_5_accuracy": 0.7})

    out = tmp_path / "curves.png"
    plot_training_curves_from_log(log_path, out)
    assert out.exists()
    assert out.stat().st_size > 0
