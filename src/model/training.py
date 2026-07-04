"""Training utilities for the Stanford Cars CNN pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a CNN training run."""

    architecture: str
    epochs: int = 10
    learning_rate: float = 1e-3
    output_dir: Path = Path("artifacts") / "models"
    checkpoint_monitor: str = "val_top_1_accuracy"


@dataclass(frozen=True)
class ModelRunResult:
    """Summary metrics and artifact paths for one model run."""

    architecture: str
    validation_top_1_accuracy: float
    validation_top_5_accuracy: float
    history_path: Path | None = None
    checkpoint_path: Path | None = None


def compile_for_classification(model: Any, learning_rate: float = 1e-3) -> Any:
    """Compile a Keras model for multiclass car classification.

    Args:
        model: Keras model to compile.
        learning_rate: Adam optimizer learning rate.

    Returns:
        The compiled model.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to compile CNN models.") from exc

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="top_1_accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )
    return model


def create_callbacks(config: TrainingConfig) -> list[Any]:
    """Create standard callbacks for checkpointing and CSV metric logging.

    Args:
        config: Training run configuration.

    Returns:
        Keras callback list.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to create training callbacks.") from exc

    run_dir = config.output_dir / config.architecture
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.keras"
    csv_log_path = run_dir / "training_log.csv"

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=config.checkpoint_monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(csv_log_path),
        tf.keras.callbacks.EarlyStopping(
            monitor=config.checkpoint_monitor,
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
    ]


def save_history(history: Any, output_path: str | Path) -> None:
    """Save a Keras history object as CSV.

    Args:
        history: Keras ``History`` object.
        output_path: CSV output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = history.history
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    metric_names = list(rows.keys())
    row_count = max(len(values) for values in rows.values())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *metric_names])
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(
                {
                    "epoch": index + 1,
                    **{
                        metric_name: rows[metric_name][index]
                        for metric_name in metric_names
                        if index < len(rows[metric_name])
                    },
                }
            )


def save_run_config(config: TrainingConfig) -> None:
    """Save the training configuration for reproducibility.

    Args:
        config: Training run configuration.
    """
    run_dir = config.output_dir / config.architecture
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def select_winner(results: list[ModelRunResult]) -> ModelRunResult:
    """Select the best model using validation top-1, then validation top-5.

    Args:
        results: Candidate model run summaries.

    Returns:
        Winning model run.

    Raises:
        ValueError: If no results are provided.
    """
    if not results:
        raise ValueError("At least one model result is required to select a winner.")

    return max(
        results,
        key=lambda result: (result.validation_top_1_accuracy, result.validation_top_5_accuracy),
    )