"""Evaluate a saved CNN checkpoint on validation or held-out test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.cnn_dataset import create_image_datasets
from src.model.architectures import preprocess_resnet50_input
from src.model.evaluation import evaluate_probability_predictions, extract_labels_from_categorical_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for saved-model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a saved Stanford Cars CNN checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a saved Keras model/checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset root containing train/test folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation artifacts.")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional batch limit for quick validation. Do not use for final test metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the model, run predictions, and save evaluation artifacts."""
    args = parse_args()
    import tensorflow as tf

    datasets = create_image_datasets(
        dataset_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    dataset = datasets.validation if args.split == "validation" else datasets.test
    if args.max_batches is not None:
        dataset = dataset.take(args.max_batches)

    # The training model contains preprocessing Lambda layers, so safe_mode=False is
    # required when loading Keras v3 `.keras` checkpoints from trusted local training.
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"preprocess_resnet50_input": preprocess_resnet50_input},
        safe_mode=False,
    )
    probabilities = model.predict(dataset, verbose=1)
    y_true = extract_labels_from_categorical_dataset(dataset)
    metrics = evaluate_probability_predictions(
        y_true=y_true,
        y_probability=probabilities,
        output_dir=args.output_dir,
        prefix=args.split,
    )
    print(metrics)


if __name__ == "__main__":
    main()