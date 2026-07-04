"""Command-line training entry point for Stage 2 CNN experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.cnn_dataset import create_image_datasets, save_class_mapping
from src.model.architectures import build_transfer_model, normalize_architecture_name
from src.model.training import TrainingConfig, compile_for_classification, create_callbacks, save_run_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CNN training."""
    parser = argparse.ArgumentParser(description="Train a Stanford Cars CNN classifier.")
    parser.add_argument("--architecture", choices=["efficientnetb0", "resnet50"], required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-trainable", action="store_true")
    parser.add_argument(
        "--smoke-batches",
        type=int,
        default=None,
        help="Limit train/validation datasets to this many batches for a quick smoke run.",
    )
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet weights for real training or none for offline smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Train one requested CNN architecture and save standard artifacts."""
    args = parse_args()
    architecture = normalize_architecture_name(args.architecture)
    config = TrainingConfig(
        architecture=architecture,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    save_run_config(config)

    datasets = create_image_datasets(
        dataset_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    train_dataset = datasets.train
    validation_dataset = datasets.validation
    if args.smoke_batches is not None:
        train_dataset = train_dataset.take(args.smoke_batches)
        validation_dataset = validation_dataset.take(args.smoke_batches)

    run_dir = args.output_dir / architecture
    save_class_mapping(datasets.class_mapping, run_dir / "class_mapping.json")

    model = build_transfer_model(
        architecture=architecture,
        num_classes=datasets.class_mapping.num_classes,
        input_shape=(args.image_size, args.image_size, 3),
        base_trainable=args.base_trainable,
        weights=None if args.weights == "none" else args.weights,
    )
    compile_for_classification(model, learning_rate=args.learning_rate)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=create_callbacks(config),
    )


if __name__ == "__main__":
    main()