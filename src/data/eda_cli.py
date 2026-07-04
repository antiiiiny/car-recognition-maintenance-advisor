"""Command-line entry point for Stanford Cars exploratory checks."""

from __future__ import annotations

import argparse

from src.data.dataset_paths import get_stanford_cars_dir
from src.data.eda import load_names, summarize_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Summarize the Stanford Cars dataset.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Path to the extracted Stanford Cars dataset. "
            "Defaults to the discovered project dataset layout."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset summary and print a compact report."""
    args = parse_args()
    dataset_dir = get_stanford_cars_dir(args.data_dir)
    summary = summarize_dataset(dataset_dir)
    print(f"Dataset root: {summary.root_dir}")
    print(f"Split counts: {summary.split_counts}")
    print(f"Classes found: {len(summary.class_counts)}")
    print(f"Image size buckets: {len(summary.image_sizes)}")
    print(f"Corrupt files: {len(summary.corrupt_files)}")
    print(f"Names loaded: {len(load_names(summary.root_dir))}")


if __name__ == "__main__":
    main()
