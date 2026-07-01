"""Command-line entry point for Stanford Cars exploratory checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.eda import load_names, summarize_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Summarize the Stanford Cars dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "raw" / "stanford-cars",
        help="Path to the extracted Stanford Cars dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset summary and print a compact report."""
    args = parse_args()
    summary = summarize_dataset(args.data_dir)
    print(f"Dataset root: {summary.root_dir}")
    print(f"Split counts: {summary.split_counts}")
    print(f"Classes found: {len(summary.class_counts)}")
    print(f"Image size buckets: {len(summary.image_sizes)}")
    print(f"Corrupt files: {len(summary.corrupt_files)}")
    print(f"Names loaded: {len(load_names(summary.root_dir))}")


if __name__ == "__main__":
    main()
