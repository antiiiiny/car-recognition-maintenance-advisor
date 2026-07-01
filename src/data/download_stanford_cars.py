"""Download helper for the Stanford Cars Kaggle dataset."""

from __future__ import annotations

from pathlib import Path


def build_kaggle_command(output_dir: str | Path) -> list[str]:
    """Return the Kaggle CLI command used to download the dataset.

    Args:
        output_dir: Target directory for the downloaded zip/extracted files.

    Returns:
        A command list suitable for subprocess execution.
    """
    return [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "jutrera/stanford-car-dataset-by-classes-folder",
        "-p",
        str(Path(output_dir)),
        "--unzip",
    ]
