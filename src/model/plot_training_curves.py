"""Plot Keras CSVLogger training curves for Stage 2 CNN runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def plot_training_curves(log_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Create loss and accuracy curve PNGs from a Keras training CSV log."""
    import matplotlib.pyplot as plt

    log = pd.read_csv(log_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write_plot(columns: list[str], title: str, filename: str) -> None:
        available = [column for column in columns if column in log.columns]
        if not available:
            return
        plt.figure(figsize=(8, 5))
        for column in available:
            plt.plot(log["epoch"], log[column], marker="o", label=column)
        plt.xlabel("Epoch")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        output_path = out_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        written.append(output_path)

    write_plot(["loss", "val_loss"], "Training and validation loss", "loss_curve.png")
    write_plot(
        ["top_1_accuracy", "val_top_1_accuracy", "top_5_accuracy", "val_top_5_accuracy"],
        "Training and validation accuracy",
        "accuracy_curve.png",
    )
    return written


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Stage 2 training curves from training_log.csv.")
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate curve PNGs from one training CSV log."""
    args = parse_args()
    paths = plot_training_curves(args.log_path, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()