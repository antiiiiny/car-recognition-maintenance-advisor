"""Compare Stage 2 CNN training runs and select the validation winner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.model.training import ModelRunResult, select_winner


def summarize_training_log(architecture: str, run_dir: str | Path) -> ModelRunResult:
    """Summarize best validation metrics from one run directory."""
    run_path = Path(run_dir)
    log_path = run_path / "training_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    log = pd.read_csv(log_path)
    required_columns = {"val_top_1_accuracy", "val_top_5_accuracy"}
    missing = required_columns - set(log.columns)
    if missing:
        raise ValueError(f"Missing columns in {log_path}: {sorted(missing)}")

    best_index = log["val_top_1_accuracy"].idxmax()
    best_row = log.loc[best_index]
    return ModelRunResult(
        architecture=architecture,
        validation_top_1_accuracy=float(best_row["val_top_1_accuracy"]),
        validation_top_5_accuracy=float(best_row["val_top_5_accuracy"]),
        history_path=log_path,
        checkpoint_path=run_path / "best_model.keras",
    )


def compare_run_directories(run_dirs: dict[str, Path], output_dir: str | Path) -> ModelRunResult:
    """Compare runs, save summary artifacts, and return the selected winner."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [summarize_training_log(architecture, run_dir) for architecture, run_dir in run_dirs.items()]
    winner = select_winner(results)

    summary_path = out_dir / "model_comparison.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "architecture",
                "validation_top_1_accuracy",
                "validation_top_5_accuracy",
                "history_path",
                "checkpoint_path",
                "selected_winner",
            ],
        )
        writer.writeheader()
        for result in results:
            payload = asdict(result)
            payload["history_path"] = str(result.history_path) if result.history_path else ""
            payload["checkpoint_path"] = str(result.checkpoint_path) if result.checkpoint_path else ""
            payload["selected_winner"] = result.architecture == winner.architecture
            writer.writerow(payload)

    winner_payload = asdict(winner)
    winner_payload["history_path"] = str(winner.history_path) if winner.history_path else None
    winner_payload["checkpoint_path"] = str(winner.checkpoint_path) if winner.checkpoint_path else None
    (out_dir / "selected_model.json").write_text(json.dumps(winner_payload, indent=2, sort_keys=True), encoding="utf-8")
    return winner


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare EfficientNetB0 and ResNet50 Stage 2 training runs.")
    parser.add_argument("--efficientnet-dir", type=Path, required=True)
    parser.add_argument("--resnet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Compare requested run directories and print the winner."""
    args = parse_args()
    winner = compare_run_directories(
        {
            "efficientnetb0": args.efficientnet_dir,
            "resnet50": args.resnet_dir,
        },
        args.output_dir,
    )
    print(f"Selected winner: {winner.architecture}")


if __name__ == "__main__":
    main()