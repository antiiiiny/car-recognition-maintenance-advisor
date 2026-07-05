"""CLI entry point for Stage 6 confusion matrix analysis.

Generates:
- ``confusion_analysis.json`` — top-K most-confused class pairs
- ``confusion_heatmap.png`` — filtered confusion matrix heatmap
- ``training_curves.png`` — accuracy/loss curves from the training log

Examples::

    python -m src.model.analyze_confusion \
        --predictions artifacts/stage2_full/efficientnetb0/eval_test/test_predictions.csv \
        --class-mapping artifacts/stage2_full/efficientnetb0/class_mapping.json \
        --training-log artifacts/stage2_full/efficientnetb0/training_log.csv \
        --output-dir artifacts/stage6_evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model.confusion_analysis import (
    analyze_confusion,
    load_class_labels,
    load_predictions_csv,
    plot_confusion_heatmap,
    plot_training_curves_from_log,
    save_confusion_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Analyze CNN confusion matrix and generate Stage 6 artifacts.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to <prefix>_predictions.csv from Stage 2 evaluation.",
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        required=True,
        help="Path to class_mapping.json.",
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=None,
        help="Optional path to training_log.csv for curve plotting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output artifacts.",
    )
    parser.add_argument("--top-k-pairs", type=int, default=15, help="Number of most-confused pairs to report.")
    parser.add_argument("--top-k-classes", type=int, default=20, help="Number of classes in the heatmap.")
    return parser.parse_args()


def main() -> None:
    """Run the confusion analysis and save artifacts."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = load_class_labels(args.class_mapping)
    data = load_predictions_csv(args.predictions)
    analysis = analyze_confusion(
        y_true=data["y_true"],
        y_pred=data["y_pred"],
        class_labels=class_labels,
        top_k=args.top_k_pairs,
    )
    save_confusion_analysis(analysis, output_dir / "confusion_analysis.json")

    plot_confusion_heatmap(
        y_true=data["y_true"],
        y_pred=data["y_pred"],
        class_labels=class_labels,
        output_path=output_dir / "confusion_heatmap.png",
        top_k_classes=args.top_k_classes,
    )

    if args.training_log is not None and args.training_log.exists():
        plot_training_curves_from_log(
            log_path=args.training_log,
            output_path=output_dir / "training_curves.png",
        )

    print(f"Total examples: {analysis.total_examples}")
    print(f"Misclassified: {analysis.total_misclassified}")
    print(f"Overall accuracy: {analysis.overall_accuracy:.4f}")
    print(f"\nTop {len(analysis.top_confused_pairs)} most-confused pairs:")
    for pair in analysis.top_confused_pairs[:10]:
        print(f"  {pair.count}x  {pair.true_label}  ->  {pair.predicted_label}")
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
