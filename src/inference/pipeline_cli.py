"""CLI for the Stage 4 image-to-report pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.pipeline import run_maintenance_pipeline
from src.inference.predictor import DEFAULT_CLASS_MAPPING_PATH, DEFAULT_INFERENCE_IMAGE_SIZE, DEFAULT_MODEL_PATH
from src.llm.fake_client import FakeMaintenanceClient


def parse_args() -> argparse.Namespace:
    """Parse pipeline CLI arguments."""
    parser = argparse.ArgumentParser(description="Run vehicle image prediction and maintenance report generation.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--class-mapping-path", type=Path, default=DEFAULT_CLASS_MAPPING_PATH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_INFERENCE_IMAGE_SIZE)
    parser.add_argument("--mileage", type=int, default=None)
    parser.add_argument("--concern", default=None)
    parser.add_argument("--fake", action="store_true", help="Use fake LLM output instead of OpenAI.")
    return parser.parse_args()


def main() -> None:
    """Run the Stage 4 pipeline and print the result."""
    args = parse_args()
    client = FakeMaintenanceClient() if args.fake else None
    result = run_maintenance_pipeline(
        image_path=args.image_path,
        model_path=args.model_path,
        class_mapping_path=args.class_mapping_path,
        image_size=args.image_size,
        mileage=args.mileage,
        user_concern=args.concern,
        llm_client=client,
    )
    print(f"Prediction: {result.prediction.label}")
    print(f"Confidence: {result.prediction.confidence:.4f}")
    print("Top predictions:")
    for item in result.prediction.top_predictions:
        print(f"- {item.label}: {item.probability:.4f}")
    print("\nMaintenance Report")
    print("==================")
    print(result.report)


if __name__ == "__main__":
    main()