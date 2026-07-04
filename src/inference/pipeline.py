"""End-to-end image-to-maintenance-report pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.inference.predictor import (
    DEFAULT_CLASS_MAPPING_PATH,
    DEFAULT_INFERENCE_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
    VehiclePrediction,
    predict_vehicle,
)
from src.llm.report_generator import MaintenanceReportClient, generate_maintenance_report


@dataclass(frozen=True)
class MaintenancePipelineResult:
    """Combined CNN prediction and maintenance report."""

    prediction: VehiclePrediction
    report: str


def run_maintenance_pipeline(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING_PATH,
    image_size: int = DEFAULT_INFERENCE_IMAGE_SIZE,
    mileage: int | None = None,
    user_concern: str | None = None,
    llm_client: MaintenanceReportClient | None = None,
    model=None,
) -> MaintenancePipelineResult:
    """Run image classification and maintenance report generation.

    Args:
        image_path: Path to the input image.
        model_path: Path to the trained Keras model.
        class_mapping_path: Path to class mapping JSON.
        image_size: Target square image size.
        mileage: Optional vehicle mileage.
        user_concern: Optional customer concern.
        llm_client: Optional fake or real LLM client.
        model: Optional preloaded Keras-compatible model.

    Returns:
        Pipeline result with prediction and report.
    """
    prediction = predict_vehicle(
        image_path=image_path,
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        image_size=image_size,
        model=model,
    )
    report = generate_maintenance_report(
        predicted_label=prediction.label,
        mileage=mileage,
        user_concern=user_concern,
        llm_client=llm_client,
    )
    return MaintenancePipelineResult(prediction=prediction, report=report)