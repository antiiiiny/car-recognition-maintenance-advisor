"""Inference utilities for predicted vehicle labels."""

from src.inference.pipeline import MaintenancePipelineResult, run_maintenance_pipeline
from src.inference.predictor import ClassProbability, VehiclePrediction, predict_vehicle

__all__ = [
	"ClassProbability",
	"MaintenancePipelineResult",
	"VehiclePrediction",
	"predict_vehicle",
	"run_maintenance_pipeline",
]
