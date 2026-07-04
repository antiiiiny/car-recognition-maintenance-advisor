"""CNN model package for Stanford Cars classification."""

from src.model.architectures import build_transfer_model, normalize_architecture_name
from src.model.evaluation import ClassificationMetrics, calculate_classification_metrics
from src.model.training import ModelRunResult, TrainingConfig, select_winner

__all__ = [
	"ClassificationMetrics",
	"ModelRunResult",
	"TrainingConfig",
	"build_transfer_model",
	"calculate_classification_metrics",
	"normalize_architecture_name",
	"select_winner",
]
"""Model training and evaluation utilities."""
