"""Single-image vehicle prediction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.cnn_dataset import load_class_mapping
from src.model.architectures import preprocess_resnet50_input


DEFAULT_MODEL_PATH = Path("artifacts") / "stage2_full" / "efficientnetb0" / "best_model.keras"
DEFAULT_CLASS_MAPPING_PATH = Path("artifacts") / "stage2_full" / "efficientnetb0" / "class_mapping.json"
DEFAULT_INFERENCE_IMAGE_SIZE = 160


@dataclass(frozen=True)
class ClassProbability:
    """Predicted class label and probability."""

    label: str
    probability: float


@dataclass(frozen=True)
class VehiclePrediction:
    """Top vehicle prediction and ranked alternatives."""

    label: str
    confidence: float
    top_predictions: list[ClassProbability]


def load_prediction_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load a trusted local Keras model for prediction.

    Args:
        model_path: Path to a saved Keras model.

    Returns:
        Loaded Keras model.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for vehicle prediction.") from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return tf.keras.models.load_model(
        path,
        custom_objects={"preprocess_resnet50_input": preprocess_resnet50_input},
        safe_mode=False,
    )


def load_image_batch(image_path: str | Path, image_size: int = DEFAULT_INFERENCE_IMAGE_SIZE) -> np.ndarray:
    """Load one image as a batch suitable for the trained CNN.

    Args:
        image_path: Path to an input image.
        image_size: Target square image size.

    Returns:
        Batch array with shape ``(1, image_size, image_size, 3)``.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for image preprocessing.") from exc

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = tf.keras.utils.load_img(path, target_size=(image_size, image_size))
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def prediction_from_probabilities(
    probabilities: np.ndarray,
    class_names: list[str],
    top_k: int = 5,
) -> VehiclePrediction:
    """Convert model probabilities to a vehicle prediction object.

    Args:
        probabilities: Model probability output for one image.
        class_names: Class labels ordered by model index.
        top_k: Number of ranked predictions to include.

    Returns:
        Vehicle prediction with top-k alternatives.
    """
    scores = np.asarray(probabilities).reshape(-1)
    if scores.size != len(class_names):
        raise ValueError(f"Expected {len(class_names)} probabilities, received {scores.size}.")

    count = min(top_k, len(class_names))
    top_indices = np.argsort(scores)[::-1][:count]
    top_predictions = [
        ClassProbability(label=class_names[index], probability=float(scores[index]))
        for index in top_indices
    ]
    winner = top_predictions[0]
    return VehiclePrediction(label=winner.label, confidence=winner.probability, top_predictions=top_predictions)


def predict_vehicle(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING_PATH,
    image_size: int = DEFAULT_INFERENCE_IMAGE_SIZE,
    top_k: int = 5,
    model=None,
) -> VehiclePrediction:
    """Predict the vehicle class for a single image.

    Args:
        image_path: Path to the input image.
        model_path: Path to the Keras model if ``model`` is not provided.
        class_mapping_path: Path to the class mapping JSON.
        image_size: Target square image size.
        top_k: Number of ranked predictions to return.
        model: Optional preloaded Keras-compatible model.

    Returns:
        Vehicle prediction result.
    """
    mapping = load_class_mapping(class_mapping_path)
    predictor = model or load_prediction_model(model_path)
    batch = load_image_batch(image_path, image_size=image_size)
    probabilities = predictor.predict(batch, verbose=0)[0]
    return prediction_from_probabilities(probabilities, mapping.class_names, top_k=top_k)