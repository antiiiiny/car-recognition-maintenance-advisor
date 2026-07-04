"""Transfer-learning CNN architectures for Stanford Cars classification."""

from __future__ import annotations

from typing import Literal


ArchitectureName = Literal["efficientnetb0", "resnet50"]


def normalize_architecture_name(name: str) -> ArchitectureName:
    """Normalize a user-provided architecture name.

    Args:
        name: Architecture name supplied by code or CLI.

    Returns:
        Normalized architecture name.

    Raises:
        ValueError: If the name is unsupported.
    """
    normalized = name.strip().lower().replace("-", "")
    if normalized in {"efficientnetb0", "efficientnet"}:
        return "efficientnetb0"
    if normalized in {"resnet50", "resnet"}:
        return "resnet50"
    raise ValueError(f"Unsupported architecture: {name}")


def build_transfer_model(
    architecture: str,
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    base_trainable: bool = False,
    dropout_rate: float = 0.2,
    weights: str | None = "imagenet",
):
    """Build an EfficientNetB0 or ResNet50 transfer-learning classifier.

    Args:
        architecture: ``efficientnetb0`` or ``resnet50``.
        num_classes: Number of output classes.
        input_shape: Input image shape.
        base_trainable: Whether to fine-tune the convolutional base immediately.
        dropout_rate: Dropout rate before the classifier layer.
        weights: Pretrained weights to load. Use ``"imagenet"`` for real
            training and ``None`` for fast local smoke runs.

    Returns:
        A compiled-ready Keras model. Compile it with training utilities.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to build CNN models.") from exc

    model_name = normalize_architecture_name(architecture)
    inputs = tf.keras.Input(shape=input_shape)
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="train_time_augmentation",
    )
    x = augmentation(inputs)

    if model_name == "efficientnetb0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        preprocess = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )

    base_model.trainable = base_trainable
    x = tf.keras.layers.Lambda(preprocess, name=f"{model_name}_preprocess")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"stanford_cars_{model_name}")