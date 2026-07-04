"""TensorFlow dataset helpers for the Stanford Cars CNN pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.dataset_paths import get_stanford_cars_dir


DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ClassMapping:
    """Serializable mapping between integer class indices and class labels."""

    class_names: list[str]

    @property
    def class_to_index(self) -> dict[str, int]:
        """Return a label-to-index lookup."""
        return {class_name: index for index, class_name in enumerate(self.class_names)}

    @property
    def index_to_class(self) -> dict[int, str]:
        """Return an index-to-label lookup."""
        return {index: class_name for index, class_name in enumerate(self.class_names)}

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.class_names)


@dataclass(frozen=True)
class DatasetBundle:
    """Container for train/validation/test TensorFlow datasets and labels."""

    train: Any
    validation: Any
    test: Any
    class_mapping: ClassMapping


def get_split_dir(dataset_dir: str | Path, split_name: str) -> Path:
    """Return a dataset split directory.

    Args:
        dataset_dir: Root dataset directory containing split folders.
        split_name: Split folder name, such as ``train`` or ``test``.

    Returns:
        Path to the requested split directory.
    """
    return Path(dataset_dir) / split_name


def resolve_dataset_root(dataset_dir: str | Path | None = None) -> Path:
    """Resolve either a direct dataset root or a base data directory.

    Args:
        dataset_dir: Optional path. If it already contains ``train`` and ``test``
            folders, it is treated as the dataset root. Otherwise, it is passed
            through the project dataset discovery helper.

    Returns:
        Dataset root containing ``train`` and ``test`` split folders.
    """
    if dataset_dir is not None:
        candidate = Path(dataset_dir)
        if get_split_dir(candidate, "train").exists() and get_split_dir(candidate, "test").exists():
            return candidate

    return get_stanford_cars_dir(dataset_dir)


def discover_class_names(dataset_dir: str | Path | None = None) -> list[str]:
    """Discover Stanford Cars class names from the train split folders.

    Args:
        dataset_dir: Optional dataset root. Defaults to the project dataset layout.

    Returns:
        Alphabetically sorted class folder names.

    Raises:
        FileNotFoundError: If the train split directory is missing.
        ValueError: If no class directories are found.
    """
    root_dir = resolve_dataset_root(dataset_dir)
    train_dir = get_split_dir(root_dir, "train")
    if not train_dir.exists():
        raise FileNotFoundError(f"Train split directory not found: {train_dir}")

    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No class folders found in train split: {train_dir}")

    return class_names


def validate_expected_class_count(class_names: list[str], expected_count: int = 196) -> None:
    """Validate that the discovered class list has the expected length.

    Args:
        class_names: Class names discovered from the dataset.
        expected_count: Expected number of classes.

    Raises:
        ValueError: If the class count differs from ``expected_count``.
    """
    actual_count = len(class_names)
    if actual_count != expected_count:
        raise ValueError(f"Expected {expected_count} classes, found {actual_count}.")


def save_class_mapping(class_mapping: ClassMapping, output_path: str | Path) -> None:
    """Save a class mapping as deterministic JSON.

    Args:
        class_mapping: Mapping to save.
        output_path: JSON output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": class_mapping.class_names,
        "class_to_index": class_mapping.class_to_index,
        "index_to_class": {str(index): label for index, label in class_mapping.index_to_class.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_class_mapping(input_path: str | Path) -> ClassMapping:
    """Load a class mapping written by :func:`save_class_mapping`.

    Args:
        input_path: JSON mapping file path.

    Returns:
        Loaded class mapping.
    """
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return ClassMapping(class_names=list(payload["class_names"]))


def create_image_datasets(
    dataset_dir: str | Path | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = DEFAULT_SEED,
) -> DatasetBundle:
    """Create train, validation, and test datasets for CNN training.

    The train split is divided into train/validation subsets. The held-out test
    split is loaded separately and must only be used after model selection.

    Args:
        dataset_dir: Optional dataset root. Defaults to the project dataset layout.
        image_size: Target image size as ``(height, width)``.
        batch_size: Number of images per batch.
        validation_split: Fraction of the training split used for validation.
        seed: Random seed used for deterministic train/validation splitting.

    Returns:
        Dataset bundle with train, validation, test, and class mapping.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to create image datasets.") from exc

    root_dir = resolve_dataset_root(dataset_dir)
    train_dir = get_split_dir(root_dir, "train")
    test_dir = get_split_dir(root_dir, "test")
    class_names = discover_class_names(root_dir)
    validate_expected_class_count(class_names)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    return DatasetBundle(
        train=train_dataset.prefetch(autotune),
        validation=validation_dataset.prefetch(autotune),
        test=test_dataset.prefetch(autotune),
        class_mapping=ClassMapping(class_names=class_names),
    )