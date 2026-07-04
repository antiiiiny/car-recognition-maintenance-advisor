"""Tests for the Stage 4 maintenance pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.cnn_dataset import ClassMapping, save_class_mapping
from src.inference.pipeline import run_maintenance_pipeline
from src.llm.fake_client import FakeMaintenanceClient


class StubModel:
    """Small Keras-like model stub for pipeline tests."""

    def predict(self, batch, verbose: int = 0):
        return np.array([[0.05, 0.9, 0.05]])


def test_pipeline_generates_prediction_and_report(tmp_path: Path) -> None:
    """Verify one callable produces both prediction and report."""
    image_path = tmp_path / "car.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path)
    mapping_path = tmp_path / "class_mapping.json"
    save_class_mapping(ClassMapping(["Acura TL Sedan 2012", "BMW M3 Coupe 2012", "Audi R8 Coupe 2012"]), mapping_path)

    result = run_maintenance_pipeline(
        image_path=image_path,
        model_path=tmp_path / "unused.keras",
        class_mapping_path=mapping_path,
        image_size=32,
        mileage=85000,
        user_concern="engine noise",
        llm_client=FakeMaintenanceClient(),
        model=StubModel(),
    )

    assert result.prediction.label == "BMW M3 Coupe 2012"
    assert result.prediction.confidence == 0.9
    assert "BMW M3 Coupe 2012" in result.report
    assert "engine noise" in result.report