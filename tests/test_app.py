"""Tests for the Stage 5 Streamlit app helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from src.app.pipeline_runner import (
    AppConfigError,
    build_llm_client,
    resolve_artifact_path,
    run_pipeline,
)
from src.data.cnn_dataset import ClassMapping, save_class_mapping
from src.llm.client import OpenAIMaintenanceClient
from src.llm.fake_client import FakeMaintenanceClient


class StubModel:
    """Small Keras-like model stub for app tests."""

    def predict(self, batch, verbose: int = 0):
        return np.array([[0.05, 0.9, 0.05]])


# --- resolve_artifact_path -------------------------------------------------


def test_resolve_artifact_path_returns_existing(tmp_path: Path) -> None:
    """Existing path resolves successfully."""
    file = tmp_path / "model.keras"
    file.write_text("dummy")
    result = resolve_artifact_path(str(file), "Model file")
    assert result == file


def test_resolve_artifact_path_missing_raises(tmp_path: Path) -> None:
    """Missing path raises a friendly AppConfigError."""
    with pytest.raises(AppConfigError, match="Model file not found"):
        resolve_artifact_path(str(tmp_path / "missing.keras"), "Model file")


def test_resolve_artifact_path_empty_raises() -> None:
    """Empty path raises a friendly AppConfigError."""
    with pytest.raises(AppConfigError, match="Model file path is empty"):
        resolve_artifact_path("", "Model file")


# --- build_llm_client ------------------------------------------------------


def test_build_llm_client_fake_returns_fake() -> None:
    """Fake toggle on returns FakeMaintenanceClient."""
    client = build_llm_client(use_fake=True)
    assert isinstance(client, FakeMaintenanceClient)


def test_build_llm_client_real_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real client without OPENAI_API_KEY raises a friendly error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    # Block dotenv from loading a real .env during this test.
    monkeypatch.setattr("src.llm.client.load_dotenv", lambda: False)
    with pytest.raises(AppConfigError, match="OPENAI_API_KEY is not configured"):
        build_llm_client(use_fake=False)


def test_build_llm_client_real_with_key_returns_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real client with explicit key returns OpenAIMaintenanceClient."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = build_llm_client(use_fake=False, api_key="sk-test-key")
    assert isinstance(client, OpenAIMaintenanceClient)
    # Never expose the key in the config object's repr via the test output.
    assert client.config.api_key == "sk-test-key"


# --- run_pipeline ----------------------------------------------------------


def test_run_pipeline_end_to_end_with_fake_client(tmp_path: Path) -> None:
    """Pipeline helper runs end-to-end with a fake client and stub model."""
    image_path = tmp_path / "car.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path)
    mapping_path = tmp_path / "class_mapping.json"
    save_class_mapping(
        ClassMapping(["Acura TL Sedan 2012", "BMW M3 Coupe 2012", "Audi R8 Coupe 2012"]),
        mapping_path,
    )

    result = run_pipeline(
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
