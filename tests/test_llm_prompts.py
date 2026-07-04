"""Tests for LLM prompt construction."""

from __future__ import annotations

import pytest

from src.llm.prompts import REPORT_SECTIONS, build_maintenance_prompt


def test_prompt_includes_label_mileage_concern_and_sections() -> None:
    """Verify prompts preserve user inputs and required report sections."""
    prompt = build_maintenance_prompt(
        predicted_label="Acura TL Sedan 2012",
        mileage=85000,
        user_concern="engine noise",
    )

    assert "Predicted vehicle label: Acura TL Sedan 2012" in prompt
    assert "85,000 miles" in prompt
    assert "engine noise" in prompt
    assert "Use the predicted vehicle label exactly as provided" in prompt
    assert "Do not change, contradict, or invent" in prompt
    for section in REPORT_SECTIONS:
        assert section in prompt


def test_prompt_handles_missing_optional_inputs() -> None:
    """Verify optional fields are represented clearly when absent."""
    prompt = build_maintenance_prompt("BMW M3 Coupe 2012")

    assert "Predicted vehicle label: BMW M3 Coupe 2012" in prompt
    assert "Mileage: not provided" in prompt
    assert "Customer concern: not provided" in prompt


def test_prompt_rejects_empty_label() -> None:
    """Verify empty labels fail before any client call."""
    with pytest.raises(ValueError, match="predicted_label"):
        build_maintenance_prompt("   ")