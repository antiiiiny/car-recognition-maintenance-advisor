"""Maintenance report generation facade."""

from __future__ import annotations

from typing import Protocol

from src.llm.client import OpenAIMaintenanceClient
from src.llm.prompts import build_maintenance_prompt


class MaintenanceReportClient(Protocol):
    """Protocol for real or fake maintenance-report clients."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


def generate_maintenance_report(
    predicted_label: str,
    mileage: int | None = None,
    user_concern: str | None = None,
    llm_client: MaintenanceReportClient | None = None,
) -> str:
    """Generate a structured vehicle maintenance report.

    Args:
        predicted_label: Predicted Stanford Cars class label.
        mileage: Optional mileage value.
        user_concern: Optional customer concern.
        llm_client: Optional fake or real LLM client. If omitted, the OpenAI
            client is used.

    Returns:
        Generated report text.
    """
    prompt = build_maintenance_prompt(
        predicted_label=predicted_label,
        mileage=mileage,
        user_concern=user_concern,
    )
    client = llm_client or OpenAIMaintenanceClient()
    return client.generate(prompt)