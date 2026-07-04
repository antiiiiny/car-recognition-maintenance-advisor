"""Deterministic fake LLM client for tests and offline demos."""

from __future__ import annotations

import re

from src.llm.prompts import REPORT_SECTIONS


class FakeMaintenanceClient:
    """Return stable maintenance reports without network or API usage."""

    def generate(self, prompt: str) -> str:
        """Generate a deterministic fake report from the prompt.

        Args:
            prompt: Prompt text containing a predicted vehicle label.

        Returns:
            Structured fake maintenance report.
        """
        label = self._extract_field(prompt, "Predicted vehicle label") or "Unknown vehicle"
        mileage = self._extract_field(prompt, "Mileage") or "not provided"
        concern = self._extract_field(prompt, "Customer concern") or "not provided"

        return "\n\n".join(
            [
                f"{REPORT_SECTIONS[0]}\n{label}",
                f"{REPORT_SECTIONS[1]}\nInspect common wear items for {label}, including fluids, brakes, tires, belts, hoses, and age-related electrical concerns.",
                f"{REPORT_SECTIONS[2]}\n- Check engine oil and filter.\n- Inspect brake pads and rotors.\n- Inspect tire condition and pressure.\n- Check coolant, transmission fluid, and battery health.",
                f"{REPORT_SECTIONS[3]}\nMileage provided: {mileage}. Use the mileage to prioritize fluid services, brake inspection, tire condition, and suspension checks.",
                f"{REPORT_SECTIONS[4]}\nFor this {label}, start with basic safety and maintenance checks, then investigate the customer concern: {concern}.",
                f"{REPORT_SECTIONS[5]}\nThis is general guidance, not a diagnosis. A qualified technician should inspect the vehicle before repairs are approved.",
            ]
        )

    @staticmethod
    def _extract_field(prompt: str, field_name: str) -> str | None:
        """Extract a simple ``Field: value`` line from a prompt."""
        match = re.search(rf"^{re.escape(field_name)}:\s*(.+)$", prompt, flags=re.MULTILINE)
        return match.group(1).strip() if match else None