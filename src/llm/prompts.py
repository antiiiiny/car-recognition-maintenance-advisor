"""Prompt templates for maintenance advisory report generation."""

from __future__ import annotations


REPORT_SECTIONS = [
    "Vehicle",
    "Common Issues",
    "Preventive Maintenance Checklist",
    "Mileage-Based Notes",
    "Customer-Friendly Summary",
    "Safety Disclaimer",
]


def build_maintenance_prompt(
    predicted_label: str,
    mileage: int | None = None,
    user_concern: str | None = None,
) -> str:
    """Build a structured maintenance advisory prompt.

    Args:
        predicted_label: Predicted Stanford Cars class label.
        mileage: Optional vehicle mileage.
        user_concern: Optional free-text concern from the user.

    Returns:
        Prompt text for the LLM.

    Raises:
        ValueError: If ``predicted_label`` is empty.
    """
    label = predicted_label.strip()
    if not label:
        raise ValueError("predicted_label must not be empty.")

    mileage_text = f"{mileage:,} miles" if mileage is not None else "not provided"
    concern_text = user_concern.strip() if user_concern and user_concern.strip() else "not provided"
    section_list = "\n".join(f"{index}. {section}" for index, section in enumerate(REPORT_SECTIONS, start=1))

    return f"""You are an automotive maintenance advisor.

Predicted vehicle label: {label}
Mileage: {mileage_text}
Customer concern: {concern_text}

Write an original, practical maintenance advisory report for the predicted vehicle.

Rules:
- Use the predicted vehicle label exactly as provided: {label}
- Do not change, contradict, or invent a different make, model, body style, or year.
- If the mileage is not provided, keep mileage advice general.
- Do not quote or reproduce manufacturer manuals, service documents, or copyrighted text.
- Avoid claiming certainty about defects; phrase known issues as possibilities to inspect.
- Keep the report customer-friendly and concise.
- Use exactly these section headings, in this order:
{section_list}

Report:"""