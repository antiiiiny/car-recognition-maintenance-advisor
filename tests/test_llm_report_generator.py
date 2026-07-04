"""Tests for maintenance report generation."""

from __future__ import annotations

from src.llm.fake_client import FakeMaintenanceClient
from src.llm.prompts import REPORT_SECTIONS
from src.llm.report_generator import generate_maintenance_report


TEST_LABELS = [
    "Acura TL Sedan 2012",
    "BMW M3 Coupe 2012",
    "Audi R8 Coupe 2012",
    "Dodge Charger Sedan 2012",
    "smart fortwo Convertible 2012",
]


def test_fake_reports_are_structurally_consistent_for_five_labels() -> None:
    """Verify fake reports cover Stage 3's five-label structure requirement."""
    client = FakeMaintenanceClient()
    reports = [
        generate_maintenance_report(label, mileage=85000, user_concern="noise", llm_client=client)
        for label in TEST_LABELS
    ]

    for label, report in zip(TEST_LABELS, reports, strict=True):
        assert label in report
        for section in REPORT_SECTIONS:
            assert section in report

    section_orders = [
        [report.index(section) for section in REPORT_SECTIONS]
        for report in reports
    ]
    assert all(order == sorted(order) for order in section_orders)


def test_fake_report_does_not_contradict_input_label() -> None:
    """Verify fake reports echo the exact label and avoid unrelated labels."""
    report = generate_maintenance_report(
        "Audi R8 Coupe 2012",
        llm_client=FakeMaintenanceClient(),
    )

    assert "Audi R8 Coupe 2012" in report
    assert "BMW M3 Coupe 2012" not in report