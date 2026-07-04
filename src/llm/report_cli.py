"""Command-line interface for generating maintenance reports."""

from __future__ import annotations

import argparse

from src.llm.fake_client import FakeMaintenanceClient
from src.llm.report_generator import generate_maintenance_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a vehicle maintenance advisory report.")
    parser.add_argument("--label", required=True, help="Predicted Stanford Cars class label.")
    parser.add_argument("--mileage", type=int, default=None, help="Optional vehicle mileage.")
    parser.add_argument("--concern", default=None, help="Optional customer concern.")
    parser.add_argument("--fake", action="store_true", help="Use deterministic fake output instead of OpenAI.")
    return parser.parse_args()


def main() -> None:
    """Generate and print a maintenance report."""
    args = parse_args()
    client = FakeMaintenanceClient() if args.fake else None
    report = generate_maintenance_report(
        predicted_label=args.label,
        mileage=args.mileage,
        user_concern=args.concern,
        llm_client=client,
    )
    print(report)


if __name__ == "__main__":
    main()