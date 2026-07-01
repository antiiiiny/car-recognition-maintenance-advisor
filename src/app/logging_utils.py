"""Logging utilities for the project."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logging with a readable format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
