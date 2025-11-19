"""Logging configuration helpers."""
from __future__ import annotations

import logging

def configure_logging(level: str) -> None:
    """Configure standard logging with timestamp and level."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

