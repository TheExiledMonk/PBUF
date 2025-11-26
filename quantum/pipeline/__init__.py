"""Data pipeline package utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final

PIPELINE_ERROR_LOG: Final[Path] = Path("logs/pipeline_errors.txt")


def log_pipeline_error(message: str) -> None:
    """Append a timestamped message to the pipeline error log."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    entry = f"[{timestamp}] {message}\n"
    try:
        PIPELINE_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PIPELINE_ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError:
        # Logging must never raise; swallow filesystem errors silently.
        pass


__all__ = [
    "PIPELINE_ERROR_LOG",
    "log_pipeline_error",
]
