"""
Formatting helpers shared by report generation panels.
"""

from typing import Any


def format_number(value: Any, spec: str) -> str:
    """Safely format numeric values, falling back to str otherwise."""
    if isinstance(value, (int, float)):
        try:
            return format(value, spec)
        except Exception:
            pass
    return str(value)
