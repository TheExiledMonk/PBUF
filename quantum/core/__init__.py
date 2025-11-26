"""
Compatibility layer exposing the quantum CLI constants/types that used to live
under ``quantum.core`` in earlier versions of the repository.
"""

from .constants import (  # noqa: F401
    ALPHA_BAND,
    DEFAULT_MIXING_SAMPLES,
    EPSILON_0,
    FIELD_CONTENT_DEGREES,
    MIXING_STRENGTH_RANGE,
    REGULATOR_COEFFICIENTS,
)
from .types import IslandSummary, ScanMetadata  # noqa: F401

__all__ = [
    "ALPHA_BAND",
    "DEFAULT_MIXING_SAMPLES",
    "EPSILON_0",
    "FIELD_CONTENT_DEGREES",
    "MIXING_STRENGTH_RANGE",
    "REGULATOR_COEFFICIENTS",
    "IslandSummary",
    "ScanMetadata",
]
