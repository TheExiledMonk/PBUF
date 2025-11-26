"""
Default parameter values for the Quantum α scan CLI.

These replace the legacy ``quantum.core`` module so existing CLI entry points
keep working now that the engine was consolidated under ``quantum.*``.
"""

from __future__ import annotations

from typing import Dict, Tuple

ALPHA_BAND: Tuple[float, float] = (0.01, 0.05)
MIXING_STRENGTH_RANGE: Tuple[float, float] = (1.0e-4, 1.0)
DEFAULT_MIXING_SAMPLES: int = 400
EPSILON_0: float = 1.0

# Loop coefficients per regulator (dimensionless).
REGULATOR_COEFFICIENTS: Dict[str, float] = {
    "hard_cutoff": 0.019894367886486918,
    "covariant": 0.026525823848649224,
    "heat_kernel": 0.013262911924324612,
}

# Effective field content degrees (N_eff) per configuration.
FIELD_CONTENT_DEGREES: Dict[str, float] = {
    "SM_min": 80.0,
    "SM_full": 110.0,
    "SM_plus_heavy": 180.0,
}

__all__ = [
    "ALPHA_BAND",
    "MIXING_STRENGTH_RANGE",
    "DEFAULT_MIXING_SAMPLES",
    "EPSILON_0",
    "REGULATOR_COEFFICIENTS",
    "FIELD_CONTENT_DEGREES",
]
