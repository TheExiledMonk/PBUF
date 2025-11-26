"""EDE-modified ΛCDM parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

_FREE_PARAMETERS: list[str] = [
    "H0",
    "Omega_b0",
    "Omega_m0",
    "Omega_k0",
    "Omega_r0",
    "Omega_EDE_0",
    "a_c",
    "n",
]

_DEFAULT_PARAMETERS: Dict[str, float] = {
    "H0": 67.5,
    "Omega_b0": 0.049,
    "Omega_m0": 0.30,
    "Omega_k0": 0.0,
    "Omega_r0": 9.0e-5,
    "Omega_EDE_0": 0.03,
    "a_c": 3.5e-4,
    "n": 3.0,
}

_PARAMETER_BOUNDS: List[Dict[str, float | str]] = [
    {"name": "H0", "lower": 40.0, "upper": 90.0, "prior": "flat"},
    {"name": "Omega_b0", "lower": 0.02, "upper": 0.08, "prior": "flat"},
    {"name": "Omega_m0", "lower": 0.05, "upper": 0.6, "prior": "flat"},
    {"name": "Omega_k0", "lower": -0.2, "upper": 0.2, "prior": "flat"},
    {"name": "Omega_r0", "lower": 9.0e-5, "upper": 9.0e-5, "prior": "fixed"},
    {"name": "Omega_EDE_0", "lower": 0.0, "upper": 0.2, "prior": "flat"},
    {"name": "a_c", "lower": 1.0e-5, "upper": 1.0e-2, "prior": "flat"},
    {"name": "n", "lower": 2.0, "upper": 10.0, "prior": "flat"},
]


@dataclass(frozen=True)
class EDELCDMParams:
    H0: float
    Omega_b0: float
    Omega_m0: float
    Omega_k0: float
    Omega_r0: float
    Omega_EDE_0: float
    a_c: float
    n: float


def get_free_parameters() -> list[str]:
    return list(_FREE_PARAMETERS)


def get_default_parameters() -> Dict[str, float]:
    return dict(_DEFAULT_PARAMETERS)


def get_parameter_bounds() -> List[Dict[str, float | str]]:
    return [dict(bound) for bound in _PARAMETER_BOUNDS]


def knobs() -> List[Dict[str, float | str]]:
    """Expose the current optimisable knobs for convenience."""

    return get_parameter_bounds()
