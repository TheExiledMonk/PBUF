"""Parameter definitions for the DESI modified ΛCDM expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

_FREE_PARAMS = [
    "H0",
    "Omega_m0",
    "Omega_b0",
    "Omega_k0",
    "Omega_r0",
    "w0",
    "wa",
]

_DEFAULT_PARAMETERS: Dict[str, float] = {
    "H0": 67.0,
    "Omega_m0": 0.3,
    "Omega_b0": 0.049,
    "Omega_k0": 0.0,
    "Omega_r0": 9.0e-5,
    "w0": -1.0,
    "wa": 0.0,
}

_PARAMETER_BOUNDS: List[Dict[str, float | str]] = [
    {"name": "H0", "lower": 40.0, "upper": 90.0, "prior": "flat"},
    {"name": "Omega_m0", "lower": 0.01, "upper": 0.7, "prior": "flat"},
    {"name": "Omega_b0", "lower": 0.02, "upper": 0.08, "prior": "flat"},
    {"name": "Omega_k0", "lower": -0.2, "upper": 0.2, "prior": "flat"},
    {"name": "Omega_r0", "lower": 9.0e-5, "upper": 9.0e-5, "prior": "fixed"},
    {"name": "w0", "lower": -2.0, "upper": -0.2, "prior": "flat"},
    {"name": "wa", "lower": -2.0, "upper": 2.0, "prior": "flat"},
]

@dataclass(frozen=True)
class DESIModParams:
    H0: float
    Omega_m0: float
    Omega_b0: float
    Omega_k0: float
    Omega_r0: float
    w0: float
    wa: float

    @property
    def Omega_DE0(self) -> float:
        """Derived late-time dark-energy fraction from closure."""

        return 1.0 - (self.Omega_m0 + self.Omega_r0 + self.Omega_k0)


def get_free_parameters() -> List[str]:
    return list(_FREE_PARAMS)


def get_default_parameters() -> Dict[str, float]:
    return dict(_DEFAULT_PARAMETERS)


def get_parameter_bounds() -> List[Dict[str, float | str]]:
    return [dict(bound) for bound in _PARAMETER_BOUNDS]
