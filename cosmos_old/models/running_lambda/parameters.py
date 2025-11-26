"""Parameter definitions for the running-Λ background."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

_FREE_PARAMETERS: List[str] = [
    "H0",
    "Omega_b0",
    "Omega_m0",
    "Omega_k0",
    "Omega_r0",
    "nu_lambda",
]

_DEFAULT_PARAMETERS: Dict[str, float] = {
    "H0": 67.5,
    "Omega_b0": 0.049,
    "Omega_m0": 0.315,
    "Omega_k0": 0.0,
    "Omega_r0": 9.0e-5,
    "nu_lambda": 0.0,
}

_PARAMETER_BOUNDS: List[Dict[str, float | str]] = [
    {"name": "H0", "lower": 40.0, "upper": 90.0, "prior": "flat"},
    {"name": "Omega_m0", "lower": 0.05, "upper": 0.6, "prior": "flat"},
    {"name": "Omega_b0", "lower": 0.02, "upper": 0.08, "prior": "flat"},
    {"name": "Omega_k0", "lower": -0.2, "upper": 0.2, "prior": "flat"},
    {"name": "Omega_r0", "lower": 9.0e-5, "upper": 9.0e-5, "prior": "fixed"},
    {"name": "nu_lambda", "lower": -0.1, "upper": 0.1, "prior": "flat"},
]


@dataclass(frozen=True)
class RunningLambdaParams:
    H0: float
    Omega_m0: float
    Omega_b0: float
    Omega_k0: float
    Omega_r0: float
    nu_lambda: float
    Omega_lambda0: float | None = None
    sigma8_0: float = 0.811

    def with_lambda(self, Omega_lambda0: float) -> "RunningLambdaParams":
        return replace(self, Omega_lambda0=Omega_lambda0)

    @property
    def Omega_lambda(self) -> float:
        if self.Omega_lambda0 is not None:
            return self.Omega_lambda0
        return 1.0 - (self.Omega_m0 + self.Omega_r0 + self.Omega_k0)


def get_free_parameters() -> List[str]:
    return list(_FREE_PARAMETERS)


def get_default_parameters() -> Dict[str, float]:
    return dict(_DEFAULT_PARAMETERS)


def get_parameter_bounds() -> List[Dict[str, float | str]]:
    return [dict(bound) for bound in _PARAMETER_BOUNDS]
