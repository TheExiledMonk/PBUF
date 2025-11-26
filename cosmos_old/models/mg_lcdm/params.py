"""MG-ΛCDM cosmological parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MGLCDMParams:
    H0: float
    Omega_m0: float
    Omega_r0: float
    Omega_k0: float
    Omega_b0: float
    Omega_lambda0: float | None = None
    sigma8_0: float = 0.811
    mu0: float = 0.0
    Sigma0: float = 0.0

    def with_lambda(self, Omega_lambda0: float) -> "MGLCDMParams":
        return replace(self, Omega_lambda0=Omega_lambda0)


_DEFAULT_PARAMETERS: dict[str, float] = {
    "H0": 67.4,
    "Omega_m0": 0.315,
    "Omega_b0": 0.049,
    "Omega_r0": 9.0e-5,
    "Omega_k0": 0.0,
    "sigma8_0": 0.811,
    "mu0": 0.0,
    "Sigma0": 0.0,
}


def get_default_parameters() -> dict[str, float]:
    """Return a copy of the default MG-ΛCDM parameter set."""

    return dict(_DEFAULT_PARAMETERS)
