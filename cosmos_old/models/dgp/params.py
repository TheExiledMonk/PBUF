"""Parameters for the Dvali–Gabadadze–Porrati braneworld cosmology."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DGPParams:
    H0: float
    Omega_m0: float
    Omega_r0: float
    Omega_k0: float
    Omega_b0: float
    Omega_rc: float
    epsilon_branch: int = 1
    sigma8_0: float = 0.811


_DEFAULT_PARAMETERS: dict[str, float | int] = {
    "H0": 67.4,
    "Omega_m0": 0.315,
    "Omega_b0": 0.049,
    "Omega_r0": 9.0e-5,
    "Omega_k0": 0.0,
    "Omega_rc": 1e-3,
    "epsilon_branch": 1,
    "sigma8_0": 0.811,
}


def get_default_parameters() -> dict[str, float | int]:
    """Return a copy of the default DGP parameter set."""

    return dict(_DEFAULT_PARAMETERS)


__all__ = ["DGPParams", "get_default_parameters"]
