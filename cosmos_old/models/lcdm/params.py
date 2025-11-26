"""LCDM cosmological parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LCDMParams:
    H0: float
    Omega_m0: float
    Omega_r0: float
    Omega_k0: float
    Omega_b0: float
    Omega_lambda0: float | None = None
    sigma8_0: float = 0.811

    def with_lambda(self, Omega_lambda0: float) -> "LCDMParams":
        return replace(self, Omega_lambda0=Omega_lambda0)
