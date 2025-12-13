"""Intrinsic-alignment helpers (NLA-style amplitude scaffolding)."""

from __future__ import annotations

import numpy as np

_Z0_DEFAULT = 0.62
_C1_RHOCRIT_DEFAULT = 0.01389  # ≈ 5e-14 h^-2 Msun^-1 Mpc^3 * ρ_crit,0
_SMALL = 1e-8


class IntrinsicAlignmentModel:
    """
    Nonlinear alignment (NLA) model:
      F(z) = -A_IA * C1 * ρ_crit * Ω_m0 / D(z) * [(1+z)/(1+z0)]^{η_IA}.

    growth_function should return the linear growth factor normalized to D(0)=1.
    """

    def __init__(
        self,
        A_IA: float = 0.0,
        eta_IA: float = 0.0,
        z0: float = _Z0_DEFAULT,
        C1_rho_crit: float = _C1_RHOCRIT_DEFAULT,
        omega_m0: float = 0.3,
        growth_function=None,
    ) -> None:
        self.A_IA = float(A_IA)
        self.eta_IA = float(eta_IA)
        self.z0 = float(z0 if z0 is not None else _Z0_DEFAULT)
        self.C1_rho_crit = float(C1_rho_crit)
        self.omega_m0 = float(omega_m0)
        self._growth_fn = growth_function

    def _growth(self, z: np.ndarray) -> np.ndarray:
        if self._growth_fn is None:
            return 1.0 / (1.0 + z)
        try:
            return np.asarray(self._growth_fn(z), dtype=float)
        except Exception:
            return 1.0 / (1.0 + z)

    def amplitude(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        D = np.clip(self._growth(z_arr), _SMALL, np.inf)
        factor = np.power((1.0 + z_arr) / (1.0 + self.z0), self.eta_IA)
        return -self.A_IA * self.C1_rho_crit * self.omega_m0 * factor / D

    def gi_power(self, P_lin: np.ndarray, z: np.ndarray) -> np.ndarray:
        return self.amplitude(z) * P_lin

    def ii_power(self, P_lin: np.ndarray, z: np.ndarray) -> np.ndarray:
        amp = self.amplitude(z)
        return (amp * amp) * P_lin


__all__ = ["IntrinsicAlignmentModel", "_C1_RHOCRIT_DEFAULT"]
