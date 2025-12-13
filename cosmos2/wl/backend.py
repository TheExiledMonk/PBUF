"""Minimal WL backend adapter exposing distances, growth, and P(k,z)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from cosmos2.power import LinearPowerSpectrum, apply_halofit

# Avoid importing cosmos2.models to keep this module lightweight (SciPy-free).
C_LIGHT_KM_S = 299792.458


class WeakLensingBackend:
    """
    Thin adapter over cosmology models so the WL layer stays model-agnostic.

    Methods implemented:
      - chi_of_z(z)
      - H_of_z(z)
      - growth_D_of_z(z)
      - Omega_m_of_z(z)
      - P_m_of_kz(k, z, nonlinear=False)
    """

    def __init__(self, model: object) -> None:
        self._model = model
        params = getattr(model, "parameters", {}) or {}
        self._H0 = float(params.get("H0", 70.0))
        self._h = self._H0 / 100.0
        self._omega_b0 = float(params.get("Omega_b0", 0.049))
        try:
            self._omega_m0 = float(model.omega_m0())
        except Exception:
            self._omega_m0 = float(params.get("Omega_m0", 0.3))
        self._sigma8_today = self._resolve_sigma8(model, params)
        self._A_s = params.get("A_s")
        self._n_s = float(params.get("n_s", 1.0))
        self._k_pivot = float(params.get("k_pivot", 0.05))
        self._power = LinearPowerSpectrum(
            self._omega_m0,
            self._omega_b0,
            self._H0,
            A_s=self._A_s,
            n_s=self._n_s,
            k_pivot=self._k_pivot,
            growth_function=self.growth_D_of_z,
            sigma8_fallback=self._sigma8_today,
        )

    @staticmethod
    def _resolve_sigma8(model: object, params: dict) -> float:
        for attr in ("sigma8_today", "sigma8"):
            candidate = getattr(model, attr, None)
            if callable(candidate):
                try:
                    return float(candidate())
                except Exception:
                    continue
        return float(params.get("sigma8_0", 0.8))

    @property
    def H0(self) -> float:
        return self._H0

    @property
    def c_light(self) -> float:
        return float(C_LIGHT_KM_S)

    @property
    def omega_m0(self) -> float:
        return self._omega_m0

    def chi_of_z(self, z: Iterable[float] | float | np.ndarray) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        candidate = getattr(self._model, "DM", None)
        if callable(candidate):
            return np.asarray(candidate(arr), dtype=float)
        candidate = getattr(self._model, "comoving_distance", None)
        if callable(candidate):
            return np.asarray(candidate(arr), dtype=float)
        raise AttributeError("Model is missing DM/comoving_distance for WL backend.")

    def H_of_z(self, z: Iterable[float] | float | np.ndarray) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        candidate = getattr(self._model, "Hubble", None)
        if callable(candidate):
            return np.asarray(candidate(arr), dtype=float)
        raise AttributeError("Model is missing Hubble(z) for WL backend.")

    def growth_D_of_z(self, z: Iterable[float] | float | np.ndarray) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        candidate = getattr(self._model, "growth_factor", None)
        if callable(candidate):
            return np.asarray(candidate(arr), dtype=float)
        # Fallback: EdS-like decay with redshift.
        return 1.0 / (1.0 + arr)

    def Omega_m_of_z(self, z: Iterable[float] | float | np.ndarray) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        E = np.clip(self.H_of_z(arr) / max(self._H0, 1e-6), 1e-12, np.inf)
        return self._omega_m0 * np.power(1.0 + arr, 3) / (E * E)

    def P_m_of_kz(
        self,
        k: Iterable[float] | float | np.ndarray,
        z: Iterable[float] | float | np.ndarray,
        nonlinear: bool = False,
    ) -> np.ndarray:
        linear_pk = self._power.linear_pk(k, z, wiggles=True)
        if not nonlinear:
            return linear_pk
        return apply_halofit(k, z, linear_pk, omega_m0=self._omega_m0, h=self._h)


__all__ = ["WeakLensingBackend"]
