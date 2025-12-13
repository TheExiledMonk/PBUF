"""Linear matter power spectrum using an Eisenstein–Hu transfer and model growth."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .transfer_eh import EisensteinHuTransfer

_SMALL = 1e-12


class LinearPowerSpectrum:
    """
    P_lin(k, z) = A_s (k/k_*)^{n_s-1} T^2(k) D^2(z).

    - T(k) comes from Eisenstein & Hu (wiggle-aware) with baryon and CDM inputs.
    - Growth D(z) is supplied by the backend model; fallback is 1 / (1 + z).
    """

    def __init__(
        self,
        omega_m0: float,
        omega_b0: float,
        H0: float,
        *,
        A_s: float | None,
        n_s: float,
        k_pivot: float,
        growth_function: Callable[[Iterable[float] | float], np.ndarray],
        sigma8_fallback: float,
    ) -> None:
        self._omega_m0 = float(omega_m0)
        self._omega_b0 = max(float(omega_b0), 0.0)
        self._H0 = float(H0)
        self._A_s = float(A_s) if A_s is not None else None
        self._n_s = float(n_s)
        self._k_pivot = max(float(k_pivot), _SMALL)
        self._growth = growth_function
        self._sigma8_fallback = max(float(sigma8_fallback), _SMALL)
        h = self._H0 / 100.0
        omega_m_h2 = self._omega_m0 * h * h
        omega_b_h2 = self._omega_b0 * h * h
        self._transfer = EisensteinHuTransfer(omega_m_h2, omega_b_h2, h)

    def _growth_factor(self, z: Iterable[float] | float | np.ndarray) -> np.ndarray:
        try:
            return np.asarray(self._growth(z), dtype=float)
        except Exception:
            arr = np.asarray(z, dtype=float)
            return 1.0 / (1.0 + arr)

    def _base_power(self, k: np.ndarray) -> np.ndarray:
        exponent = self._n_s - 1.0
        scale = np.power(np.clip(k, _SMALL, np.inf) / self._k_pivot, exponent)
        if self._A_s is not None:
            return self._A_s * scale
        return (self._sigma8_fallback ** 2) * scale

    def linear_pk(
        self,
        k: Iterable[float] | float | np.ndarray,
        z: Iterable[float] | float | np.ndarray,
        *,
        wiggles: bool = True,
    ) -> np.ndarray:
        """
        Return P_lin(k, z) evaluated either element-wise (matching shapes) or with broadcasting.
        """

        k_arr = np.asarray(k, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        T = self._transfer.transfer(k_arr, wiggles=wiggles)
        base = self._base_power(k_arr) * np.square(T)
        growth = self._growth_factor(z_arr)
        if k_arr.shape == z_arr.shape:
            return np.clip(base * np.square(growth), 0.0, np.inf)
        return np.clip(np.square(growth)[..., None] * base, 0.0, np.inf)


__all__ = ["LinearPowerSpectrum"]
