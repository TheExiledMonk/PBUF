"""Eisenstein & Hu (1998) matter transfer functions (wiggle + no-wiggle)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

_T_CMB = 2.7255  # K, only enters via theta = Tcmb/2.7
_SMALL = 1e-12


@dataclass
class _EHParams:
    omega_m_h2: float
    omega_b_h2: float
    h: float
    theta_cmb: float
    z_eq: float
    k_eq: float
    z_drag: float
    R_drag: float
    R_eq: float
    sound_horizon: float
    k_silk: float
    alpha_c: float
    beta_c: float
    alpha_b: float
    beta_b: float
    beta_node: float
    f_baryon: float
    f_cdm: float


def _compute_drag_epoch(omega_m_h2: float, omega_b_h2: float) -> float:
    """Return z_drag from the Eisenstein & Hu fitting formula."""

    b1 = 0.313 * omega_m_h2 ** -0.419 * (1 + 0.607 * omega_m_h2 ** 0.674)
    b2 = 0.238 * omega_m_h2 ** 0.223
    z_drag = (1291.0 * omega_m_h2 ** 0.251 / (1 + 0.659 * omega_m_h2 ** 0.828)) * (1 + b1 * omega_b_h2 ** b2)
    return float(z_drag)


def _G(y: float) -> float:
    """Eisenstein & Hu helper G(y) (eq. 15 in arXiv:astro-ph/9805239)."""

    sqrt_term = math.sqrt(1.0 + y)
    return y * (
        -6.0 * sqrt_term
        + (2.0 + 3.0 * y) * math.log((sqrt_term + 1.0) / max(sqrt_term - 1.0, _SMALL))
    )


def _derive_params(omega_m_h2: float, omega_b_h2: float, h: float, theta_cmb: float) -> _EHParams:
    omega_m_h2 = max(float(omega_m_h2), _SMALL)
    omega_b_h2 = max(float(omega_b_h2), _SMALL)
    h = max(float(h), _SMALL)
    theta_cmb = float(theta_cmb)
    omega_cdm_h2 = max(omega_m_h2 - omega_b_h2, _SMALL)
    f_baryon = min(max(omega_b_h2 / omega_m_h2, 0.0), 1.0)
    f_cdm = 1.0 - f_baryon

    z_eq = 2.50e4 * omega_m_h2 * theta_cmb ** -4
    k_eq = 0.0746 * omega_m_h2 * theta_cmb ** -2

    z_drag = _compute_drag_epoch(omega_m_h2, omega_b_h2)
    R_drag = 31.5 * omega_b_h2 * theta_cmb ** -4 * (1e3 / max(z_drag, _SMALL))
    R_eq = 31.5 * omega_b_h2 * theta_cmb ** -4 * (1e3 / max(z_eq, _SMALL))

    sound_horizon = (2.0 / (3.0 * k_eq)) * math.sqrt(6.0 / max(R_eq, _SMALL)) * math.log(
        (math.sqrt(1.0 + R_drag) + math.sqrt(R_drag + R_eq)) / (1.0 + math.sqrt(R_eq))
    )

    k_silk = 1.6 * omega_b_h2 ** 0.52 * omega_m_h2 ** 0.73 * (1 + (10.4 * omega_m_h2) ** -0.95)

    # CDM suppression factors (eqs. 16-19)
    a1 = (46.9 * omega_m_h2) ** 0.670 * (1 + (32.1 * omega_m_h2) ** -0.532)
    a2 = (12.0 * omega_m_h2) ** 0.424 * (1 + (45.0 * omega_m_h2) ** -0.582)
    alpha_c = a1 ** (-f_baryon) * a2 ** (-f_baryon ** 3)
    beta_c = 1.0 / (1.0 + 0.944 / (1.0 + (458.0 * omega_m_h2) ** -0.708))

    # Baryon features (eqs. 21-26)
    y_drag = (1.0 + z_eq) / max(1.0 + z_drag, _SMALL)
    alpha_b = 2.07 * k_eq * sound_horizon * (1 + R_drag) ** -0.75 * _G(y_drag)
    beta_b = 0.5 + f_baryon + (3.0 - 2.0 * f_baryon) * math.sqrt((17.2 * omega_m_h2) ** 2 + 1.0)
    beta_node = 8.41 * omega_m_h2 ** 0.435

    return _EHParams(
        omega_m_h2=omega_m_h2,
        omega_b_h2=omega_b_h2,
        h=h,
        theta_cmb=theta_cmb,
        z_eq=z_eq,
        k_eq=k_eq,
        z_drag=z_drag,
        R_drag=R_drag,
        R_eq=R_eq,
        sound_horizon=sound_horizon,
        k_silk=k_silk,
        alpha_c=alpha_c,
        beta_c=beta_c,
        alpha_b=alpha_b,
        beta_b=beta_b,
        beta_node=beta_node,
        f_baryon=f_baryon,
        f_cdm=f_cdm,
    )


class EisensteinHuTransfer:
    """
    Eisenstein & Hu transfer function for CDM+baryons (astro-ph/9805239).

    Inputs are in the usual h-scaled units (k in h/Mpc).
    """

    def __init__(self, omega_m_h2: float, omega_b_h2: float, h: float, *, T_cmb: float = _T_CMB) -> None:
        theta_cmb = float(T_cmb) / 2.7
        self._params = _derive_params(omega_m_h2, omega_b_h2, h, theta_cmb)

    @property
    def params(self) -> _EHParams:
        return self._params

    def _transfer_cdm(self, k: np.ndarray) -> np.ndarray:
        p = self._params
        q = k / (13.41 * p.k_eq)
        L0 = np.log(np.e + 1.8 * p.beta_c * q)
        C0 = 14.2 / p.alpha_c + 386.0 / (1.0 + 69.9 * q ** 1.08)
        return L0 / (L0 + C0 * q * q)

    def _transfer_baryon(self, k: np.ndarray, T_c: np.ndarray) -> np.ndarray:
        p = self._params
        ks = k * p.sound_horizon
        q = k / (13.41 * p.k_eq)
        L0 = np.log(np.e + 1.8 * q)
        C0 = 14.2 / p.alpha_c + 386.0 / (1.0 + 69.9 * q ** 1.08)
        T0 = L0 / (L0 + C0 * q * q)

        T_b1 = T0 / (1.0 + (ks / 5.2) ** 2)
        T_b2 = (p.alpha_b / (1.0 + (p.beta_b / ks) ** 3.0)) * np.exp(-(k / p.k_silk) ** 1.4)
        Tb = (T_b1 + T_b2) * np.sin(ks) / np.where(ks == 0.0, 1.0, ks)
        return Tb

    def transfer(self, k: Iterable[float] | float | np.ndarray, *, wiggles: bool = True) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(k, dtype=float))
        arr = np.clip(arr, _SMALL, np.inf)
        T_c = self._transfer_cdm(arr)
        if not wiggles or self._params.f_baryon <= 0.0:
            return T_c
        T_b = self._transfer_baryon(arr, T_c)
        return self._params.f_cdm * T_c + self._params.f_baryon * T_b


def eisenstein_hu_transfer(
    k: Iterable[float] | float | np.ndarray,
    omega_m_h2: float,
    omega_b_h2: float,
    h: float,
    *,
    wiggles: bool = True,
    T_cmb: float = _T_CMB,
) -> np.ndarray:
    """
    Convenience wrapper returning T(k) for the supplied cosmology.
    """

    transfer = EisensteinHuTransfer(omega_m_h2, omega_b_h2, h, T_cmb=T_cmb)
    return transfer.transfer(k, wiggles=wiggles)


__all__ = ["EisensteinHuTransfer", "eisenstein_hu_transfer"]
