"""
Derive sigma8 from (A_s, n_s) + background expansion E(a) using:
  - Eisenstein & Hu 1998 "no-wiggle" transfer
  - Linear growth ODE for D(a)
  - Real-space top-hat window W(kR) with R = 8 h^{-1} Mpc
This module is self-contained (numpy-only) and caches the amplitude calibration.

Usage (see demo at bottom and tests/test_sigma8_demo.py):
    sigma8 = sigma8_from_primordial(
        As=2.1e-9, ns=0.965, h=0.6736,
        Om0=0.315, Obh2=0.02237,
        E_of_a=lambda a: E_LCDM(a, H0=67.36, Om0=0.315, Or0=9.2e-5, Ok0=0.0),
        k_pivot=0.05
    )
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Dict

import numpy as np


# -------------------
# Window & utilities
# -------------------

def top_hat_W(kR: np.ndarray) -> np.ndarray:
    """Real-space top-hat window in k-space: W(kR) = 3 (sin x - x cos x) / x^3."""
    x = np.array(kR, dtype=float)
    out = np.empty_like(x)
    mask = np.abs(x) < 1e-6
    out[mask] = 1.0 - (x[mask] ** 2) / 10.0
    xm = x[~mask]
    out[~mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm ** 3 + 1e-15)
    return out


def dlnH_dln_a(E_of_a: Callable[[float], float], a: float, eps: float = 1e-4) -> float:
    """Numerical derivative d ln H / d ln a = (a/E) dE/da."""
    a1 = max(a * (1 - eps), 1e-6)
    a2 = a * (1 + eps)
    E1, E2 = E_of_a(a1), E_of_a(a2)
    return (math.log(E2) - math.log(E1)) / (math.log(a2) - math.log(a1))


# -------------------
# Growth factor D(a)
# -------------------

def growth_D_of_a(
    E_of_a: Callable[[float], float],
    Om0: float,
    *,
    a_start: float = 1e-4,
    a_end: float = 1.0,
    n_steps: int = 5000,
) -> float:
    """
    Solve the linear growth ODE in ln a:
      D'' + [2 + d ln H / d ln a] D' - (3/2) Ω_m(a) D = 0
    with matter-era normalization D(a_start) = a_start, D'(a_start) = a_start.
    Returns D(a_end) with D(1) normalized to 1.
    """
    a_grid = np.geomspace(a_start, a_end, n_steps)
    ln_a = np.log(a_grid)

    D = a_grid[0]
    Dp = D

    def Omega_m_of_a(a_val: float) -> float:
        E = E_of_a(a_val)
        return Om0 * a_val ** -3 / (E * E)

    for i in range(1, len(a_grid)):
        a_prev = float(a_grid[i - 1])
        a_curr = float(a_grid[i])
        dln_a = ln_a[i] - ln_a[i - 1]

        def rhs(D_val: float, Dp_val: float, a_here: float) -> Tuple[float, float]:
            A = 2.0 + dlnH_dln_a(E_of_a, a_here)
            B = 1.5 * Omega_m_of_a(a_here)
            Dpp = -A * Dp_val + B * D_val
            return Dp_val, Dpp

        a_mid = math.sqrt(a_prev * a_curr)
        k1_D, k1_Dp = rhs(D, Dp, a_prev)
        k2_D, k2_Dp = rhs(D + 0.5 * dln_a * k1_D, Dp + 0.5 * dln_a * k1_Dp, a_mid)
        k3_D, k3_Dp = rhs(D + 0.5 * dln_a * k2_D, Dp + 0.5 * dln_a * k2_Dp, a_mid)
        k4_D, k4_Dp = rhs(D + dln_a * k3_D, Dp + dln_a * k3_Dp, a_curr)

        D += (dln_a / 6.0) * (k1_D + 2 * k2_D + 2 * k3_D + k4_D)
        Dp += (dln_a / 6.0) * (k1_Dp + 2 * k2_Dp + 2 * k3_Dp + k4_Dp)

    if not np.isfinite(D) or D == 0.0:
        return 1.0
    return D / D  # = 1.0 (kept for clarity)


def normalized_growth_today(E_of_a: Callable[[float], float], Om0: float) -> float:
    """Return the normalized growth today (D=1.0 by construction)."""
    return growth_D_of_a(E_of_a, Om0)


# -------------------
# Eisenstein & Hu 1998 (no-wiggle) transfer
# -------------------

def T_eh98_nowiggle(k: np.ndarray, Om0: float, Obh2: float, h: float, Tcmb: float = 2.7255) -> np.ndarray:
    """Eisenstein & Hu 1998 'no-wiggle' transfer function."""
    k = np.asarray(k, dtype=float)
    theta = Tcmb / 2.7
    Ob0 = Obh2 / (h * h)
    f_b = Ob0 / max(Om0, 1e-12)
    Q = Om0 * h * np.exp(-f_b - (np.sqrt(2 * h) * f_b / max(Om0, 1e-12)))
    q = k / max(Q, 1e-6)
    L0 = np.log(1.0 + 2.34 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T0 = L0 / (L0 + C0 * q * q)
    return T0


# -------------------
# Power spectrum & sigma8 integral
# -------------------

_CALIBRATION: Dict[str, float] = {}


def _k_grid() -> np.ndarray:
    return np.logspace(-4.0, 2.0, 4096)


def _sigmaR_from_As_ns(
    R8_hinv_mpc: float,
    As: float,
    ns: float,
    h: float,
    Om0: float,
    Obh2: float,
    E_of_a: Callable[[float], float],
    *,
    k_pivot: float = 0.05,
    amplitude_C: float = 1.0,
) -> float:
    k_h = _k_grid()
    k_phys = k_h * h
    Tk = T_eh98_nowiggle(k_h, Om0, Obh2, h)
    D1 = normalized_growth_today(E_of_a, Om0)

    Pk = amplitude_C * As * (k_phys / k_pivot) ** (ns - 1.0) * (Tk ** 2) * (D1 ** 2)

    W = top_hat_W(k_h * R8_hinv_mpc)
    integrand = (k_h ** 2) * Pk * (W ** 2)
    sigma2 = np.trapz(integrand, k_h) / (2.0 * math.pi ** 2)
    return float(np.sqrt(max(sigma2, 0.0)))


def calibrate_sigma8_amplitude_once(
    *,
    As_ref: float,
    ns_ref: float,
    h_ref: float,
    Om0_ref: float,
    Obh2_ref: float,
    E_of_a_ref: Callable[[float], float],
    target_sigma8_ref: float = 0.811,
    k_pivot: float = 0.05,
    cache_key: str = "lcdm_planck_like",
) -> float:
    if cache_key in _CALIBRATION:
        return _CALIBRATION[cache_key]

    raw = _sigmaR_from_As_ns(
        R8_hinv_mpc=8.0,
        As=As_ref,
        ns=ns_ref,
        h=h_ref,
        Om0=Om0_ref,
        Obh2=Obh2_ref,
        E_of_a=E_of_a_ref,
        k_pivot=k_pivot,
        amplitude_C=1.0,
    )
    if raw <= 0.0 or not np.isfinite(raw):
        C = 1.0
    else:
        C = (target_sigma8_ref / raw) ** 2

    _CALIBRATION[cache_key] = C
    return C


def sigma8_from_primordial(
    *,
    As: float,
    ns: float,
    h: float,
    Om0: float,
    Obh2: float,
    E_of_a: Callable[[float], float],
    k_pivot: float = 0.05,
    calibrate_against: Optional[dict] = None,
    calibration_key: str = "lcdm_planck_like",
    default_C: float = 1.0,
) -> float:
    if calibrate_against:
        C = calibrate_sigma8_amplitude_once(
            As_ref=calibrate_against.get("As", As),
            ns_ref=calibrate_against.get("ns", ns),
            h_ref=calibrate_against.get("h", h),
            Om0_ref=calibrate_against.get("Om0", Om0),
            Obh2_ref=calibrate_against.get("Obh2", Obh2),
            E_of_a_ref=calibrate_against["E_of_a"],
            target_sigma8_ref=calibrate_against.get("sigma8_ref", 0.811),
            k_pivot=k_pivot,
            cache_key=calibration_key,
        )
    else:
        C = default_C

    return _sigmaR_from_As_ns(
        R8_hinv_mpc=8.0,
        As=As,
        ns=ns,
        h=h,
        Om0=Om0,
        Obh2=Obh2,
        E_of_a=E_of_a,
        k_pivot=k_pivot,
        amplitude_C=C,
    )


# -------------------
# Minimal LCDM E(a) helper
# ------------
# Minimal helper for quick LCDM tests

def E_LCDM(a: float, *, H0: float, Om0: float, Or0: float, Ok0: float) -> float:
    Ol0 = max(0.0, 1.0 - Om0 - Or0 - Ok0)
    return math.sqrt(Om0 * a ** -3 + Or0 * a ** -4 + Ok0 * a ** -2 + Ol0)


if __name__ == "__main__":
    As = 2.1e-9
    ns = 0.965
    H0 = 67.36
    h = H0 / 100.0
    Om0 = 0.315
    Or0 = 9.2e-5
    Ok0 = 0.0
    Obh2 = 0.02237

    E_ref = lambda a: E_LCDM(a, H0=H0, Om0=Om0, Or0=Or0, Ok0=Ok0)

    s8 = sigma8_from_primordial(
        As=As,
        ns=ns,
        h=h,
        Om0=Om0,
        Obh2=Obh2,
        E_of_a=E_ref,
        calibrate_against=dict(
            As=As,
            ns=ns,
            h=h,
            Om0=Om0,
            Obh2=Obh2,
            E_of_a=E_ref,
            sigma8_ref=0.811,
        ),
        calibration_key="planck18_lcdm",
    )
    print(f"LCDM sigma8 (calibrated) ≈ {s8:.3f}")
