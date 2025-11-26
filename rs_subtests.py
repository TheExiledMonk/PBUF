#!/usr/bin/env python3
"""
Sub-tests for the sound horizon r_s(a) pipeline.

This script builds a clean reference chain:

  T(a), g_star(a) -> Omega_r(a)
  Omega_r(a), Omega_m0, Omega_k0, Omega_sigma(a) -> H(a)
  T(a), g_star(a), Omega_b0 -> R(a)
  R(a) -> c_s(a)
  c_s(a), H(a) -> r_s integral

You can:
  - Run it as-is with simple T(a) and g_star(a) models, or
  - Plug in your LUT arrays (a_lut, T_lut, gstar_lut) from cosmos2.

Then compare each function against the corresponding cosmos2 implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


# ----------------------------------------------------------------------
# Basic constants
# ----------------------------------------------------------------------

C_LIGHT_KM_S = 299792.458  # km/s
T_CMB_0 = 2.7255           # K, present CMB temperature
GSTAR_0 = 3.36             # effective g_* today (photons + neutrinos)


@dataclass
class CosmoParams:
    H0: float         # km/s/Mpc
    Omega_m0: float
    Omega_b0: float
    Omega_r0: float   # present-day radiation fraction (photons+nu)
    Omega_k0: float
    # Optional elastic piece (can be turned off)
    Omega_sigma0: float = 0.0


# ----------------------------------------------------------------------
# A. Helper: interpolation for LUT-based fields
# ----------------------------------------------------------------------

def make_interpolator(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Simple 1D interpolation wrapper for monotonic x and y.
    Assumes x is sorted ascending.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def interp_fn(xq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=np.float64)
        return np.interp(xq, x, y)

    return interp_fn


# ----------------------------------------------------------------------
# B. Thermal sector: T(a), g_star(a), Omega_r(a)
# ----------------------------------------------------------------------

def T_of_a_simple(a: np.ndarray) -> np.ndarray:
    """
    Simple adiabatic scaling: T(a) = T0 / a.
    This is a fallback if you do not plug in a LUT.
    """
    a = np.asarray(a, dtype=np.float64)
    return T_CMB_0 / a


def make_T_of_a_from_lut(a_lut: np.ndarray, T_lut: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build T(a) interpolator from LUT in (a_lut, T_lut).
    """
    return make_interpolator(a_lut, T_lut)


def make_gstar_of_a_from_lut(a_lut: np.ndarray, gstar_lut: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build g_star(a) interpolator from LUT.
    """
    return make_interpolator(a_lut, gstar_lut)


def omega_r_of_a(
    a: np.ndarray,
    params: CosmoParams,
    T_of_a: Callable[[np.ndarray], np.ndarray],
    gstar_of_a: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Radiation density as a function of scale factor.

    Omega_r(a) = Omega_r0 * (g_star(a)/g_star0) * (T(a)/T0)^4

    This reduces to ~Omega_r0 / a^4 if T ~ 1/a and g_star constant,
    but this form lets you test LUT-driven behavior.
    """
    a = np.asarray(a, dtype=np.float64)
    T_a = T_of_a(a)
    g_a = gstar_of_a(a)

    factor_T = (T_a / T_CMB_0) ** 4
    factor_g = (g_a / GSTAR_0)

    return params.Omega_r0 * factor_T * factor_g


# ----------------------------------------------------------------------
# C. Baryon-photon ratio R(a) and sound speed c_s(a)
# ----------------------------------------------------------------------

def R_of_a(
    a: np.ndarray,
    params: CosmoParams,
    T_of_a: Callable[[np.ndarray], np.ndarray],
    gstar_of_a: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Baryon-to-photon momentum density ratio:

    R(a) = 3 rho_b / (4 rho_gamma)
         = (3 Omega_b0 / a^3) / (4 Omega_gamma(a))

    We approximate Omega_gamma(a) as part of Omega_r(a) scaled
    by temperature and g_star. This is slightly idealized but
    enough for sub-tests.
    """
    a = np.asarray(a, dtype=np.float64)

    # total radiation from omega_r_of_a
    omega_r_a = omega_r_of_a(a, params, T_of_a, gstar_of_a)

    # if you want to split photons vs neutrinos, you can add a factor here.
    # For now we treat omega_r_a as photon-dominated at early times.
    omega_gamma_a = omega_r_a

    rho_b = params.Omega_b0 / (a ** 3)
    rho_gamma = omega_gamma_a

    return (3.0 * rho_b) / (4.0 * rho_gamma)


def sound_speed_cs_of_a(R_a: np.ndarray) -> np.ndarray:
    """
    c_s(a) = c / sqrt(3 (1 + R(a)))
    """
    R_a = np.asarray(R_a, dtype=np.float64)
    return C_LIGHT_KM_S / np.sqrt(3.0 * (1.0 + R_a))


# ----------------------------------------------------------------------
# D. Hubble function H(a) given Omega sectors
# ----------------------------------------------------------------------

def omega_m_of_a(a: np.ndarray, params: CosmoParams) -> np.ndarray:
    return params.Omega_m0 / (a ** 3)


def omega_k_of_a(a: np.ndarray, params: CosmoParams) -> np.ndarray:
    return params.Omega_k0 / (a ** 2)


def omega_sigma_of_a_simple(a: np.ndarray, params: CosmoParams) -> np.ndarray:
    """
    Placeholder elastic term, here just taken as constant Omega_sigma0.
    Replace with your PBUF omega_sigma(a) if you want to mirror the code.
    """
    a = np.asarray(a, dtype=np.float64)
    return np.full_like(a, params.Omega_sigma0, dtype=np.float64)


def E2_of_a(
    a: np.ndarray,
    params: CosmoParams,
    omega_r_a: np.ndarray,
    omega_sigma_a: np.ndarray,
) -> np.ndarray:
    """
    Dimensionless expansion rate squared:

    E^2(a) = Omega_m(a) + Omega_r(a) + Omega_k(a) + Omega_sigma(a) + Omega_rest(a)

    Here we assume Omegas are normalized to close at a=1,
    so no explicit Omega_Lambda term appears.
    """
    a = np.asarray(a, dtype=np.float64)

    omega_m_a = omega_m_of_a(a, params)
    omega_k_a = omega_k_of_a(a, params)

    return omega_m_a + omega_r_a + omega_k_a + omega_sigma_a


def H_of_a(
    a: np.ndarray,
    params: CosmoParams,
    omega_r_a: np.ndarray,
    omega_sigma_a: np.ndarray,
) -> np.ndarray:
    """
    H(a) = H0 * sqrt(E^2(a))
    """
    E2 = E2_of_a(a, params, omega_r_a, omega_sigma_a)
    return params.H0 * np.sqrt(E2)


# ----------------------------------------------------------------------
# E. Sound horizon integral r_s
# ----------------------------------------------------------------------

def compute_rs(
    a_grid: np.ndarray,
    H_a: np.ndarray,
    c_s_a: np.ndarray,
    a_upper: float,
) -> float:
    """
    Compute r_s(a_upper) = \int_0^{a_upper} c_s(a) / (a^2 H(a)) da

    Assumes a_grid is sorted ascending and covers [a_min, a_max >= a_upper].
    Integrates with simple trapezoidal rule on a.
    """
    a_grid = np.asarray(a_grid, dtype=np.float64)
    H_a = np.asarray(H_a, dtype=np.float64)
    c_s_a = np.asarray(c_s_a, dtype=np.float64)

    # restrict to a <= a_upper
    mask = a_grid <= a_upper
    a_sub = a_grid[mask]
    H_sub = H_a[mask]
    cs_sub = c_s_a[mask]

    integrand = cs_sub / (a_sub ** 2 * H_sub)
    return np.trapz(integrand, a_sub)


def compute_rs_ref(
    params: CosmoParams,
    a_min: float = 1e-6,
    a_max: float = 1.0,
    n_grid: int = 10000,
    a_drag: float = 1.0 / (1.0 + 1059.0),  # approximate drag epoch
    T_of_a: Callable[[np.ndarray], np.ndarray] = None,
    gstar_of_a: Callable[[np.ndarray], np.ndarray] = None,
    omega_sigma_fn: Callable[[np.ndarray, CosmoParams], np.ndarray] = None,
) -> Tuple[float, dict]:
    """
    End-to-end r_s calculation using the reference pipeline.

    Returns:
      r_s       – scalar sound horizon
      diag      – dict with a_grid, H(a), c_s(a), Omega_r(a), R(a)
    """
    # grid
    a_grid = np.linspace(a_min, a_max, n_grid, dtype=np.float64)

    # choose default thermal sector if none provided
    if T_of_a is None:
        T_of_a = T_of_a_simple
    if gstar_of_a is None:
        # default: constant g_star = GSTAR_0
        def gstar_of_a(x):
            x = np.asarray(x, dtype=np.float64)
            return np.full_like(x, GSTAR_0, dtype=np.float64)

    if omega_sigma_fn is None:
        omega_sigma_fn = omega_sigma_of_a_simple

    # build components
    omega_r_a = omega_r_of_a(a_grid, params, T_of_a, gstar_of_a)
    omega_sigma_a = omega_sigma_fn(a_grid, params)
    H_a = H_of_a(a_grid, params, omega_r_a, omega_sigma_a)

    R_a = R_of_a(a_grid, params, T_of_a, gstar_of_a)
    c_s_a = sound_speed_cs_of_a(R_a)

    # compute r_s
    r_s_val = compute_rs(a_grid, H_a, c_s_a, a_upper=a_drag)

    diag = {
        "a_grid": a_grid,
        "H_a": H_a,
        "omega_r_a": omega_r_a,
        "omega_sigma_a": omega_sigma_a,
        "R_a": R_a,
        "c_s_a": c_s_a,
    }
    return r_s_val, diag


# ----------------------------------------------------------------------
# F. Simple driver: compare sub-functions vs your cosmos2 numbers
# ----------------------------------------------------------------------

def main():
    # Example parameters close to your usual ones
    params = CosmoParams(
        H0=67.4,
        Omega_m0=0.315,
        Omega_b0=0.049,
        Omega_r0=9.0e-5,
        Omega_k0=0.0,
        Omega_sigma0=0.0,
    )

    # Build reference r_s using simple T(a)=T0/a and constant g*
    r_s_val, diag = compute_rs_ref(params)

    print("Reference r_s (simple T(a), constant g*):", r_s_val, "[Mpc-ish units]")

    # Example: inspect sub-functions at a few sample points
    sample_a = np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0], dtype=np.float64)
    T_a = T_of_a_simple(sample_a)
    g_a = np.full_like(sample_a, GSTAR_0)
    omega_r_a = omega_r_of_a(sample_a, params, T_of_a_simple, lambda x: np.full_like(x, GSTAR_0))
    R_a = R_of_a(sample_a, params, T_of_a_simple, lambda x: np.full_like(x, GSTAR_0))
    c_s_a = sound_speed_cs_of_a(R_a)

    print("\nSample diagnostics:")
    print("a           ", sample_a)
    print("T(a) [K]    ", T_a)
    print("g*(a)       ", g_a)
    print("Omega_r(a)  ", omega_r_a)
    print("R(a)        ", R_a)
    print("c_s(a) [km/s]", c_s_a)

    # You can paste cosmos2 values here and compare by hand, for example:
    # cosmos2_H_a = np.array([...])  # from a debug print in the PBUFModel build
    # print("H_ref(a):", np.interp(sample_a, diag["a_grid"], diag["H_a"]))
    # print("H_cosmos2(a):", cosmos2_H_a)

if __name__ == "__main__":
    main()
