#!/usr/bin/env python3
"""
pbuf_thermal_reference.py

Standalone reference math for PBUF with thermal LUT:
- T(a), eps0(T), alpha(T), g_star(T), g_starS(T) interpolation
- Omega_r(a) from g_star and T(a)
- Omega_sigma(a) from alpha(T), eps0(T), Rmax, k_sat
- Closure to get Omega_k0 from today's budget
- E^2(a) and H(a)
- Optional sound horizon integral r_s(a)

Use this as a cross-check against cosmos2 PBUF kernel.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

C_LIGHT_KMS = 299792.458  # km/s


# ----------------------------
# 1. Interpolation utilities
# ----------------------------

def interp_scalar(x: float, x_grid: np.ndarray, y_grid: np.ndarray) -> float:
    """Simple 1D linear interpolation for a scalar x."""
    return float(np.interp(x, x_grid, y_grid))


def interp_array(x: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Interpolation for an array of x values."""
    return np.interp(x, x_grid, y_grid)


# ----------------------------
# 2. LUT container
# ----------------------------

@dataclass
class ThermalLUT:
    a: np.ndarray           # scale factor grid
    T: np.ndarray           # temperature [K]
    eps0_T: np.ndarray      # epsilon0(T)
    alpha_T: np.ndarray     # alpha(T)
    g_star: np.ndarray      # g*(T)
    g_starS: np.ndarray     # g*_S(T)


# ----------------------------
# 3. Interpolation from LUT
# ----------------------------

def get_T_eps_alpha_g(a: np.ndarray, lut: ThermalLUT) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given an array of a values, interpolate T(a), eps0(a), alpha(a),
    g_star(a), g_starS(a) from the LUT.
    """
    T_a       = interp_array(a, lut.a, lut.T)
    eps0_a    = interp_array(a, lut.a, lut.eps0_T)
    alpha_a   = interp_array(a, lut.a, lut.alpha_T)
    gstar_a   = interp_array(a, lut.a, lut.g_star)
    gstarS_a  = interp_array(a, lut.a, lut.g_starS)
    return T_a, eps0_a, alpha_a, gstar_a, gstarS_a


# ----------------------------
# 4. Radiation sector
# ----------------------------

def omega_r_of_a(
    a: np.ndarray,
    omega_r0_today: float,
    T_a: np.ndarray,
    gstar_a: np.ndarray,
    T0: float,
    gstar0: float,
) -> np.ndarray:
    """
    Radiation density as a function of a using g* and T(a).

    We normalize so that Omega_r(a=1) = omega_r0_today, and scale with
    (g* / g*_0) * (T / T0)^4.
    """
    ratio_T = T_a / T0
    ratio_g = gstar_a / gstar0
    return omega_r0_today * ratio_g * ratio_T**4


def omega_r_today_from_lut(
    lut: ThermalLUT,
    omega_r0_today: float,
    T0: float,
) -> Tuple[float, float]:
    """
    Helper to get gstar(T0) from LUT and return (Omega_r(today), gstar0).
    For this reference script we treat omega_r0_today as input and
    just return it with the matched gstar0 for consistency.
    """
    # Today is a=1
    gstar0 = interp_scalar(1.0, lut.a, lut.g_star)
    return omega_r0_today, gstar0


# ----------------------------
# 5. Elastic sector
# ----------------------------

def omega_sigma_of_a(
    a: np.ndarray,
    alpha_a: np.ndarray,
    eps0_a: np.ndarray,
    Rmax: float,
    k_sat: float,
) -> np.ndarray:
    """
    Elastic component Ω_sigma(a) using the PBUF-style elastic law:
        Ω_sigma(a) = alpha(a) * (1/eps0(a)) * (a / Rmax) * (1 - exp(-k_sat * a))

    This matches the conceptual spec we are using for PBUF.
    """
    a = np.asarray(a)
    factor = (1.0 / np.clip(eps0_a, 1e-12, None)) * (a / Rmax) * (1.0 - np.exp(-k_sat * a))
    return alpha_a * factor


# ----------------------------
# 6. Closure and E^2 / H(a)
# ----------------------------

def compute_omega_k0(
    omega_m0: float,
    omega_r_today: float,
    omega_sigma_today: float,
) -> float:
    """
    Closure condition at a=1:
        1 = Ω_m0 + Ω_r(today) + Ω_k0 + Ω_sigma(today)
    ->  Ω_k0 = 1 - Ω_m0 - Ω_r(today) - Ω_sigma(today)
    """
    return 1.0 - omega_m0 - omega_r_today - omega_sigma_today


def E2_of_a(
    a: np.ndarray,
    omega_m0: float,
    omega_k0: float,
    omega_r_a: np.ndarray,
    omega_sigma_a: np.ndarray,
) -> np.ndarray:
    """
    Dimensionless E^2(a) = (H(a)/H0)^2:
        E^2(a) = Ω_m0 a^-3 + Ω_r(a) + Ω_k0 a^-2 + Ω_sigma(a)
    """
    a = np.asarray(a)
    term_m = omega_m0 / np.clip(a**3, 1e-30, None)
    term_k = omega_k0 / np.clip(a**2, 1e-30, None)
    return term_m + term_k + omega_r_a + omega_sigma_a


def H_of_a(
    a: np.ndarray,
    H0: float,
    E2_a: np.ndarray,
) -> np.ndarray:
    """H(a) in km/s/Mpc."""
    return H0 * np.sqrt(np.clip(E2_a, 0.0, None))


# ----------------------------
# 7. Sound horizon r_s (optional)
# ----------------------------

def R_baryon(a: np.ndarray, omega_b0: float, T_a: np.ndarray, T0: float) -> np.ndarray:
    """
    Approximate baryon-to-photon ratio:
        R(a) = 3 rho_b / (4 rho_gamma)
             ∝ Ω_b0 a^-3 / (T(a)^4 / T0^4)
    We normalize with today's photon temperature T0.
    This is a rough reference version.
    """
    a = np.asarray(a)
    rho_b = omega_b0 / np.clip(a**3, 1e-30, None)
    # photon part ∝ T^4, but we only need ratio relative to today
    rho_gamma = (T_a / T0) ** 4
    return 3.0 * rho_b / (4.0 * np.clip(rho_gamma, 1e-30, None))


def sound_speed(a: np.ndarray, R_a: np.ndarray) -> np.ndarray:
    """
    c_s(a) = c / sqrt(3 (1 + R(a))).
    Return in km/s, consistent with H units.
    """
    return C_LIGHT_KMS / np.sqrt(3.0 * (1.0 + R_a))


def r_s_integral(
    a_grid: np.ndarray,
    H_a: np.ndarray,
    omega_b0: float,
    T_a: np.ndarray,
    T0: float,
) -> float:
    """
    Compute sound horizon r_s by integrating:
        r_s = ∫ c_s(a) / (a^2 H(a)) da
    over a_grid from a_min to a_decoupling.

    For this reference script we integrate over the full a_grid,
    assuming decoupling happens within its range.
    """
    a = np.asarray(a_grid)
    R_a = R_baryon(a, omega_b0, T_a, T0)
    c_s = sound_speed(a, R_a)
    integrand = c_s / (np.clip(a**2 * H_a, 1e-30, None))

    # simple trapezoidal integration over a
    return float(np.trapz(integrand, a))


# ----------------------------
# 8. Example / sanity run
# ----------------------------

def main():
    # Example dummy parameters (you can change these)
    H0 = 67.4
    omega_m0 = 0.315
    omega_b0 = 0.049
    omega_r0_today = 9e-5   # Planck-ish
    Rmax = 1.0e6
    k_sat = 0.98
    T0 = 2.7255

    # Dummy LUT grids (in practice replace with real LUT loaded from JSON)
    # Here we make a log-spaced a-grid and simple toy profiles.
    lut_a = np.logspace(-6, 0, 256)
    lut_T = T0 / lut_a  # simple T ~ 1/a scaling
    lut_eps0_T = np.ones_like(lut_a)  # flat epsilon0 = 1
    lut_alpha_T = np.full_like(lut_a, 0.022)  # flat alpha
    # toy g*, g*_S
    lut_gstar = np.full_like(lut_a, 3.36)
    lut_gstarS = np.full_like(lut_a, 3.9)

    lut = ThermalLUT(
        a=lut_a,
        T=lut_T,
        eps0_T=lut_eps0_T,
        alpha_T=lut_alpha_T,
        g_star=lut_gstar,
        g_starS=lut_gstarS,
    )

    # Build an evaluation grid
    a_eval = np.array([1e-3, 1e-2, 1e-1, 0.3, 0.5, 1.0])

    # Interpolate thermal fields
    T_a, eps0_a, alpha_a, gstar_a, gstarS_a = get_T_eps_alpha_g(a_eval, lut)

    # Radiation today + gstar0
    omega_r_today, gstar0 = omega_r_today_from_lut(
        lut=lut,
        omega_r0_today=omega_r0_today,
        T0=T0,
    )

    # Omega_r(a)
    omega_r_a = omega_r_of_a(
        a=a_eval,
        omega_r0_today=omega_r_today,
        T_a=T_a,
        gstar_a=gstar_a,
        T0=T0,
        gstar0=gstar0,
    )

    # Elastic Ω_sigma(a)
    omega_sigma_a = omega_sigma_of_a(
        a=a_eval,
        alpha_a=alpha_a,
        eps0_a=eps0_a,
        Rmax=Rmax,
        k_sat=k_sat,
    )

    # Today's elastic + radiation term (at a=1) for closure
    # (just pick the last element where a_eval includes 1.0)
    idx_today = np.argmax(a_eval)
    omega_sigma_today = omega_sigma_a[idx_today]
    omega_r_today_eff = omega_r_a[idx_today]

    omega_k0 = compute_omega_k0(
        omega_m0=omega_m0,
        omega_r_today=omega_r_today_eff,
        omega_sigma_today=omega_sigma_today,
    )

    # E^2(a) and H(a)
    E2 = E2_of_a(
        a=a_eval,
        omega_m0=omega_m0,
        omega_k0=omega_k0,
        omega_r_a=omega_r_a,
        omega_sigma_a=omega_sigma_a,
    )
    H_a = H_of_a(a_eval, H0=H0, E2_a=E2)

    # Print summary for manual comparison with cosmos2
    print("=== PBUF thermal reference check ===")
    print(f"H0 = {H0}")
    print(f"Omega_m0 = {omega_m0}")
    print(f"Omega_r0(today) (input) = {omega_r0_today}")
    print(f"Omega_r(today) (LUT-scaled) = {omega_r_today_eff}")
    print(f"Omega_sigma(today) = {omega_sigma_today}")
    print(f"Omega_k0 (derived) = {omega_k0}")
    print(f"Closure sum today = {omega_m0 + omega_r_today_eff + omega_sigma_today + omega_k0:.6f}")
    print()

    for a_val, T_val, E2_val, H_val, Or_val, Os_val in zip(a_eval, T_a, E2, H_a, omega_r_a, omega_sigma_a):
        print(
            f"a={a_val:.3e}  T(a)={T_val:.3e} K  "
            f"E^2(a)={E2_val:.6e}  H(a)={H_val:.3f}  "
            f"Omega_r(a)={Or_val:.3e}  Omega_sigma(a)={Os_val:.3e}"
        )

    # Optional: rough sound horizon with a finer grid
    a_grid = np.logspace(-4, 0, 2048)
    T_full, eps0_full, alpha_full, gstar_full, gstarS_full = get_T_eps_alpha_g(a_grid, lut)

    omega_r_full = omega_r_of_a(
        a=a_grid,
        omega_r0_today=omega_r_today,
        T_a=T_full,
        gstar_a=gstar_full,
        T0=T0,
        gstar0=gstar0,
    )
    omega_sigma_full = omega_sigma_of_a(
        a=a_grid,
        alpha_a=alpha_full,
        eps0_a=eps0_full,
        Rmax=Rmax,
        k_sat=k_sat,
    )
    E2_full = E2_of_a(
        a=a_grid,
        omega_m0=omega_m0,
        omega_k0=omega_k0,
        omega_r_a=omega_r_full,
        omega_sigma_a=omega_sigma_full,
    )
    H_full = H_of_a(a_grid, H0=H0, E2_a=E2_full)

    r_s_val = r_s_integral(
        a_grid=a_grid,
        H_a=H_full,
        omega_b0=omega_b0,
        T_a=T_full,
        T0=T0,
    )
    print()
    print(f"Approx sound horizon r_s (reference units) = {r_s_val:.6e}")


if __name__ == "__main__":
    main()
