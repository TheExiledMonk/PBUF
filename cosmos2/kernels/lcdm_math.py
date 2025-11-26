"""LCDM background and growth kernel (Numba-ready)."""

import math

import numba
import numpy as np

from .common import growth

C_LIGHT = 299_792.458  # km/s
T_CMB = 2.7255


@numba.njit
def _z_drag_eh(Om0: float, Ob0: float, H0: float) -> float:
    """Eisenstein & Hu (1998) drag epoch approximation."""
    h = H0 / 100.0
    Obh2 = Ob0 * h * h
    Omh2 = Om0 * h * h
    b1 = 0.313 * Omh2 ** -0.419 * (1.0 + 0.607 * Omh2 ** 0.674)
    b2 = 0.238 * Omh2 ** 0.223
    numerator = 1291.0 * Omh2 ** 0.251
    denominator = 1.0 + 0.659 * Omh2 ** 0.828
    return numerator / denominator * (1.0 + b1 * Obh2 ** b2)


@numba.njit
def _omega_gamma0(H0: float) -> float:
    """Photon density today (Ω_γ)."""
    h = H0 / 100.0
    Omega_gamma_h2 = 2.469e-5 * (T_CMB / 2.7255) ** 4
    return Omega_gamma_h2 / (h * h)


@numba.njit
def _sound_speed(z: float, Ob0: float, Og0: float) -> float:
    """Baryon-photon sound speed c_s(z)."""
    R_b = 0.75 * (Ob0 / Og0) / (1.0 + z)
    return C_LIGHT / math.sqrt(3.0 * (1.0 + R_b))


@numba.njit
def _integrand_sound_horizon(a: float, Ob0: float, Og0: float, H0: float, Om0: float, Or0: float, Ok0: float, Ol0: float) -> float:
    """Integrand c_s / (a^2 H(a))."""
    a_val = max(a, 1.0e-8)
    z = 1.0 / a_val - 1.0
    c_s = _sound_speed(z, Ob0, Og0)
    inv_a = 1.0 / a_val
    E_sq = Om0 * inv_a ** 3 + Or0 * inv_a ** 4 + Ok0 * inv_a ** 2 + Ol0
    if E_sq < 0.0:
        E_sq = 0.0
    H = H0 * math.sqrt(E_sq)
    return c_s / (a_val * a_val * H)


@numba.njit
def _sound_horizon_drag(Ob0: float, Om0: float, Or0: float, Ok0: float, Ol0: float, H0: float) -> float:
    """Compute r_d with a high-resolution Simpson rule from a≈0 to a_drag."""
    z_drag = _z_drag_eh(Om0, Ob0, H0)
    a_drag = 1.0 / (1.0 + z_drag)
    Og0 = _omega_gamma0(H0)

    steps = 500_000  # Simpson requires even number of intervals
    if steps < 2 or steps % 2 == 1:
        return 0.0
    h = a_drag / steps

    total = _integrand_sound_horizon(0.0, Ob0, Og0, H0, Om0, Or0, Ok0, Ol0)
    for i in range(1, steps):
        a = i * h
        weight = 4.0 if i % 2 == 1 else 2.0
        total += weight * _integrand_sound_horizon(a, Ob0, Og0, H0, Om0, Or0, Ok0, Ol0)
    total += _integrand_sound_horizon(a_drag, Ob0, Og0, H0, Om0, Or0, Ok0, Ol0)

    return total * h / 3.0


def z_star_hu_sugiyama(Om0: float, Ob0: float, H0: float) -> float:
    """Hu & Sugiyama (1996) recombination redshift approximation."""
    h = H0 / 100.0
    Obh2 = Ob0 * h * h
    Omh2 = Om0 * h * h
    g1 = (0.0783 * Obh2 ** -0.238) / (1.0 + 39.5 * Obh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * Obh2 ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * Obh2 ** -0.738) * (1.0 + g1 * Omh2 ** g2)


def sound_horizon_to_z(z_target: float, Ob0: float, Om0: float, Or0: float, Ok0: float, Ol0: float, H0: float, steps: int = 500_000) -> float:
    """
    Sound horizon r_s integrated up to an explicit redshift (e.g., recombination z_star).
    """
    if z_target <= 0.0:
        return 0.0
    a_target = 1.0 / (1.0 + z_target)
    if steps < 2:
        steps = 2
    if steps % 2 == 1:
        steps += 1
    h = a_target / steps
    Og0 = _omega_gamma0(H0)
    total = _integrand_sound_horizon(0.0, Ob0, Og0, H0, Om0, Or0, Ok0, Ol0)
    for i in range(1, steps):
        a = i * h
        weight = 4.0 if i % 2 == 1 else 2.0
        total += weight * _integrand_sound_horizon(a, Ob0, Og0, H0, Om0, Or0, Ok0, Ol0)
    total += _integrand_sound_horizon(a_target, Ob0, Og0, H0, Om0, Or0, Ok0, Ol0)
    return total * h / 3.0


def comoving_distance_to_z(z_target: float, Om0: float, Or0: float, Ok0: float, Ol0: float, H0: float, steps: int = 4096) -> float:
    """
    LOS comoving distance χ(z) = (c/H0) ∫_0^{z} dz' / E(z') using direct Simpson integration.
    """
    if z_target <= 0.0:
        return 0.0
    z_target = float(z_target)
    steps = max(int(steps), 2)
    if steps % 2 == 1:
        steps += 1
    h = z_target / steps

    def E_inv(z: float) -> float:
        fac = 1.0 + z
        E_sq = Om0 * fac**3 + Or0 * fac**4 + Ok0 * fac**2 + Ol0
        if E_sq <= 0.0:
            return 0.0
        return 1.0 / math.sqrt(E_sq)

    total = E_inv(0.0) + E_inv(z_target)
    for i in range(1, steps):
        z = i * h
        total += (4.0 if i % 2 == 1 else 2.0) * E_inv(z)
    return (C_LIGHT / H0) * (h / 3.0) * total


def omega_gamma0_from_Tcmb(H0: float, T_cmb: float = T_CMB) -> float:
    h = H0 / 100.0
    Omega_gamma_h2 = 2.469e-5 * (T_cmb / 2.7255) ** 4
    return Omega_gamma_h2 / (h * h)


def omega_r0_from_Tcmb(H0: float, T_cmb: float = T_CMB, N_eff: float = 3.046) -> tuple[float, float]:
    """Return (Omega_r0_total, Omega_gamma0) using the usual (1 + 0.2271 N_eff) factor."""
    Og0 = omega_gamma0_from_Tcmb(H0, T_cmb=T_cmb)
    Or0 = Og0 * (1.0 + 0.2271 * N_eff)
    return Or0, Og0


def sound_horizon_high_z(z_start: float, z_max: float, Ob0: float, Om0: float, Or0: float, Og0: float, Ok0: float, H0: float, steps: int = 200_000) -> float:
    """
    Integrate r_s from z_start to z_max (default 1e5) mirroring the standalone script style.
    """
    if z_start < 0.0:
        z_start = 0.0
    if z_max <= z_start:
        return 0.0
    steps = max(int(steps), 2)
    if steps % 2 == 1:
        steps += 1
    z_grid = np.linspace(z_start, z_max, steps + 1)
    zp1 = 1.0 + z_grid
    E = np.sqrt(Om0 * zp1 ** 3 + Or0 * zp1 ** 4 + Ok0 * zp1 ** 2 + (1.0 - Om0 - Or0 - Ok0))
    H = H0 * E
    R_b = (3.0 * Ob0) / (4.0 * Og0) / zp1
    c_s = C_LIGHT / np.sqrt(3.0 * (1.0 + R_b))
    integrand = c_s / H
    h = (z_max - z_start) / steps
    total = integrand[0] + integrand[-1] + 4.0 * integrand[1:-1:2].sum() + 2.0 * integrand[2:-2:2].sum()
    return float((h / 3.0) * total)


@numba.njit
def kernel_lcdm_math(params: np.ndarray, a_grid: np.ndarray):
    """
    Compute LCDM background + growth quantities on the supplied scale-factor grid.

    params: [H0, Omega_m0, Omega_b0, Omega_k0, Omega_r0, Omega_Lambda]
    Returns:
        E_of_a      (1D)
        H_of_a      (1D)
        D_of_a      (1D)
        r_d         (scalar)
        sigma8      (scalar)
    """
    H0 = params[0]
    Om0 = params[1]
    Ob0 = params[2]
    Ok0 = params[3]
    Or0 = params[4]
    Ol0 = params[5]

    n = a_grid.shape[0]
    E_of_a = np.empty(n, dtype=np.float64)
    H_of_a = np.empty(n, dtype=np.float64)

    for i in range(n):
        a = a_grid[i]
        inv_a = 1.0 / a
        inv_a2 = inv_a * inv_a
        E_sq = Om0 * inv_a2 * inv_a + Or0 * inv_a2 * inv_a2 + Ok0 * inv_a2 + Ol0
        if E_sq < 0.0:
            E_sq = 0.0
        E_val = math.sqrt(E_sq)
        E_of_a[i] = E_val
        H_of_a[i] = H0 * E_val

    D_of_a, sigma8 = growth.solve_growth(a_grid, E_of_a, omega_m0=Om0)
    r_d = _sound_horizon_drag(Ob0, Om0, Or0, Ok0, Ol0, H0)

    return E_of_a, H_of_a, D_of_a, r_d, sigma8
