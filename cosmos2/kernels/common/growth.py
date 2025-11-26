"""Growth factor solver (Numba-safe)."""

import math

import numba
import numpy as np

_SIGMA8_TODAY = 1.0  # default normalisation; override at caller if needed


@numba.njit
def _dE_da(a_grid: np.ndarray, E_of_a: np.ndarray, out: np.ndarray) -> None:
    """Central differences for dE/da on a fixed grid."""
    n = a_grid.shape[0]
    if n == 0:
        return
    if n == 1:
        out[0] = 0.0
        return

    out[0] = (E_of_a[1] - E_of_a[0]) / (a_grid[1] - a_grid[0])
    for i in range(1, n - 1):
        da = a_grid[i + 1] - a_grid[i - 1]
        out[i] = (E_of_a[i + 1] - E_of_a[i - 1]) / da
    out[n - 1] = (E_of_a[n - 1] - E_of_a[n - 2]) / (a_grid[n - 1] - a_grid[n - 2])


@numba.njit
def solve_growth(a_grid: np.ndarray, E_of_a: np.ndarray, sigma8_today: float = _SIGMA8_TODAY, omega_m0: float = -1.0) -> tuple[np.ndarray, float]:
    """
    Solve linear growth D(a) with a fixed-step RK4 integrator.

    Uses supplied Omega_m0 when provided; otherwise infers from early-time scaling.
    Returns (D_of_a normalized to D(1)=1, sigma8).
    """
    n = a_grid.shape[0]
    D = np.empty(n, dtype=np.float64)
    G = np.empty(n, dtype=np.float64)
    if n == 0:
        return D, 0.0

    # Estimate Omega_m0 from earliest scale factor assuming matter dominance.
    Omega_m0_est = omega_m0
    if Omega_m0_est <= 0.0:
        Omega_m0_est = E_of_a[0] * E_of_a[0] * a_grid[0] * a_grid[0] * a_grid[0]
        if Omega_m0_est <= 0.0:
            Omega_m0_est = 1.0e-6

    dE = np.empty(n, dtype=np.float64)
    _dE_da(a_grid, E_of_a, dE)

    # Initial conditions: D ~ a in matter domination, D' = 1.
    D[0] = a_grid[0]
    G[0] = 1.0

    for i in range(n - 1):
        a0 = a_grid[i]
        a1 = a_grid[i + 1]
        h = a1 - a0

        E0 = E_of_a[i]
        E1 = E_of_a[i + 1]
        d0 = dE[i]
        d1 = dE[i + 1]

        D0 = D[i]
        G0 = G[i]

        mass0 = 1.5 * Omega_m0_est / (a0 * a0 * a0 * a0 * a0 * E0 * E0)
        fric0 = (3.0 / a0) + (d0 / max(E0, 1e-12))

        k1_D = G0
        k1_G = mass0 * D0 - fric0 * G0

        a_mid = a0 + 0.5 * h
        E_mid = 0.5 * (E0 + E1)
        d_mid = 0.5 * (d0 + d1)

        D_mid = D0 + 0.5 * h * k1_D
        G_mid = G0 + 0.5 * h * k1_G

        mass_mid = 1.5 * Omega_m0_est / (a_mid * a_mid * a_mid * a_mid * a_mid * E_mid * E_mid)
        fric_mid = (3.0 / a_mid) + (d_mid / max(E_mid, 1e-12))

        k2_D = G_mid
        k2_G = mass_mid * D_mid - fric_mid * G_mid

        D_mid2 = D0 + 0.5 * h * k2_D
        G_mid2 = G0 + 0.5 * h * k2_G

        k3_D = G_mid2
        k3_G = mass_mid * D_mid2 - fric_mid * G_mid2

        D_end = D0 + h * k3_D
        G_end = G0 + h * k3_G

        mass_end = 1.5 * Omega_m0_est / (a1 * a1 * a1 * a1 * a1 * E1 * E1)
        fric_end = (3.0 / a1) + (d1 / max(E1, 1e-12))

        k4_D = G_end
        k4_G = mass_end * D_end - fric_end * G_end

        inv6 = h / 6.0
        D[i + 1] = D0 + inv6 * (k1_D + 2.0 * k2_D + 2.0 * k3_D + k4_D)
        G[i + 1] = G0 + inv6 * (k1_G + 2.0 * k2_G + 2.0 * k3_G + k4_G)

    norm = D[n - 1]
    if norm != 0.0:
        inv_norm = 1.0 / norm
        for i in range(n):
            D[i] *= inv_norm
            G[i] *= inv_norm

    sigma8 = D[n - 1]
    return D, sigma8
