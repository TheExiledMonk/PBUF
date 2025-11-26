"""Distance computations using E(a) arrays (Numba-safe)."""

import math

import numba
import numpy as np

C_LIGHT = 299_792.458  # km/s


@numba.njit
def _interp_scalar(x: float, grid_x: np.ndarray, grid_y: np.ndarray) -> float:
    n = grid_x.shape[0]
    if n == 0:
        return 0.0
    if x <= grid_x[0]:
        return grid_y[0]
    if x >= grid_x[n - 1]:
        return grid_y[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if grid_x[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0 = grid_x[lo]
    x1 = grid_x[hi]
    y0 = grid_y[lo]
    y1 = grid_y[hi]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@numba.njit
def comoving_distance(a_grid: np.ndarray, E_of_a: np.ndarray, H0: float, Omega_k0: float = 0.0) -> np.ndarray:
    """
    Line-of-sight comoving distance χ(a) in Mpc for a precomputed E(a) grid.

    Integrates χ(z) = (c/H0) ∫_0^{z(a)} dz / E(z) with trapezoidal rule over the
    implicit z-grid defined by ``a_grid`` (z = 1/a - 1). Returns transverse
    distance (curvature-corrected) if Ω_k0 ≠ 0.
    """
    n = a_grid.shape[0]
    chi = np.empty(n, dtype=np.float64)
    if n == 0:
        return chi

    chi[n - 1] = 0.0  # at a=1 -> z=0
    if n == 1:
        return chi

    z_grid = 1.0 / a_grid - 1.0

    # integrate backward from today to early times using z-spacing
    for i in range(n - 2, -1, -1):
        z0 = z_grid[i]
        z1 = z_grid[i + 1]
        dz = z0 - z1
        if dz < 0.0:
            dz = -dz
        inv0 = 1.0 / E_of_a[i]
        inv1 = 1.0 / E_of_a[i + 1]
        chi[i] = chi[i + 1] + 0.5 * (inv0 + inv1) * dz

    scale = C_LIGHT / H0
    for i in range(n):
        chi[i] *= scale

    if abs(Omega_k0) < 1.0e-12:
        return chi

    sqrt_abs = math.sqrt(abs(Omega_k0))
    prefactor = C_LIGHT / (H0 * sqrt_abs)
    arg_scale = sqrt_abs * H0 / C_LIGHT  # D_C is in Mpc, so arg is dimensionless.

    if Omega_k0 > 0.0:
        for i in range(n):
            chi[i] = prefactor * math.sinh(arg_scale * chi[i])
    else:
        for i in range(n):
            chi[i] = prefactor * math.sin(arg_scale * chi[i])
    return chi


def comoving_distance_simpson_z(a_grid: np.ndarray, E_of_a: np.ndarray, H0: float, steps: int = 300_000) -> np.ndarray:
    """
    High-accuracy comoving distance via Simpson integration over redshift.
    Uses a uniform z-grid from 0 to z_max implied by a_grid (z=1/a-1).
    """
    n = a_grid.shape[0]
    chi = np.zeros(n, dtype=float)
    if n == 0:
        return chi
    # z points corresponding to the provided a-grid (note: a_grid is ascending, z is descending)
    z_points = 1.0 / a_grid - 1.0
    z_max = float(np.max(z_points))
    steps = max(steps, int(math.ceil(z_max / 0.01)))
    steps = min(steps, 2_000_000)
    if steps < 2:
        steps = 2
    if steps % 2 == 1:
        steps += 1
    z_grid = np.linspace(0.0, z_max, steps + 1)
    a_for_z_dec = 1.0 / (1.0 + z_grid)  # decreasing with increasing z
    # Interpolate E(a) requires increasing x; flip a_for_z and flip back.
    a_inc = np.ascontiguousarray(a_grid)
    E_inc = np.ascontiguousarray(E_of_a)
    E_z = np.interp(a_for_z_dec[::-1], a_inc, E_inc)[::-1]
    f = 1.0 / E_z
    h = z_grid[1] - z_grid[0]
    total = f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-2:2].sum()
    _ = (C_LIGHT / H0) * (h / 3.0) * total  # total integral not used directly
    # Build cumulative trapezoid for mapping to arbitrary z
    cum = np.zeros_like(z_grid)
    cum[1:] = (C_LIGHT / H0) * np.cumsum((f[:-1] + f[1:]) * 0.5 * h)
    # Interpolate cumulative integral at desired z points (make z increasing)
    z_pts_inc = z_points[::-1]
    chi_inc = np.interp(z_pts_inc, z_grid, cum)
    chi = chi_inc[::-1]
    return chi


def comoving_distance_simpson_to_z(z_star: float, a_grid: np.ndarray, E_of_a: np.ndarray, H0: float, *, max_steps: int = 1_000_000) -> float:
    """
    Scalar χ(z_star) using a dense Simpson rule in redshift with on-the-fly E(a) interpolation.

    Keeps the global grid untouched but provides a high-resolution LOS integral for CMB-only use.
    """
    if z_star <= 0.0:
        return 0.0
    z_star = float(z_star)
    # Target step ~1e-3 in z with a generous upper cap.
    steps = max(int(math.ceil(z_star / 1.0e-3)), 2)
    steps = min(steps, max_steps)
    if steps % 2 == 1:
        steps += 1
    z_grid = np.linspace(0.0, z_star, steps + 1)
    a_for_z = 1.0 / (1.0 + z_grid)
    E_z = np.interp(a_for_z, a_grid, E_of_a)
    f = 1.0 / E_z
    h = z_grid[1] - z_grid[0]
    total = f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-2:2].sum()
    chi = (C_LIGHT / H0) * (h / 3.0) * total
    return float(chi)


@numba.njit
def angular_diameter_distance(a_grid: np.ndarray, transverse_distance: np.ndarray) -> np.ndarray:
    """Angular diameter distance D_A = a * D_M for each scale factor in the grid."""
    n = a_grid.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = a_grid[i] * transverse_distance[i]
    return out


@numba.njit
def luminosity_distance(a_grid: np.ndarray, transverse_distance: np.ndarray) -> np.ndarray:
    """Luminosity distance D_L = D_M / a = (1+z) D_M for each scale factor."""
    n = a_grid.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = transverse_distance[i] / a_grid[i]
    return out


@numba.njit
def transverse_comoving_distance(chi: float, H0: float, Omega_k0: float) -> float:
    """
    Apply curvature to a scalar comoving distance.
    D_M = chi for flat; otherwise c/(H0 sqrt|Ωk|) * sinn(sqrt|Ωk| H0 chi / c).
    """
    if chi <= 0.0 or abs(Omega_k0) < 1.0e-12:
        return chi
    sqrt_abs = math.sqrt(abs(Omega_k0))
    prefactor = C_LIGHT / (H0 * sqrt_abs)
    arg = sqrt_abs * H0 * chi / C_LIGHT
    if Omega_k0 > 0.0:
        return prefactor * math.sinh(arg)
    return prefactor * math.sin(arg)


@numba.njit
def transverse_distance_grid(a_grid: np.ndarray, E_of_a: np.ndarray, H0: float, Omega_k0: float) -> np.ndarray:
    """Build D_M(a) applying curvature to the integrated comoving distance."""
    chi = comoving_distance(a_grid, E_of_a, H0, 0.0)
    n = chi.shape[0]
    DM = np.empty(n, dtype=np.float64)
    for i in range(n):
        DM[i] = transverse_comoving_distance(chi[i], H0, Omega_k0)
    return DM


@numba.njit
def distance_modulus_from_dm(z: float, a_grid: np.ndarray, D_M: np.ndarray) -> float:
    """Distance modulus given a precomputed D_M grid."""
    a = 1.0 / (1.0 + z)
    DM = _interp_scalar(a, a_grid, D_M)
    DL = DM * (1.0 + z)
    if DL <= 0.0:
        return 1.0e30
    return 5.0 * (math.log10(DL) + 5.0)


@numba.njit
def dv_over_rd(z: float, a_grid: np.ndarray, D_M: np.ndarray, H_of_a: np.ndarray, r_d: float) -> float:
    """Compute DV/rd at redshift z using precomputed grids."""
    a = 1.0 / (1.0 + z)
    DM = _interp_scalar(a, a_grid, D_M)
    H = _interp_scalar(a, a_grid, H_of_a)
    D_A = DM / (1.0 + z)
    dv = (D_A * D_A * (1.0 + z) * (1.0 + z) * C_LIGHT * z / H) ** (1.0 / 3.0)
    return dv / r_d
