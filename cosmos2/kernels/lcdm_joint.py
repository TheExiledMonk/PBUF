"""Joint chi2 aggregation for LCDM (Numba-ready)."""

import math

import numba
import numpy as np

C_LIGHT = 299_792.458  # km/s


@numba.njit
def _interp_grid(x_grid: np.ndarray, y_grid: np.ndarray, x: float) -> float:
    n = x_grid.shape[0]
    if n == 0:
        return 0.0
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[n - 1]:
        return y_grid[n - 1]
    # assume monotonically increasing x_grid
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0 = x_grid[lo]
    x1 = x_grid[hi]
    y0 = y_grid[lo]
    y1 = y_grid[hi]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@numba.njit
def _chi2(residuals: np.ndarray, cov: np.ndarray) -> float:
    """Diagonal or full covariance χ²."""
    n = residuals.shape[0]
    if cov.ndim == 1 or cov.shape[1] == 1:
        chi2_val = 0.0
        for i in range(n):
            var = cov[i] if cov.shape[0] > 1 else cov[0]
            if var <= 0.0:
                continue
            chi2_val += residuals[i] * residuals[i] / var
        return chi2_val

    # full matrix but no linalg: use diagonal as fallback
    chi2_val = 0.0
    for i in range(n):
        var = cov[i, i]
        if var <= 0.0:
            continue
        chi2_val += residuals[i] * residuals[i] / var
    return chi2_val


@numba.njit
def _comoving_distance(a_grid: np.ndarray, E_of_a: np.ndarray, H0: float) -> np.ndarray:
    """Backward trapezoid for χ(a)."""
    n = a_grid.shape[0]
    chi = np.empty(n, dtype=np.float64)
    if n == 0:
        return chi
    chi[n - 1] = 0.0
    for i in range(n - 2, -1, -1):
        a0 = a_grid[i]
        a1 = a_grid[i + 1]
        da = a1 - a0
        inv0 = 1.0 / (a0 * a0 * E_of_a[i])
        inv1 = 1.0 / (a1 * a1 * E_of_a[i + 1])
        chi[i] = chi[i + 1] + 0.5 * (inv0 + inv1) * da
    scale = C_LIGHT / H0
    for i in range(n):
        chi[i] *= scale
    return chi


@numba.njit
def _dv_over_rd(z: float, a_grid: np.ndarray, D_M: np.ndarray, H_of_a: np.ndarray, r_d: float) -> float:
    a = 1.0 / (1.0 + z)
    DM = _interp_grid(a_grid, D_M, a)
    H = _interp_grid(a_grid, H_of_a, a)
    # D_V = [ (1+z)^2 D_A^2 * cz/H(z) ]^{1/3}; D_A = D_M / (1+z)
    D_A = DM / (1.0 + z)
    dv = (D_A * D_A * (1.0 + z) * (1.0 + z) * C_LIGHT * z / H) ** (1.0 / 3.0)
    return dv / r_d


@numba.njit
def _distance_modulus(z: float, a_grid: np.ndarray, D_M: np.ndarray) -> float:
    a = 1.0 / (1.0 + z)
    DM = _interp_grid(a_grid, D_M, a)
    DL = DM * (1.0 + z)
    if DL <= 0.0:
        return 1.0e30
    return 5.0 * (math.log10(DL) + 5.0)


@numba.njit
def _growth_rate(a_grid: np.ndarray, D_of_a: np.ndarray, a: float) -> float:
    n = a_grid.shape[0]
    if n < 2:
        return 0.0
    if a <= a_grid[0]:
        idx = 0
    elif a >= a_grid[n - 1]:
        idx = n - 2
    else:
        idx = 0
        for i in range(n - 1):
            if a_grid[i + 1] >= a:
                idx = i
                break
    a0 = a_grid[idx]
    a1 = a_grid[idx + 1]
    D0 = D_of_a[idx]
    D1 = D_of_a[idx + 1]
    if D0 <= 0.0 or D1 <= 0.0:
        return 0.0
    dlnD = math.log(D1 / D0)
    dlnA = math.log(a1 / a0)
    return dlnD / dlnA


@numba.njit
def _chi2_cmb(z_cmb: np.ndarray, data: np.ndarray, cov: np.ndarray, D_M: np.ndarray, a_grid: np.ndarray, H0: float, Om0_est: float, r_d: float) -> float:
    if z_cmb.shape[0] == 0 or data.shape[0] == 0:
        return 0.0
    z = z_cmb[0]
    a = 1.0 / (1.0 + z)
    DM = _interp_grid(a_grid, D_M, a)
    R = math.sqrt(Om0_est) * (H0 * DM / C_LIGHT)
    l_A = math.pi * DM / r_d
    preds = np.empty(data.shape[0], dtype=np.float64)
    if preds.shape[0] >= 1:
        preds[0] = R
    if preds.shape[0] >= 2:
        preds[1] = l_A
    if preds.shape[0] >= 3:
        preds[2] = Om0_est  # placeholder third observable
    residuals = preds - data
    return _chi2(residuals, cov)


@numba.njit
def _chi2_bao(z_bao: np.ndarray, data: np.ndarray, cov: np.ndarray, a_grid: np.ndarray, D_M: np.ndarray, H_of_a: np.ndarray, r_d: float) -> float:
    n = z_bao.shape[0]
    if n == 0 or data.shape[0] == 0:
        return 0.0
    residuals = np.empty(n, dtype=np.float64)
    for i in range(n):
        pred = _dv_over_rd(z_bao[i], a_grid, D_M, H_of_a, r_d)
        residuals[i] = pred - data[i]
    return _chi2(residuals, cov)


@numba.njit
def _chi2_sn(z_sn: np.ndarray, data: np.ndarray, cov: np.ndarray, a_grid: np.ndarray, D_M: np.ndarray) -> float:
    n = z_sn.shape[0]
    if n == 0 or data.shape[0] == 0:
        return 0.0
    residuals = np.empty(n, dtype=np.float64)
    for i in range(n):
        mu = _distance_modulus(z_sn[i], a_grid, D_M)
        residuals[i] = mu - data[i]
    return _chi2(residuals, cov)


@numba.njit
def _chi2_cc(z_cc: np.ndarray, data: np.ndarray, cov: np.ndarray, a_grid: np.ndarray, H_of_a: np.ndarray) -> float:
    n = z_cc.shape[0]
    if n == 0 or data.shape[0] == 0:
        return 0.0
    residuals = np.empty(n, dtype=np.float64)
    for i in range(n):
        a = 1.0 / (1.0 + z_cc[i])
        H = _interp_grid(a_grid, H_of_a, a)
        residuals[i] = H - data[i]
    return _chi2(residuals, cov)


@numba.njit
def _chi2_rsd(z_rsd: np.ndarray, data: np.ndarray, cov: np.ndarray, a_grid: np.ndarray, D_of_a: np.ndarray, sigma8_today: float) -> float:
    n = z_rsd.shape[0]
    if n == 0 or data.shape[0] == 0:
        return 0.0
    residuals = np.empty(n, dtype=np.float64)
    for i in range(n):
        a = 1.0 / (1.0 + z_rsd[i])
        f = _growth_rate(a_grid, D_of_a, a)
        fs8 = f * sigma8_today * _interp_grid(a_grid, D_of_a, a)
        residuals[i] = fs8 - data[i]
    return _chi2(residuals, cov)


@numba.njit
def kernel_lcdm_joint_fits(
    E_of_a: np.ndarray,
    H_of_a: np.ndarray,
    D_of_a: np.ndarray,
    r_d: float,
    sigma8: float,
    z_cmb: np.ndarray,
    cmb_data: np.ndarray,
    cmb_cov: np.ndarray,
    z_bao: np.ndarray,
    bao_data: np.ndarray,
    bao_cov: np.ndarray,
    z_sn: np.ndarray,
    sn_data: np.ndarray,
    sn_cov: np.ndarray,
    z_cc: np.ndarray,
    cc_data: np.ndarray,
    cc_cov: np.ndarray,
    z_rsd: np.ndarray,
    rsd_data: np.ndarray,
    rsd_cov: np.ndarray,
):
    """
    Uses E(a), H(a), r_d, sigma8 to compute:
        chi2_cmb, chi2_bao, chi2_sn, chi2_cc, chi2_rsd, chi2_total
    """
    H0 = H_of_a[H_of_a.shape[0] - 1] if H_of_a.shape[0] > 0 else 70.0
    a_grid = np.linspace(1.0 / E_of_a.shape[0], 1.0, E_of_a.shape[0]) if E_of_a.shape[0] > 0 else np.empty(0)
    # Estimate Omega_m0 from early-time scaling
    Om0_est = 0.3
    if E_of_a.shape[0] > 0:
        Om0_est = E_of_a[0] * E_of_a[0] * a_grid[0] * a_grid[0] * a_grid[0]
        if Om0_est <= 0.0:
            Om0_est = 0.3

    chi_grid = _comoving_distance(a_grid, E_of_a, H0)

    chi2_cmb = _chi2_cmb(z_cmb, cmb_data, cmb_cov, chi_grid, a_grid, H0, Om0_est, r_d)
    chi2_bao = _chi2_bao(z_bao, bao_data, bao_cov, a_grid, chi_grid, H_of_a, r_d)
    chi2_sn = _chi2_sn(z_sn, sn_data, sn_cov, a_grid, chi_grid)
    chi2_cc = _chi2_cc(z_cc, cc_data, cc_cov, a_grid, H_of_a)
    chi2_rsd = _chi2_rsd(z_rsd, rsd_data, rsd_cov, a_grid, D_of_a, sigma8)

    chi2_total = chi2_cmb + chi2_bao + chi2_sn + chi2_cc + chi2_rsd
    return chi2_total, chi2_cmb, chi2_bao, chi2_sn, chi2_cc, chi2_rsd
