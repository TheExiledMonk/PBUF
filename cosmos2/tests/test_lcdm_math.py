import numpy as np
import pytest

numba = pytest.importorskip("numba")

from cosmos2.kernels import lcdm_math
from cosmos2.kernels.common import distances as dist2


def _expected_E(a_grid: np.ndarray, params: dict[str, float]) -> np.ndarray:
    Om = params["Omega_m0"]
    Ob = params["Omega_b0"]
    Ok = params["Omega_k0"]
    Or = params["Omega_r0"]
    Ol = params["Omega_lambda0"]
    Om_tot = Om  # Omega_m0 already includes baryons
    inv_a = 1.0 / a_grid
    inv_a2 = inv_a * inv_a
    return np.sqrt(Om_tot * inv_a2 * inv_a + Or * inv_a2 * inv_a2 + Ok * inv_a2 + Ol)


def _integrate_comoving(a_grid: np.ndarray, E_grid: np.ndarray, H0: float) -> np.ndarray:
    integrand = 1.0 / (a_grid * a_grid * E_grid)
    # integrate from today backwards using trapezoidal rule
    acc = np.zeros_like(a_grid)
    for i in range(a_grid.shape[0] - 2, -1, -1):
        da = a_grid[i + 1] - a_grid[i]
        acc[i] = acc[i + 1] + 0.5 * da * (integrand[i] + integrand[i + 1])
    return acc * (dist2.C_LIGHT / H0)


def test_lcdm_kernel_matches_analytic_background():
    params = {
        "H0": 67.4,
        "Omega_m0": 0.315,
        "Omega_r0": 9.0e-5,
        "Omega_k0": 0.0,
        "Omega_b0": 0.049,
        "Omega_lambda0": 1.0 - 0.315 - 9.0e-5,
        "sigma8_0": 0.811,
    }
    params_arr = np.array(
        [params["H0"], params["Omega_m0"], params["Omega_b0"], params["Omega_k0"], params["Omega_r0"], params["Omega_lambda0"]],
        dtype=np.float64,
    )
    a_grid = np.linspace(1.0e-3, 1.0, 2048, dtype=np.float64)
    E_of_a, H_of_a, D_of_a, r_d, sigma8 = lcdm_math.kernel_lcdm_math(params_arr, a_grid)

    E_expected = _expected_E(a_grid, params)
    np.testing.assert_allclose(E_of_a, E_expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(H_of_a, params["H0"] * E_expected, rtol=1e-4, atol=1e-6)

    chi_kernel = dist2.comoving_distance(a_grid, E_of_a, params["H0"], params["Omega_k0"])
    chi_expected = _integrate_comoving(a_grid, E_expected, params["H0"])
    np.testing.assert_allclose(chi_kernel, chi_expected, rtol=2e-3, atol=1e-3)

    assert np.isclose(D_of_a[-1], 1.0, rtol=1e-5)
    assert np.isfinite(r_d) and r_d > 0.0
    assert np.isclose(sigma8, 1.0, rtol=1e-3)
