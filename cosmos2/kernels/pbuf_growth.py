"""Numba kernels for PBUF linear growth ODE."""

from __future__ import annotations

import numba

from .pbuf_distances import E_kernel


@numba.njit(cache=True)
def growth_rhs_kernel(
    a_val: float,
    D: float,
    D_prime: float,
    Omega_m0: float,
    Omega_r0: float,
    alpha: float,
    omega_sigma: float,
    eps: float = 1e-5,
) -> tuple[float, float]:
    """
    Compute RHS [D', D''] of the growth system using a finite-difference dE/da.
    """

    E_a = E_kernel(a_val, Omega_m0, Omega_r0, alpha, omega_sigma)
    E_a_plus = E_kernel(a_val + eps, Omega_m0, Omega_r0, alpha, omega_sigma)
    E_a_minus = E_kernel(a_val - eps, Omega_m0, Omega_r0, alpha, omega_sigma)
    dE_da = (E_a_plus - E_a_minus) / (2.0 * eps)

    term1 = -(3.0 / a_val + dE_da / E_a) * D_prime
    term2 = 1.5 * Omega_m0 / (a_val ** 5 * E_a ** 2) * D
    return D_prime, term1 + term2


__all__ = ["growth_rhs_kernel"]
