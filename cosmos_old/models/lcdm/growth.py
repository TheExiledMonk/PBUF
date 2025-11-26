"""LCDM growth helper skeleton."""

from __future__ import annotations

import numpy as np

from cosmos.models.lcdm.distances import E
from cosmos.models.lcdm.params import LCDMParams


def growth_ode_rhs(a: float, y, params: LCDMParams):
    y = np.asarray(y)
    D, D_prime = y

    eps = 1e-5
    E_a = E(a, params)
    E_a_plus = E(a + eps, params)
    E_a_minus = E(a - eps, params)
    dE_da = (E_a_plus - E_a_minus) / (2.0 * eps)

    term1 = -(3.0 / a + dE_da / E_a) * D_prime
    term2 = 1.5 * params.Omega_m0 / (a ** 5 * E_a ** 2) * D

    return np.array([D_prime, term1 + term2])
