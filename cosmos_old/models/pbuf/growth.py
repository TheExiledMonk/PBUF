"""Linear growth helpers (skeleton)."""

from __future__ import annotations

import numpy as np

from cosmos.models.pbuf.distances import E
from cosmos.models.pbuf.params import PBUFParams
from cosmos.models.pbuf.thermal_table import ThermalTable


def growth_ode_rhs(a: float, y, params: PBUFParams, table: ThermalTable):
    """
    Right-hand side of the growth equation:
        y = [D(a), D'(a)]
    """

    y = np.asarray(y)
    D, D_prime = y

    eps = 1e-5
    E_a = E(a, params, table)
    E_a_plus = E(a + eps, params, table)
    E_a_minus = E(a - eps, params, table)
    dE_da = (E_a_plus - E_a_minus) / (2.0 * eps)

    term1 = -(3.0 / a + dE_da / E_a) * D_prime
    term2 = 1.5 * params.Omega_m0 / (a ** 5 * E_a ** 2) * D

    return np.array([D_prime, term1 + term2])
