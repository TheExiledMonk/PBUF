"""Growth ODE helpers for the running-Λ model."""

from __future__ import annotations

import numpy as np

from cosmos.models.running_lambda import expansion
from cosmos.models.running_lambda.parameters import RunningLambdaParams


def growth_ode_rhs(a: float, y, params: RunningLambdaParams):
    y = np.asarray(y, dtype=float)
    D, D_prime = y

    eps = 1e-5
    E_a = expansion.E(a, params)
    a_minus = max(a - eps, 1e-8)
    E_a_plus = expansion.E(a + eps, params)
    E_a_minus = expansion.E(a_minus, params)
    dE_da = (E_a_plus - E_a_minus) / (2.0 * eps)

    term1 = -(3.0 / a + dE_da / E_a) * D_prime
    matter_exponent = -3.0 * (1.0 - params.nu_lambda)
    matter_term = params.Omega_m0 * a ** matter_exponent
    term2 = 1.5 * matter_term / (a**2 * E_a**2) * D

    return np.array([D_prime, term1 + term2])
