"""Pytest coverage for the DESI modified ΛCDM helpers."""

from __future__ import annotations

import math
from typing import Callable

import pytest

from cosmos.models.common.distance_utils import luminosity_distance as common_luminosity_distance
from cosmos.models.desi_mod import distances as desi_distances
from cosmos.models.desi_mod import expansion as desi_expansion
from cosmos.models.desi_mod import parameters as desi_parameters
from cosmos.models.desi_mod import sanity as desi_sanity
from cosmos.models.lcdm import distances as lcdm_distances
from cosmos.models.lcdm.params import LCDMParams


Integrator = Callable[[Callable[[float], float], float, float], float]


def _simple_integrator(func: Callable[[float], float], lower: float, upper: float, *, steps: int = 2048) -> float:
    if lower == upper:
        return 0.0
    dx = (upper - lower) / steps
    total = 0.5 * (func(lower) + func(upper))
    for i in range(1, steps):
        total += func(lower + dx * i)
    return total * dx


@pytest.fixture
def default_params() -> desi_parameters.DESIModParams:
    return desi_parameters.DESIModParams(**desi_parameters.get_default_parameters())


def test_omega_de_shape(default_params: desi_parameters.DESIModParams) -> None:
    values = []
    for a in (1.0, 0.5, 0.1):
        values.append(desi_expansion.omega_de(a, default_params))
    omega_de0 = max(default_params.Omega_DE0, 0.0)
    assert math.isclose(values[0], omega_de0, rel_tol=1e-9)
    assert values[1] == values[0]
    assert values[2] == values[0]


def test_E_squared_matches_manual(default_params: desi_parameters.DESIModParams) -> None:
    for a in (1.0, 0.2, 0.7):
        E2 = desi_expansion.E_squared(a, default_params)
        ode = desi_expansion.omega_de(a, default_params)
        expected = (
            default_params.Omega_m0 / a**3
            + default_params.Omega_r0 / a**4
            + default_params.Omega_k0 / a**2
            + ode
        )
        assert math.isclose(E2, expected, rel_tol=1e-12)
        assert math.isclose(desi_expansion.E(a, default_params), math.sqrt(E2), rel_tol=1e-12)


def test_dE_da_is_finite(default_params: desi_parameters.DESIModParams) -> None:
    for a in (0.01, 0.1, 0.5, 1.0):
        derivative = desi_expansion.dE_da(a, default_params)
        assert math.isfinite(derivative)


def test_H_positive(default_params: desi_parameters.DESIModParams) -> None:
    for a in (1e-4, 0.01, 0.1, 0.5, 1.0):
        assert desi_expansion.H(a, default_params) > 0.0


def test_distances_match_lcdm(default_params: desi_parameters.DESIModParams) -> None:
    lcdm_params = LCDMParams(
        H0=default_params.H0,
        Omega_m0=default_params.Omega_m0,
        Omega_r0=default_params.Omega_r0,
        Omega_k0=default_params.Omega_k0,
        Omega_b0=default_params.Omega_b0,
    )
    lcdm_params = lcdm_params.with_lambda(1.0 - (lcdm_params.Omega_m0 + lcdm_params.Omega_r0 + lcdm_params.Omega_k0))

    integrator: Integrator = lambda f, l, u: _simple_integrator(f, l, u, steps=2048)
    for z in (0.1, 0.5, 1.0):
        desi_chi = desi_distances.comoving_distance(z, default_params, integrator)
        lcdm_chi = lcdm_distances.comoving_distance(z, lcdm_params, integrator)
        assert math.isclose(desi_chi, lcdm_chi, rel_tol=2e-5)

        desi_da = desi_distances.angular_diameter_distance(z, default_params, integrator)
        lcdm_da = lcdm_distances.angular_diameter_distance(z, lcdm_params, integrator)
        assert math.isclose(desi_da, lcdm_da, rel_tol=2e-5)

        desi_dl = desi_distances.luminosity_distance(z, default_params, integrator)
        lcdm_dl = common_luminosity_distance(lcdm_chi, z)
        assert math.isclose(desi_dl, lcdm_dl, rel_tol=2e-5)


def test_closure_matches(default_params: desi_parameters.DESIModParams) -> None:
    closure = (
        default_params.Omega_m0
        + default_params.Omega_r0
        + default_params.Omega_k0
        + default_params.Omega_DE0
    )
    assert math.isclose(closure, 1.0, rel_tol=1e-9)


def test_sanity_checks_detect_closure_violation() -> None:
    params = desi_parameters.get_default_parameters()
    params["Omega_m0"] = 0.8
    params["Omega_k0"] = 0.3
    result = desi_sanity.sanity_checks(params)
    assert not result.ok
    assert any("closure" in reason for reason in result.reasons)


def test_sanity_checks_pass_default() -> None:
    result = desi_sanity.sanity_checks(desi_parameters.get_default_parameters())
    assert result.ok
