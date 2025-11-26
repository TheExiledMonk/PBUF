"""Pytest coverage for the Early Dark Energy LCDM helper suite."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.models.ede_lcdm import expansion, parameters, phase6a


@pytest.fixture
def default_params() -> parameters.EDELCDMParams:
    return parameters.EDELCDMParams(**parameters.get_default_parameters())


def test_omega_ede_formula_matches_manual(default_params: parameters.EDELCDMParams) -> None:
    for a in (1.0, 0.1, 0.001):
        manual = default_params.Omega_EDE_0 / (1.0 + (a / default_params.a_c) ** default_params.n)
        assert math.isclose(expansion.omega_ede(a, default_params), manual, rel_tol=1e-12)


def test_E_reduces_to_lcdm_when_omega_ede_zero(default_params: parameters.EDELCDMParams) -> None:
    lcdm_params = parameters.EDELCDMParams(**{
        **parameters.get_default_parameters(),
        "Omega_EDE_0": 0.0,
    })
    lambda_term = expansion.omega_lambda(lcdm_params)
    for a in (1.0, 0.2, 0.5):
        expected_E2 = (
            lcdm_params.Omega_m0 / a**3
            + lcdm_params.Omega_r0 / a**4
            + lcdm_params.Omega_k0 / a**2
            + lambda_term
        )
        assert math.isclose(expansion.E_squared(a, lcdm_params), expected_E2, rel_tol=1e-12)
        assert math.isclose(expansion.E(a, lcdm_params), math.sqrt(expected_E2), rel_tol=1e-12)


def test_phase6a_accepts_default_parameters() -> None:
    ok, reason = phase6a.phase6a_ede(parameters.get_default_parameters())
    assert ok
    assert reason is None


def test_phase6a_detects_closure_violation() -> None:
    broken = {**parameters.get_default_parameters(), "Omega_m0": 0.8, "Omega_k0": 0.2}
    ok, reason = phase6a.phase6a_ede(broken)
    assert not ok
    assert reason is not None and "closure" in reason.lower()


def test_zero_redshift_consistency(default_params: parameters.EDELCDMParams) -> None:
    h_at_one = expansion.H(1.0, default_params)
    assert math.isclose(h_at_one, default_params.H0, rel_tol=1e-12)
    omega_total = (
        default_params.Omega_m0
        + default_params.Omega_r0
        + default_params.Omega_k0
        + expansion.omega_ede(1.0, default_params)
        + expansion.omega_lambda(default_params)
    )
    assert math.isclose(omega_total, 1.0, rel_tol=1e-8)


def test_hubble_is_monotonic(default_params: parameters.EDELCDMParams) -> None:
    a_grid = np.logspace(-5, 0, 200)
    H_vals = np.array([expansion.H(a, default_params) for a in a_grid])
    diffs = np.diff(H_vals)
    assert np.all(diffs < 0.0)


def test_energy_densities_stay_non_negative(default_params: parameters.EDELCDMParams) -> None:
    a_grid = np.logspace(-5, 0, 200)
    ede_vals = np.array([expansion.omega_ede(a, default_params) for a in a_grid])
    assert np.all(ede_vals >= 0.0)
    assert expansion.omega_lambda(default_params) >= 0.0
