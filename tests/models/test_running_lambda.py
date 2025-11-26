"""Pytest coverage for the running-Λ helper suite."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.models.lcdm import distances as lcdm_distances
from cosmos.models.lcdm.params import LCDMParams
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.running_lambda import expansion, parameters, sanity
from cosmos.models.running_lambda.model import RunningLambdaModel
from cosmos.models.running_lambda.parameters import RunningLambdaParams


@pytest.fixture
def default_parameters() -> Dict[str, float]:
    return parameters.get_default_parameters()


def _lcdm_params_from_running(params: RunningLambdaParams) -> LCDMParams:
    lcdm_params = LCDMParams(
        H0=params.H0,
        Omega_m0=params.Omega_m0,
        Omega_r0=params.Omega_r0,
        Omega_k0=params.Omega_k0,
        Omega_b0=params.Omega_b0,
    )
    return lcdm_params.with_lambda(params.Omega_lambda)


def test_E_squared_matches_lcdm_limit(default_parameters: Dict[str, float]) -> None:
    rl_params = RunningLambdaParams(**default_parameters)
    lcdm_params = _lcdm_params_from_running(rl_params)

    for a in (1.0, 0.2, 0.5, 0.9):
        running_E2 = expansion.E_squared(a, rl_params)
        lcdm_E2 = lcdm_distances.E_squared(a, lcdm_params)
        assert math.isclose(running_E2, lcdm_E2, rel_tol=1e-10)
        assert math.isclose(expansion.E(a, rl_params), math.sqrt(lcdm_E2), rel_tol=1e-10)


def test_small_nu_behaviour(default_parameters: Dict[str, float]) -> None:
    for nu in (0.05, -0.05):
        params = RunningLambdaParams(**{**default_parameters, "nu_lambda": nu})
        a_grid = np.logspace(-4, 0, 200)
        H_vals = np.array([expansion.H(a, params) for a in a_grid])
        assert np.min(np.diff(H_vals)) < 1e-8
        assert np.max(np.diff(H_vals)) < 1e-6
        E2_vals = expansion.E_squared(a_grid, params)
        assert np.all(E2_vals > 0.0)
        closure = params.Omega_m0 + params.Omega_r0 + params.Omega_k0 + params.Omega_lambda
        assert math.isclose(closure, 1.0, rel_tol=1e-9)


def test_growth_matches_lcdm_limit(default_parameters: Dict[str, float]) -> None:
    rl_params = dict(default_parameters)
    running_model = RunningLambdaModel(**rl_params)
    lcdm_model = LCDMModel(
        H0=default_parameters["H0"],
        Omega_m0=default_parameters["Omega_m0"],
        Omega_b0=default_parameters["Omega_b0"],
        Omega_r0=default_parameters["Omega_r0"],
        Omega_k0=default_parameters["Omega_k0"],
    )

    for z in (0.0, 0.5, 1.0, 2.0):
        rl_growth = running_model.growth_factor(z)
        lcdm_growth = lcdm_model.growth_factor(z)
        assert math.isclose(rl_growth, lcdm_growth, rel_tol=1e-8)
        rl_fs8 = running_model.fs8(z)
        lcdm_fs8 = lcdm_model.fs8(z)
        assert math.isclose(rl_fs8, lcdm_fs8, rel_tol=1e-8)

    assert math.isclose(running_model.sigma8(), lcdm_model.sigma8(), rel_tol=1e-9)


def test_growth_behaviour_small_nu(default_parameters: Dict[str, float]) -> None:
    params = dict(default_parameters, nu_lambda=0.05)
    model = RunningLambdaModel(**params)
    a_grid = np.linspace(0.2, 1.0, 40)
    z_grid = [1.0 / a - 1.0 for a in a_grid]
    growth_vals = np.array([model.growth_factor(z) for z in z_grid])
    assert np.all(growth_vals > 0.0)
    assert np.all(np.diff(growth_vals) >= -1e-8)
    assert math.isclose(growth_vals[-1], 1.0, rel_tol=1e-10)


def test_sanity_rejects_large_nu(default_parameters: Dict[str, float]) -> None:
    params = dict(default_parameters, nu_lambda=0.5)
    result = sanity.sanity_checks(params)
    assert not result.ok
    assert any("nu" in reason.lower() for reason in result.reasons)
