"""Pytest coverage for the MG-ΛCDM cosmology."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.mg_lcdm import MGLCDMModel, get_optimisable_parameters
from cosmos.models.mg_lcdm.params import get_default_parameters


@pytest.fixture
def default_params() -> dict[str, float]:
    return get_default_parameters()


def _strip_mg_parameters(params: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in params.items() if key not in {"mu0", "Sigma0"}}


def test_mu_sigma_traces_dark_energy(default_params: dict[str, float]) -> None:
    params = {**default_params, "mu0": 0.12, "Sigma0": 0.25}
    model = MGLCDMModel(**params)
    norm = model.omega_de(1.0)
    for a in (1.0, 0.5, 0.1):
        expected_mu = 1.0 + params["mu0"] * model.omega_de(a) / norm
        expected_sigma = 1.0 + params["Sigma0"] * model.omega_de(a) / norm
        assert math.isclose(model.mu(a), expected_mu, rel_tol=1e-12)
        assert math.isclose(model.Sigma(a), expected_sigma, rel_tol=1e-12)


def test_gr_limit_matches_lcdm(default_params: dict[str, float]) -> None:
    lcdm_model = LCDMModel(**_strip_mg_parameters(default_params))
    mg_model = MGLCDMModel(**default_params)

    for z in (0.0, 0.5, 1.0, 2.0):
        assert math.isclose(mg_model.Hubble(z), lcdm_model.Hubble(z), rel_tol=1e-10)
        assert math.isclose(mg_model.growth_factor(z), lcdm_model.growth_factor(z), rel_tol=1e-8)
        assert math.isclose(mg_model.fs8(z), lcdm_model.fs8(z), rel_tol=1e-8)

    assert math.isclose(mg_model.sigma8(), lcdm_model.sigma8(), rel_tol=1e-9)
    assert math.isclose(mg_model.mu(0.1), 1.0, rel_tol=1e-12)
    assert math.isclose(mg_model.Sigma(0.1), 1.0, rel_tol=1e-12)


def test_nonzero_mg_smoke(default_params: dict[str, float]) -> None:
    mg_params = {**default_params, "mu0": 0.3, "Sigma0": 0.2}
    mg_model = MGLCDMModel(**mg_params)
    lcdm_model = LCDMModel(**_strip_mg_parameters(mg_params))

    assert math.isclose(mg_model.mu(0.01), 1.0, rel_tol=1e-5)
    assert mg_model.mu(1.0) > mg_model.mu(0.01)
    assert mg_model.Sigma(1.0) > mg_model.Sigma(0.01)

    z_grid = np.linspace(2.0, 0.0, 50)
    growth = np.array([mg_model.growth_factor(z) for z in z_grid])
    assert np.all(growth > 0.0)
    assert np.all(np.diff(growth) >= -1e-8)

    assert abs(mg_model.growth_factor(0.5) - lcdm_model.growth_factor(0.5)) > 1e-6


def test_api_surface() -> None:
    specs = get_optimisable_parameters()
    names = {spec["name"] for spec in specs}
    assert "mu0" in names and "Sigma0" in names

    model = MGLCDMModel(**get_default_parameters())
    assert callable(model.mu)
    assert callable(model.Sigma)
    assert math.isclose(model.mu(1.0), 1.0, rel_tol=1e-12)
    assert math.isclose(model.Sigma(1.0), 1.0, rel_tol=1e-12)
