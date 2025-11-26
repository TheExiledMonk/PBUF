"""Pytests covering the new DGP branching cosmology."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.models.dgp import DGPModel, get_default_parameters
from cosmos.models.lcdm.model import LCDMModel


@pytest.fixture
def default_dgp_params() -> dict[str, float | int]:
    return get_default_parameters()


def _build_lcdm_reference(params: dict[str, float | int]) -> LCDMModel:
    lcdm_settings = {
        "H0": params["H0"],
        "Omega_m0": params["Omega_m0"],
        "Omega_r0": params["Omega_r0"],
        "Omega_k0": params["Omega_k0"],
        "Omega_b0": params["Omega_b0"],
        "sigma8_0": params.get("sigma8_0", 0.811),
    }
    omega_lambda = 1.0 - (params["Omega_m0"] + params["Omega_r0"] + params["Omega_k0"])
    lcdm_settings["Omega_lambda0"] = omega_lambda
    return LCDMModel(**lcdm_settings)


def _matter_only_E(a: float, params: dict[str, float | int]) -> float:
    return math.sqrt(
        params["Omega_m0"] / a**3 + params["Omega_r0"] / a**4 + params["Omega_k0"] / a**2
    )


def test_epsilon_plus_behaves_like_lcdm_as_omegarc_vanishes(default_dgp_params: dict[str, float | int]) -> None:
    params = {**default_dgp_params, "Omega_rc": 1e-6, "epsilon_branch": 1}
    dgp = DGPModel(**params)
    for a in (0.05, 0.3, 0.7, 1.0):
        z = 1.0 / a - 1.0
        E_matter = _matter_only_E(a, params)
        assert math.isclose(dgp.E(a), E_matter, rel_tol=5e-2)


def test_branches_diverge_at_late_times(default_dgp_params: dict[str, float | int]) -> None:
    base = {**default_dgp_params, "Omega_rc": 0.05}
    plus = DGPModel(**{**base, "epsilon_branch": 1})
    minus = DGPModel(**{**base, "epsilon_branch": -1})
    assert plus.E(1.0) > minus.E(1.0)
    assert plus.mu(1.0) < minus.mu(1.0)


def test_mu_limits_and_positivity(default_dgp_params: dict[str, float | int]) -> None:
    params = {**default_dgp_params, "Omega_rc": 1e-3}
    dgp = DGPModel(**params)
    mus = [dgp.mu(a) for a in np.linspace(0.1, 1.0, 5)]
    assert all(math.isfinite(mu) and mu > 0.0 for mu in mus)
    small_rc = DGPModel(**{**params, "Omega_rc": 1e-10})
    assert math.isclose(small_rc.mu(1.0), 1.0, rel_tol=1e-4)


def test_growth_tracks_lcdm_at_high_z_and_is_suppressed_late(default_dgp_params: dict[str, float | int]) -> None:
    params = {**default_dgp_params, "Omega_rc": 0.02}
    dgp = DGPModel(**params)
    lcdm = _build_lcdm_reference(params)
    high_z = 2.0
    low_z = 0.1
    assert abs(dgp.growth_factor(high_z) - lcdm.growth_factor(high_z)) < 3e-2
    assert dgp.growth_factor(low_z) < lcdm.growth_factor(low_z)
