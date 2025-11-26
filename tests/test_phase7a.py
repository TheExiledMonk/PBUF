"""Unit tests for the Phase-7a sanity suite."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.model import PBUFModel
from cosmos.models.pbuf.phase7a import (
    Phase7aConfig,
    _check_hubble_grid,
    _check_omega_constraints,
    _check_thermal_lut,
    check_pbuf_phase7a_sanity,
)
from cosmos.models.pbuf.thermal_table import ThermalTable
from cosmos.optim.sanity_base import SanityResult


def _make_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for a in np.logspace(-9, -3, 6):
        rows.append(
            {
                "a": float(a),
                "z": float(1.0 / a - 1.0),
                "T": float(1.0 / a),
                "epsilon0_T": 0.001,
                "alpha_T": 0.0005,
                "dln_epsilon0_dlnT": 0.0,
                "dln_alpha_dlnT": 0.0,
                "g_star": 1.0,
                "g_starS": 1.0,
            }
        )
    return rows


@pytest.fixture
def thermal_table() -> ThermalTable:
    return ThermalTable(_make_rows())


@pytest.fixture
def pbuf_params() -> dict[str, float]:
    return {
        "H0": 70.0,
        "Omega_m0": 0.3,
        "Omega_r0": 1e-4,
        "alpha": 0.0,
        "Omega_b0": 0.05,
        "Rmax": 1.0e6,
    }


def _lcdm_factory(**kwargs: float) -> LCDMModel:
    return LCDMModel(**kwargs)


def _make_model(table: ThermalTable, params: dict[str, float]) -> PBUFModel:
    return PBUFModel(thermal_table=table, thermal_metadata={}, **params)


def _run_phase7a(params: dict[str, float], table: ThermalTable) -> SanityResult:
    model = _make_model(table, params)
    return check_pbuf_phase7a_sanity(params.copy(), model, lcdm_model_factory=_lcdm_factory)


def _run_thermal_check(table: ThermalTable, params: dict[str, float]) -> SanityResult:
    result = SanityResult()
    _check_thermal_lut(result, Phase7aConfig(), params.copy(), table)
    return result


def test_phase7a_rejects_alpha_range(thermal_table, pbuf_params):
    table = thermal_table
    table.alpha[2] = 0.2
    result = _run_thermal_check(table, pbuf_params)
    assert not result.ok
    assert "alpha" in "".join(result.reasons)


def test_phase7a_rejects_epsilon_range(thermal_table, pbuf_params):
    table = thermal_table
    table.eps[1] = 3.0
    result = _run_thermal_check(table, pbuf_params)
    assert not result.ok
    assert "epsilon0" in "".join(result.reasons)


def test_phase7a_rejects_k_sat_range(thermal_table, pbuf_params):
    table = thermal_table
    table.alpha[:] = table.eps + 0.1
    result = _run_thermal_check(table, pbuf_params)
    assert not result.ok
    assert "k_sat" in "".join(result.reasons)


def test_phase7a_rejects_alpha_smoothness(thermal_table, pbuf_params):
    table = thermal_table
    table.alpha[2] = table.alpha[1] + 0.02
    result = _run_thermal_check(table, pbuf_params)
    assert not result.ok
    assert "Δalpha" in "".join(result.reasons)


def test_phase7a_rejects_derivative_threshold(thermal_table, pbuf_params):
    table = thermal_table
    table.dln_alpha[0] = 6.0
    result = _run_thermal_check(table, pbuf_params)
    assert not result.ok
    assert "dln alpha" in "".join(result.reasons)



def test_phase7a_hubble_df_smoothness():
    config = Phase7aConfig()
    a_grid = np.logspace(np.log10(config.a_min), np.log10(config.a_max), config.n_a)
    H_values = np.ones_like(a_grid)
    split = len(H_values) // 2
    H_values[:split] = 2.0
    H_values[split:] = 0.5
    result = SanityResult()
    _check_hubble_grid(result, config, a_grid, H_values)
    assert not result.ok
    assert "Δ(d ln H/d ln a)" in "".join(result.reasons)


def test_phase7a_hubble_curvature_ratio():
    config = Phase7aConfig()
    a_grid = np.logspace(np.log10(config.a_min), np.log10(config.a_max), config.n_a)
    H_values = np.linspace(10.0, 0.1, config.n_a)
    result = SanityResult()
    _check_hubble_grid(result, config, a_grid, H_values)
    assert not result.ok
    assert "|H''/H'|" in "".join(result.reasons)


def test_phase7a_omega_constraints_negative(pbuf_params):
    config = Phase7aConfig()
    a_grid = np.logspace(np.log10(config.a_min), np.log10(config.a_max), config.n_a)
    omega_sigma = np.zeros_like(a_grid)
    omega_total = np.ones_like(a_grid)
    omega_sigma[3] = -0.01
    result = SanityResult()
    _check_omega_constraints(result, config, pbuf_params, a_grid, omega_sigma, omega_total)
    assert not result.ok
    assert "Ωσ<0" in "".join(result.reasons)


def test_phase7a_omega_constraints_exceeds(pbuf_params):
    config = Phase7aConfig()
    a_grid = np.logspace(np.log10(config.a_min), np.log10(config.a_max), config.n_a)
    omega_sigma = np.ones_like(a_grid) * 0.5
    omega_total = np.ones_like(a_grid) * 0.25
    result = SanityResult()
    _check_omega_constraints(result, config, pbuf_params, a_grid, omega_sigma, omega_total)
    assert not result.ok
    assert "Ωσ>Ω_total" in "".join(result.reasons)


def test_phase7a_closure_failure(thermal_table, pbuf_params):
    params = dict(pbuf_params)
    params["alpha"] = 0.8
    params["omega_normalization"] = "free"
    result = _run_phase7a(params, thermal_table)
    assert not result.ok
    assert "closure" in "".join(result.reasons)


def test_phase7a_early_lcdm_limit(monkeypatch, thermal_table, pbuf_params):
    model = _make_model(thermal_table, pbuf_params)

    def fake_lcdm_factory(**kwargs: float) -> LCDMModel:
        return LCDMModel(**kwargs)

    def fake_lcdm_H(a: float, params: LCDMModel) -> float:
        return 0.5

    monkeypatch.setattr("cosmos.models.pbuf.phase7a.lcdm_H_of_a", fake_lcdm_H)
    result = check_pbuf_phase7a_sanity(pbuf_params.copy(), model, lcdm_model_factory=fake_lcdm_factory)
    assert not result.ok
    assert "early LCDM" in "".join(result.reasons)
