"""Regression tests that compare cosmos2 PBUF outputs to the legacy cosmos_old stack."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "numba" not in sys.modules:
    numba_stub = types.SimpleNamespace(njit=lambda *args, **kwargs: (lambda fn: fn))
    sys.modules["numba"] = numba_stub

if not hasattr(np, "cumtrapz"):
    def _cumtrapz(y: np.ndarray, x: np.ndarray, initial: float | None = None) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        dx = np.diff(x_arr)
        integrand = dx * 0.5 * (y_arr[:-1] + y_arr[1:])
        cumulative = np.cumsum(integrand)
        if initial is None:
            return cumulative
        return np.concatenate(([float(initial)], cumulative + float(initial)))

    np.cumtrapz = _cumtrapz  # type: ignore[attr-defined]


BASELINE_PARAMS = dict(
    H0=70.0,
    Omega_m0=0.3,
    Omega_b0=0.05,
    Omega_r0=1.0e-4,
    alpha=0.001,
    Rmax=1.0e6,
)


def _install_cosmos_old_shim() -> Path:
    """
    Expose the cosmos_old tree as a synthetic ``cosmos`` package so we can
    import the legacy PBUF model without resurrecting the full legacy layout.
    """

    root = Path(__file__).resolve().parents[2] / "cosmos_old"
    if not root.exists():
        pytest.skip("cosmos_old reference tree missing")

    if "cosmos" not in sys.modules:
        module = types.ModuleType("cosmos")
        module.__path__ = [str(root)]
        sys.modules["cosmos"] = module
    else:
        module = sys.modules["cosmos"]
        path = list(getattr(module, "__path__", []))
        if str(root) not in path:
            module.__path__ = path + [str(root)]
    return root


def _make_rows() -> list[dict[str, float]]:
    a_grid = np.logspace(-4, 0, 32)
    rows: list[dict[str, float]] = []
    for a in a_grid:
        rows.append(
            {
                "a": float(a),
                "z": float(1.0 / a - 1.0),
                "T": 2.7255 / a,
                "epsilon0_T": 0.002 + 5.0e-4 * (1.0 - a),
                "alpha_T": 0.001 + 2.0e-4 * (1.0 - a),
                "dln_epsilon0_dlnT": 0.0,
                "dln_alpha_dlnT": 0.0,
                "g_star": 3.36,
                "g_starS": 3.9,
            }
        )
    return rows


def _build_models():
    _install_cosmos_old_shim()
    from cosmos2.models.pbuf.model import PBUFModel as NewPBUFModel
    from cosmos2.models.pbuf.thermal_table import ThermalTable as NewThermalTable
    from cosmos.models.pbuf.model import PBUFModel as LegacyPBUFModel
    from cosmos.models.pbuf.thermal_table import ThermalTable as LegacyThermalTable

    rows = _make_rows()

    new_model = NewPBUFModel(
        thermal_table=NewThermalTable(rows),
        thermal_metadata={},
        normalization_mode="flat_today",
        **BASELINE_PARAMS,
    )
    legacy_model = LegacyPBUFModel(
        thermal_table=LegacyThermalTable(rows),
        thermal_metadata={},
        normalization_mode="flat_today",
        **BASELINE_PARAMS,
    )
    new_model.thermal_table = new_model._thermal
    return new_model, legacy_model


def test_background_matches_legacy():
    new_model, legacy_model = _build_models()

    a_samples = np.logspace(-4, 0, 16, dtype=float)
    z_samples = 1.0 / a_samples - 1.0

    from cosmos2.models.pbuf import distances as dist_new
    from cosmos.models.pbuf import distances as dist_old

    E_new = np.array([dist_new.E(a, new_model._params, new_model._thermal) for a in a_samples], dtype=float)
    E_old = np.array([dist_old.E(a, legacy_model.params, legacy_model.thermal_table) for a in a_samples], dtype=float)
    np.testing.assert_allclose(E_new, E_old, rtol=1e-10, atol=0.0)

    H_new_direct = np.array([dist_new.H(a, new_model._params, new_model._thermal) for a in a_samples], dtype=float)
    H_old_direct = np.array([dist_old.H(a, legacy_model.params, legacy_model.thermal_table) for a in a_samples], dtype=float)
    np.testing.assert_allclose(H_new_direct, H_old_direct, rtol=1e-10, atol=0.0)

    H_new = np.asarray(new_model.Hubble(z_samples), dtype=float)
    H_old = np.asarray(legacy_model.Hubble(z_samples), dtype=float)
    np.testing.assert_allclose(H_new, H_old, rtol=1e-5, atol=0.0)

    DM_new = np.asarray(new_model.DM(z_samples), dtype=float)
    DM_old = np.asarray(legacy_model.DM(z_samples), dtype=float)
    np.testing.assert_allclose(DM_new, DM_old, rtol=1e-2, atol=0.0)


def test_sound_horizon_and_cmb_match_legacy():
    new_model, legacy_model = _build_models()

    np.testing.assert_allclose(new_model.sound_horizon(), legacy_model.sound_horizon(), rtol=1e-6, atol=0.0)

    cmb_new = new_model.cmb(None)
    cmb_old = legacy_model.cmb(None)
    np.testing.assert_allclose(
        [cmb_new.R, cmb_new.l_A, cmb_new.theta_star],
        [cmb_old.R, cmb_old.l_A, cmb_old.theta_star],
        rtol=1e-6,
        atol=0.0,
    )

    from cosmos2.models.pbuf.fits import run_cmb_fit as run_cmb_new
    from cosmos.fits.cmb.cmb import run_fit as run_cmb_old

    chi_new, _ = run_cmb_new(new_model)
    chi_old, _ = run_cmb_old(legacy_model)
    np.testing.assert_allclose(chi_new, chi_old, rtol=1e-6, atol=0.0)


def test_joint_chi2_matches_legacy_default_config():
    _install_cosmos_old_shim()
    from cosmos2.models.pbuf.fits import build_pbuf_joint_chi2
    from cosmos.fits.joint import build_joint_chi2_evaluator as build_joint_old
    from cosmos2.models.pbuf.model import PBUFModel as NewPBUFModel
    from cosmos.models.pbuf.model import PBUFModel as LegacyPBUFModel
    from cosmos2.models.pbuf.thermal_table import ThermalTable as NewThermalTable
    from cosmos.models.pbuf.thermal_table import ThermalTable as LegacyThermalTable

    rows = _make_rows()
    joint_config = Path(__file__).resolve().parents[2] / "configs" / "joint" / "default.json"

    def new_factory(_params: dict[str, float]) -> NewPBUFModel:
        model = NewPBUFModel(
            thermal_table=NewThermalTable(rows),
            thermal_metadata={},
            normalization_mode="flat_today",
            **BASELINE_PARAMS,
        )
        model.thermal_table = model._thermal
        return model

    def legacy_factory(_params: dict[str, float]) -> LegacyPBUFModel:
        return LegacyPBUFModel(
            thermal_table=LegacyThermalTable(rows),
            thermal_metadata={},
            normalization_mode="flat_today",
            **BASELINE_PARAMS,
        )

    chi2_new = build_pbuf_joint_chi2(new_factory, joint_config)
    chi2_old = build_joint_old(legacy_factory, joint_config)

    np.testing.assert_allclose(chi2_new(BASELINE_PARAMS), chi2_old(BASELINE_PARAMS), rtol=2e-3, atol=0.0)
