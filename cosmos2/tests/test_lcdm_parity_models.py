import json
from pathlib import Path

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from cosmos2.fits.joint import build_joint_chi2_evaluator, resolve_joint_fits
from cosmos2.fits.registry import FIT_REGISTRY
from cosmos2.models.model_factory import create_model as create_cosmos2_model


def _params() -> dict[str, float]:
    omega_r0 = 9.0e-5
    omega_m0 = 0.3
    omega_k0 = 0.0
    omega_lambda = 1.0 - omega_m0 - omega_r0 - omega_k0
    return {
        "H0": 70.0,
        "Omega_m0": omega_m0,
        "Omega_b0": 0.05,
        "Omega_k0": omega_k0,
        "Omega_r0": omega_r0,
        "Omega_Lambda": omega_lambda,
        "sigma8_0": 0.8,
    }


def _load_joint_config_path():
    candidate = Path("config/science_runs/minimal.json")
    if candidate.exists():
        return candidate
    pytest.skip("Joint config not available for LCDM model checks")


def _load_joint_fits(path: Path):
    payload = json.loads(path.read_text())
    fits = payload.get("joint_config", {}).get("fits") or payload.get("fits")
    if not fits:
        pytest.skip("Joint config missing fits list")
    return fits


def test_lcdm_model_background_consistency():
    params = _params()
    model = create_cosmos2_model("lcdm", **params)

    z = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
    E = np.asarray(model.Hubble(z), dtype=float) / params["H0"]
    assert np.all(np.isfinite(E))
    assert np.all(E > 0.0)

    DM = np.asarray(model.DM(z), dtype=float)
    assert np.all(np.isfinite(DM))
    assert DM[0] == pytest.approx(0.0)
    assert np.all(np.diff(DM) >= 0.0)

    rd = float(model.sound_horizon())
    assert np.isfinite(rd) and rd > 0.0


def test_lcdm_joint_chi2_reduces_to_sum():
    joint_path = _load_joint_config_path()
    fits, weights = resolve_joint_fits(joint_path)
    params = _params()

    joint_fn = build_joint_chi2_evaluator(lambda p: create_cosmos2_model("lcdm", **p), joint_path)
    model = create_cosmos2_model("lcdm", **params)

    try:
        chi_joint = float(joint_fn(params))
    except FileNotFoundError:
        pytest.skip("Datasets not available for joint chi2 test")

    total = 0.0
    for fit_name in fits:
        fit_fn = FIT_REGISTRY.get(fit_name)
        if fit_fn is None:
            continue
        result = fit_fn(model)
        chi_val = result[0] if isinstance(result, tuple) else result
        total += float(weights.get(fit_name, 1.0)) * float(chi_val)

    assert np.isfinite(chi_joint)
    assert np.isclose(chi_joint, total, rtol=5e-2, atol=1.0)
