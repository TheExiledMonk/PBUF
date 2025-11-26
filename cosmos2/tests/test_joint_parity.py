import json
from pathlib import Path

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from cosmos2.fits.joint import build_joint_chi2_evaluator as build_joint_cosmos2, resolve_joint_fits  # noqa: E402
from cosmos2.fits.registry import FIT_REGISTRY  # noqa: E402
from cosmos2.models.model_factory import create_model as create_cosmos2_model  # noqa: E402


def _load_joint_config_path():
    candidate = Path("config/science_runs/minimal.json")
    if candidate.exists():
        return candidate
    pytest.skip("Joint config not available for parity test")


def _load_joint_fits(path: Path):
    payload = json.loads(path.read_text())
    fits = payload.get("joint_config", {}).get("fits") or payload.get("fits")
    if not fits:
        pytest.skip("Joint config missing fits list")
    return fits


def test_joint_chi2_parity_lcdm_minimal():
    joint_path = _load_joint_config_path()
    fits, weights = resolve_joint_fits(joint_path)

    joint_cosmos2 = build_joint_cosmos2(lambda p: create_cosmos2_model("lcdm", **p), joint_path)

    params = {
        "H0": 70.0,
        "Omega_m0": 0.3,
        "Omega_b0": 0.05,
        "Omega_k0": 0.0,
        "Omega_r0": 9.0e-5,
        "Omega_Lambda": 1.0 - 0.3 - 9.0e-5,
    }

    try:
        chi_cosmos2 = float(joint_cosmos2(params))
    except FileNotFoundError:
        pytest.skip("Datasets not available for joint parity test")

    assert np.isfinite(chi_cosmos2)

    # Manually recompute weighted sum of fits and compare.
    model = create_cosmos2_model("lcdm", **params)
    total = 0.0
    for fit_name in fits:
        fit_fn = FIT_REGISTRY.get(fit_name)
        if fit_fn is None:
            continue
        result = fit_fn(model)
        chi_val = result[0] if isinstance(result, tuple) else result
        weight = float(weights.get(fit_name, 1.0))
        total += weight * float(chi_val)
    assert np.isclose(chi_cosmos2, total, rtol=1e-6, atol=1e-6)
