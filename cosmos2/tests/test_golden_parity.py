import json
from pathlib import Path

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from cosmos2.fits.registry import FIT_REGISTRY as COSMOS2_FITS  # noqa: E402
from cosmos2.models.model_factory import create_model as create_cosmos2_model  # noqa: E402


def _load_joint_config_path():
    candidate = Path("config/science_runs/minimal.json")
    if candidate.exists():
        return candidate
    pytest.skip("Joint config not available for golden parity harness")


def _load_joint_fits(path: Path):
    payload = json.loads(path.read_text())
    fits = payload.get("joint_config", {}).get("fits") or payload.get("fits")
    if not fits:
        pytest.skip("Joint config missing fits list")
    return fits


def _lcdm_params() -> list[dict[str, float]]:
    def _omega_lambda(p):
        return 1.0 - p["Omega_m0"] - p.get("Omega_r0", 9.0e-5) - p.get("Omega_k0", 0.0)

    candidates = [
        {"H0": 70.0, "Omega_m0": 0.3, "Omega_b0": 0.05, "Omega_k0": 0.0, "Omega_r0": 9.0e-5},
        {"H0": 67.4, "Omega_m0": 0.315, "Omega_b0": 0.049, "Omega_k0": 0.0, "Omega_r0": 9.0e-5},
    ]
    for entry in candidates:
        entry["Omega_Lambda"] = _omega_lambda(entry)
    return candidates


def _metrics_for_model(model, *, z_samples: np.ndarray) -> dict[str, np.ndarray]:
    H = np.asarray(model.Hubble(z_samples), dtype=float)
    E = H / float(model.parameters.get("H0", 1.0))
    DM = np.asarray(model.DM(z_samples), dtype=float)
    DV = np.asarray(model.DV(z_samples), dtype=float)
    rd = float(model.sound_horizon())
    mu = np.asarray(model.distance_modulus(z_samples), dtype=float)
    fs8 = np.asarray(model.fs8(z_samples), dtype=float)
    return {
        "E": E,
        "H": H,
        "DM": DM,
        "DV_over_rd": DV / rd if rd > 0 else np.full_like(DV, np.inf),
        "mu": mu,
        "fs8": fs8,
        "sigma8": float(model.sigma8()),
        "r_d": rd,
    }


def _assert_monotonic(values: np.ndarray) -> None:
    diffs = np.diff(values)
    assert np.all(diffs >= -1e-9)


def _check_fits(model, fit_names: list[str]) -> None:
    for fit in fit_names:
        fit_fn = COSMOS2_FITS.get(fit)
        if fit_fn is None:
            continue
        try:
            result = fit_fn(model)
        except FileNotFoundError:
            pytest.skip(f"Dataset for fit '{fit}' unavailable")
        chi2 = result[0] if isinstance(result, tuple) else result
        assert np.isfinite(chi2)


def test_lcdm_golden_metrics_and_fits_smoke():
    joint_path = _load_joint_config_path()
    joint_fits = _load_joint_fits(joint_path)
    z_samples = np.array([0.01, 0.25, 0.5, 1.0, 2.0], dtype=float)

    for params in _lcdm_params():
        model = create_cosmos2_model("lcdm", **params)
        metrics = _metrics_for_model(model, z_samples=z_samples)
        for arr in metrics.values():
            assert np.all(np.isfinite(arr))
        _assert_monotonic(metrics["DM"])
        _assert_monotonic(metrics["DV_over_rd"])
        _check_fits(model, joint_fits)
