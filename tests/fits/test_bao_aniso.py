import sys
from pathlib import Path

import numpy as np
import pytest
import warnings

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.fits.bao_aniso import load_bao_aniso_dataset, run_bao_aniso_fit
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.microphysics import ensure_thermal_table
from cosmos.models.pbuf.model import PBUFModel
from cosmos.optim.sanity import evaluate_candidate, HUGE_CHI2


def _lcdm_params(h0: float = 67.0) -> dict[str, float]:
    return {
        "H0": h0,
        "Omega_m0": 0.3,
        "Omega_b0": 0.05,
        "Omega_r0": 9.0e-5,
        "Omega_k0": 0.0,
    }


def _pbuf_params(h0: float = 67.0) -> dict[str, float]:
    return {
        "H0": h0,
        "Omega_m0": 0.3,
        "Omega_b0": 0.05,
        "Omega_r0": 9.0e-5,
        "alpha": 0.0,
        "Rmax": 1.0e6,
    }


def test_bao_aniso_dataset_loads():
    dataset = load_bao_aniso_dataset()

    expected_size = len(dataset["z"]) * dataset["observables_per_bin"]
    assert dataset["obs"].shape == (expected_size,)
    assert dataset["cov"].shape[0] == expected_size
    assert dataset["cov"].shape[1] == expected_size
    identity = dataset["cov"] @ dataset["inv_cov"]
    assert np.allclose(identity, np.eye(identity.shape[0]), atol=1e-8)


def test_lcdm_bao_aniso_fit_changes_with_h0():
    dataset = load_bao_aniso_dataset()

    model_low = LCDMModel(**_lcdm_params(h0=67.0))
    chi2_low, extras_low = run_bao_aniso_fit(model_low, dataset)

    model_high = LCDMModel(**_lcdm_params(h0=71.0))
    chi2_high, extras_high = run_bao_aniso_fit(model_high, dataset)

    assert np.isfinite(chi2_low)
    assert np.isfinite(chi2_high)
    assert chi2_low != pytest.approx(chi2_high)

    delta = np.max(np.abs(extras_low["bao_aniso_model"] - extras_high["bao_aniso_model"]))
    assert delta > 1e-3


def test_pbuf_bao_aniso_fit_runs_and_reports_vector():
    dataset = load_bao_aniso_dataset()
    model = PBUFModel(thermal_table=ensure_thermal_table(), **_pbuf_params())

    chi2, extras = run_bao_aniso_fit(model, dataset)

    assert np.isfinite(chi2)
    assert extras["bao_aniso_model"].shape == dataset["obs"].shape


def test_bao_aniso_model_isolation_from_cosmology_math():
    content = Path("cosmos/fits/bao_aniso/bao_aniso.py").read_text()
    forbidden = {"comoving_distance", "simpson", "integrator", "H_z"}
    for token in forbidden:
        assert token not in content, f"bao_aniso.py should not implement '{token}'"


def test_bao_aniso_invalid_candidate_returns_huge_chi2():
    params = _lcdm_params(h0=67.0)
    params["Omega_m0"] = 0.6
    params["Omega_b0"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        chi2, extras = evaluate_candidate("lcdm", params, datasets=["bao_aniso"])
    assert chi2 == HUGE_CHI2
    assert extras["sanity_failed"] is True
