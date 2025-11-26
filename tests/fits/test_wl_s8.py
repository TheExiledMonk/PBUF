import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.microphysics import ensure_thermal_table
from cosmos.models.pbuf.model import PBUFModel
from cosmos.optim.sanity import HUGE_CHI2, evaluate_candidate
from cosmos.fits.wl import load_wl_s8_dataset, run_wl_s8_fit


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


def test_wl_dataset_loads():
    dataset = load_wl_s8_dataset()
    assert dataset["S8_obs"].shape == dataset["S8_err"].shape
    assert dataset["gamma"].shape == dataset["S8_obs"].shape
    assert "cov" in dataset
    assert "inv_cov" in dataset
    cov = dataset["cov"]
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1] == len(dataset["S8_obs"])


def test_lcdm_s8_predictions_change_when_parameters_vary():
    dataset = load_wl_s8_dataset()
    baseline = LCDMModel(**_lcdm_params())
    tuned = LCDMModel(**{**_lcdm_params(), "Omega_m0": 0.25})

    _, base_extras = run_wl_s8_fit(baseline, dataset)
    _, tuned_extras = run_wl_s8_fit(tuned, dataset)

    assert base_extras["S8_model"].shape == tuned_extras["S8_model"].shape
    assert not np.allclose(base_extras["S8_model"], tuned_extras["S8_model"])


def test_wl_candidate_with_insane_model_returns_infinite_chi2():
    params = _lcdm_params()
    params["Omega_m0"] = 1.5
    params["Omega_b0"] = 0.1
    params["Omega_k0"] = 0.0

    chi2, extras = evaluate_candidate("lcdm", params, datasets=["wl_s8"])
    assert chi2 == HUGE_CHI2
    assert extras["sanity_failed"]
