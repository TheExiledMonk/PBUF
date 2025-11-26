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
from cosmos.fits.rsd import load_rsd_dataset, run_rsd_fit


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


def test_rsd_dataset_loads():
    dataset = load_rsd_dataset()
    assert dataset["z"].shape == dataset["obs"].shape
    assert dataset["cov"].shape[0] == dataset["cov"].shape[1]
    assert dataset["cov"].shape[0] == len(dataset["z"])
    assert "meta" in dataset
    assert dataset["inv_cov"] is not None


def test_lcdm_rsd_predictions_change_with_parameters():
    dataset = load_rsd_dataset()
    z = dataset["z"]

    baseline = LCDMModel(**_lcdm_params())
    tuned = LCDMModel(**{**_lcdm_params(), "Omega_m0": 0.25})

    fs8_baseline = baseline.fs8(z)
    fs8_tuned = tuned.fs8(z)

    assert fs8_baseline.shape == fs8_tuned.shape
    assert not np.allclose(fs8_baseline, fs8_tuned)


def test_pbuf_rsd_fit_runs_and_reports_vector():
    dataset = load_rsd_dataset()
    model = PBUFModel(thermal_table=ensure_thermal_table(), **_pbuf_params())

    chi2, extras = run_rsd_fit(model, dataset)
    assert np.isfinite(chi2)
    assert extras["fs8_model"].shape == dataset["z"].shape


def test_rsd_candidate_with_insane_model_returns_infinite_chi2():
    params = _lcdm_params()
    params["Omega_m0"] = 1.5
    params["Omega_b0"] = 0.1
    params["Omega_k0"] = 0.0

    chi2, extras = evaluate_candidate("lcdm", params, datasets=["rsd"])
    assert chi2 == HUGE_CHI2
    assert extras["sanity_failed"]
