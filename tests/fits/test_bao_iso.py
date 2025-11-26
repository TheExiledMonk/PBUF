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
from cosmos.optim.sanity import evaluate_candidate
from cosmos.fits.bao_iso import load_bao_iso_dataset, run_bao_iso_fit


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


def test_bao_iso_dataset_loads():
    dataset = load_bao_iso_dataset()
    assert dataset["z"].shape == dataset["obs"].shape
    assert dataset["cov"].shape[0] == dataset["cov"].shape[1]
    assert dataset["cov"].shape[0] == len(dataset["z"])
    assert dataset["inv_cov"] is not None
    identity = dataset["cov"] @ dataset["inv_cov"]
    assert np.allclose(identity, np.eye(identity.shape[0]), atol=1e-8)


def test_lcdm_bao_iso_fit_changes_with_h0():
    dataset = load_bao_iso_dataset()

    model_low = LCDMModel(**_lcdm_params(h0=67.0))
    chi2_low, extras_low = run_bao_iso_fit(model_low, dataset)

    model_high = LCDMModel(**_lcdm_params(h0=71.0))
    chi2_high, extras_high = run_bao_iso_fit(model_high, dataset)

    assert np.isfinite(chi2_low)
    assert np.isfinite(chi2_high)
    assert chi2_low != pytest.approx(chi2_high)

    delta = np.max(np.abs(extras_low["DV_over_rd_model"] - extras_high["DV_over_rd_model"]))
    assert delta > 1e-3


def test_pbuf_bao_iso_fit_runs_and_reports_vector():
    dataset = load_bao_iso_dataset()
    model = PBUFModel(thermal_table=ensure_thermal_table(), **_pbuf_params())

    chi2, extras = run_bao_iso_fit(model, dataset)

    assert np.isfinite(chi2)
    assert extras["DV_over_rd_model"].shape == dataset["z"].shape


def test_bao_iso_evaluate_candidate_integration():
    params = _lcdm_params()
    chi2, extras = evaluate_candidate("lcdm", params, datasets=["bao_iso"])

    assert chi2 >= 0.0
    assert extras["dataset_summaries"]["bao_iso"]["chi2"] == pytest.approx(chi2)
    assert "DV_over_rd_model" in extras["dataset_summaries"]["bao_iso"]
