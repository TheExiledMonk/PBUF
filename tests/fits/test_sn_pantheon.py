import numpy as np
import pytest

from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.microphysics import ensure_thermal_table
from cosmos.models.pbuf.model import PBUFModel
from fits.sn.sn_pantheon import load_sn_pantheon_dataset, run_sn_pantheon_fit


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


def test_sn_dataset_loads():
    dataset = load_sn_pantheon_dataset()
    assert dataset["z"].shape == dataset["obs"].shape
    assert dataset["cov"] is not None
    assert dataset["cov"].shape[0] == dataset["cov"].shape[1] == len(dataset["z"])
    assert dataset["inv_cov"] is not None


def test_lcdm_sn_fit_changes_with_h0():
    dataset = load_sn_pantheon_dataset()
    model_low = LCDMModel(**_lcdm_params(h0=67.0))
    chi2_low, extras_low = run_sn_pantheon_fit(model_low, dataset)

    model_high = LCDMModel(**_lcdm_params(h0=71.0))
    chi2_high, extras_high = run_sn_pantheon_fit(model_high, dataset)

    assert np.isfinite(chi2_low)
    assert np.isfinite(chi2_high)
    assert chi2_low != pytest.approx(chi2_high)
    delta_mu = np.max(np.abs(extras_low["mu_model"] - extras_high["mu_model"]))
    assert delta_mu > 1e-3
    assert extras_low["mu_model"].shape == dataset["z"].shape


def test_pbuf_sn_fit_runs_and_returns_vector():
    dataset = load_sn_pantheon_dataset()
    model = PBUFModel(thermal_table=ensure_thermal_table(), **_pbuf_params())
    chi2, extras = run_sn_pantheon_fit(model, dataset)

    assert np.isfinite(chi2)
    assert extras["mu_model"].shape == dataset["z"].shape
