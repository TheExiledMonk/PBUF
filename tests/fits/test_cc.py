import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.fits.cc import load_cc_dataset, run_cc_fit
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.microphysics import ensure_thermal_table
from cosmos.models.pbuf.model import PBUFModel
from cosmos.optim.sanity import HUGE_CHI2, evaluate_candidate


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


def test_cc_dataset_loads():
    dataset = load_cc_dataset()
    z = dataset["z"]
    cov = dataset["cov"]
    inv_cov = dataset["inv_cov"]

    assert z.shape == dataset["obs"].shape
    assert cov.shape[0] == cov.shape[1] == len(z)
    assert inv_cov is not None
    identity = cov @ inv_cov
    assert np.allclose(identity, np.eye(identity.shape[0]), atol=1e-8)


def test_lcdm_cc_fit_changes_with_h0():
    dataset = load_cc_dataset()
    model_low = LCDMModel(**_lcdm_params(h0=67.0))
    chi2_low, extras_low = run_cc_fit(model_low, dataset)

    model_high = LCDMModel(**_lcdm_params(h0=71.0))
    chi2_high, extras_high = run_cc_fit(model_high, dataset)

    assert np.isfinite(chi2_low)
    assert np.isfinite(chi2_high)
    assert chi2_low != pytest.approx(chi2_high)

    delta = np.max(np.abs(extras_low["H_model"] - extras_high["H_model"]))
    assert delta > 1e-3


def test_pbuf_cc_fit_runs_and_reports_vector():
    dataset = load_cc_dataset()
    model = PBUFModel(thermal_table=ensure_thermal_table(), **_pbuf_params())

    chi2, extras = run_cc_fit(model, dataset)

    assert np.isfinite(chi2)
    assert extras["H_model"].shape == dataset["z"].shape


def test_cc_fit_module_has_no_friedmann_terms():
    try:
        result = subprocess.run(
            ["rg", "-n", "Omega", "cosmos/fits/cc/cc.py"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("rg is not available in this environment")
        return

    assert result.returncode == 1
    assert result.stdout == ""


def test_evaluate_candidate_cc_summary():
    params = _lcdm_params()
    chi2, extras = evaluate_candidate("lcdm", params, datasets=["cc"])

    assert np.isfinite(chi2)
    summary = extras["dataset_summaries"]["cc"]
    assert summary["chi2"] == pytest.approx(chi2)
    dataset = load_cc_dataset()
    assert summary["H_model"].shape == dataset["z"].shape


def test_evaluate_candidate_cc_sanity_failure():
    params = _lcdm_params()
    params["Omega_m0"] = 1.2

    chi2, extras = evaluate_candidate("lcdm", params, datasets=["cc"])

    assert chi2 == pytest.approx(HUGE_CHI2)
    assert extras["sanity_failed"] is True
    assert "cc" in extras["dataset_summaries"]
