import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.fits.lensing_cross import load_lensing_cross_dataset, run_lensing_cross_fit
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


def test_lensing_cross_dataset_loads():
    dataset = load_lensing_cross_dataset()
    assert dataset["A_obs"].shape == dataset["A_err"].shape == dataset["p_exponent"].shape
    assert dataset["q_exponent"].shape == dataset["z_eff"].shape == dataset["S8_fid"].shape
    assert dataset["fs8_fid"].shape == dataset["gamma"].shape
    assert dataset["weights"].shape == dataset["A_obs"].shape
    assert dataset["n_datasets"] == len(dataset["A_obs"])
    if "cov" in dataset:
        assert "inv_cov" in dataset
    else:
        assert "inv_cov" not in dataset
    assert np.all(dataset["weights"] > 0.0)


def test_lcdm_amplitudes_change_when_parameters_vary():
    dataset = load_lensing_cross_dataset()
    base = LCDMModel(**_lcdm_params())
    tuned = LCDMModel(**{**_lcdm_params(), "Omega_m0": 0.25})

    _, base_extras = run_lensing_cross_fit(base, dataset)
    _, tuned_extras = run_lensing_cross_fit(tuned, dataset)

    assert base_extras["A_model"].shape == tuned_extras["A_model"].shape
    assert not np.allclose(base_extras["A_model"], tuned_extras["A_model"])


def test_pbuf_amplitude_matches_manual_scaling():
    dataset = load_lensing_cross_dataset()
    model = PBUFModel(thermal_table=ensure_thermal_table(), **_pbuf_params())

    chi2, extras = run_lensing_cross_fit(model, dataset)
    assert np.isfinite(chi2)
    A_model = extras["A_model"]

    gamma = np.asarray(dataset["gamma"], dtype=float)
    S8_model = np.asarray([model.S8(g) for g in gamma], dtype=float)
    fs8_model = np.asarray(model.fs8(np.asarray(dataset["z_eff"], dtype=float)), dtype=float)
    expected = (S8_model / dataset["S8_fid"]) ** dataset["p_exponent"] * (
        fs8_model / dataset["fs8_fid"]
    ) ** dataset["q_exponent"]

    assert np.allclose(A_model, expected)


def test_lensing_cross_scaled_errors_reflect_weights():
    dataset = load_lensing_cross_dataset()
    baseline = PBUFModel(thermal_table=ensure_thermal_table(), **_pbuf_params())
    _, extras = run_lensing_cross_fit(baseline, dataset)
    assert "scaled_err" in extras
    assert np.allclose(extras["scaled_err"], dataset["A_err"] / dataset["weights"])


def test_lensing_cross_candidate_with_insane_model_returns_infinite_chi2():
    params = _lcdm_params()
    params["Omega_m0"] = 1.5
    params["Omega_b0"] = 0.1

    chi2, extras = evaluate_candidate("lcdm", params, datasets=["lensing_cross"])
    assert chi2 == HUGE_CHI2
    assert extras["sanity_failed"]
