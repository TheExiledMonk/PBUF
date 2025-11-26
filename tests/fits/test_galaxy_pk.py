import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.fits.galaxy_pk import load_galaxy_pk_dataset, run_galaxy_pk_fit
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


def test_galaxy_pk_dataset_loads():
    dataset = load_galaxy_pk_dataset()
    z = dataset["z"]
    obs = np.asarray(dataset["obs"], dtype=float)
    assert obs.ndim == 2
    assert len(dataset["labels"]) == obs.shape[1]
    assert obs.shape[0] == len(z)
    assert dataset["cov"].shape[0] == dataset["cov"].shape[1]
    assert dataset["cov"].shape[0] == obs.size
    assert "DM" in dataset["fiducials"]
    assert dataset["fiducials"]["DM"].shape[0] == len(z)


def test_galaxy_pk_predictions_change_with_lcdm_parameters():
    dataset = load_galaxy_pk_dataset()
    baseline = LCDMModel(**_lcdm_params())
    tuned = LCDMModel(**{**_lcdm_params(), "Omega_m0": 0.35})
    vec_base = run_galaxy_pk_fit(baseline, dataset)[1]["galaxy_pk_model_vector"]
    vec_tuned = run_galaxy_pk_fit(tuned, dataset)[1]["galaxy_pk_model_vector"]
    assert vec_base.shape == vec_tuned.shape
    assert not np.allclose(vec_base, vec_tuned)


def test_lcdm_and_pbuf_fit_runs_and_report_vectors():
    dataset = load_galaxy_pk_dataset()
    lcdm = LCDMModel(**_lcdm_params())
    pbuf = PBUFModel(thermal_table=ensure_thermal_table(), **_pbuf_params())

    expected_length = dataset["obs"].shape[0] * dataset["obs"].shape[1]
    for model in (lcdm, pbuf):
        chi2, extras = run_galaxy_pk_fit(model, dataset)
        assert np.isfinite(chi2)
        assert extras["galaxy_pk_model_vector"].shape == (expected_length,)


def test_missing_fiducials_raises_error():
    dataset = load_galaxy_pk_dataset()
    broken = copy.deepcopy(dataset)
    broken["fiducials"].pop("DM", None)
    model = LCDMModel(**_lcdm_params())
    with pytest.raises(ValueError, match="DM"):
        run_galaxy_pk_fit(model, broken)


def test_unknown_observable_label_raises():
    dataset = load_galaxy_pk_dataset()
    fake = {
        "name": "test",
        "type": "GALAXY_PK",
        "z": dataset["z"],
        "obs": np.zeros((len(dataset["z"]), 1), dtype=float),
        "cov": np.eye(len(dataset["z"]), dtype=float) * 0.01,
        "meta": {"test": True},
        "labels": ["unsupported"],
        "fiducials": dataset["fiducials"],
    }
    model = LCDMModel(**_lcdm_params())
    with pytest.raises(ValueError, match="Unsupported Galaxy PK observable"):
        run_galaxy_pk_fit(model, fake)


def test_evaluate_candidate_with_unhealthy_model_returns_infinite_chi2():
    params = _lcdm_params()
    params["Omega_m0"] = 1.5
    chi2, extras = evaluate_candidate("lcdm", params, datasets=["galaxy_pk"])
    assert chi2 == HUGE_CHI2
    assert extras["sanity_failed"]
