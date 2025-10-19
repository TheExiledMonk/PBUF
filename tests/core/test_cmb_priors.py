import numpy as np

from core import cmb_priors, pbuf_models


def test_theta_star_monotonic_in_ksat():
    base_params = {
        "H0": 67.4,
        "Om0": 0.315,
        "Obh2": 0.02237,
    }
    k_values = np.array([0.5, 1.0, 2.0, 4.0])
    theta_values = [
        cmb_priors.theta_star({**base_params, "k_sat": float(k)}, model=pbuf_models) for k in k_values
    ]
    assert all(theta_values[i] > theta_values[i + 1] for i in range(len(theta_values) - 1))


def test_chi2_cmb_zero_for_identical_predictions():
    labels = ["R", "lA", "Omegabh2", "ns"]
    means = {label: float(idx + 1) for idx, label in enumerate(labels)}
    cov = np.eye(len(labels))
    priors = {"means": means, "labels": labels, "cov": cov}
    assert cmb_priors.chi2_cmb(means, priors) == 0.0
