import numpy as np

from cosmos.pbuf.model import PBUF
from cosmos.pbuf.equations import omega_sigma_total


def _build_model(params):
    return PBUF(
        omega_m=params["Om0"],
        h=params["H0"] / 100.0,
        alpha=params["alpha"],
        Rmax=params["Rmax"],
        k_sat=params["k_sat"],
        eps0=params.get("eps0", 0.7),
        omega_k=params.get("Ok0", 0.0),
        omega_r=params.get("Or0", 9.2e-5),
    )


def test_elastic_influence_visible():
    base_params = {
        "H0": 70.0,
        "Om0": 0.3,
        "Or0": 9.2e-5,
        "Ok0": 0.0,
        "alpha": 1.0,
        "eps0": 1.0,
        "Rmax": 0.8,
        "k_sat": 1.0,
        "n_alpha": 2.0,
        "n_eps": 1.0,
    }

    alt_params = {
        **base_params,
        "Rmax": 5.0,
        "k_sat": 0.5,
        "n_alpha": 4.0,
        "n_eps": 2.0,
    }

    base_model = _build_model(base_params)
    alt_model = _build_model(alt_params)

    z_grid = np.linspace(0.0, 2.0, 100)
    H_base = np.array([base_model.H(z) for z in z_grid])
    H_alt = np.array([alt_model.H(z) for z in z_grid])

    frac = np.max(np.abs((H_alt - H_base) / H_base))

    assert frac > 1e-4, "Elastic sector ineffective (<1e-4 impact on H(z))"


def test_elastic_density_positive():
    params = {
        "H0": 70.0,
        "Om0": 0.3,
        "Or0": 9.2e-5,
        "Ok0": 0.0,
        "alpha": 1.0,
        "eps0": 1.0,
        "Rmax": 1.0,
        "k_sat": 1.0,
        "n_alpha": 2.0,
        "n_eps": 1.0,
        "n_R": 0.0,
    }
    omega_today = omega_sigma_total(1.0, params)
    omega_high_z = omega_sigma_total(1.0 / (1.0 + 5.0), params)
    assert omega_today > 0.0
    assert omega_high_z >= 0.0
    assert omega_high_z < omega_today, "Elastic density should diminish towards high redshift."
