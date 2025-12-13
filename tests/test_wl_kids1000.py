import math

import numpy as np

from cosmos2.fits.weak_lensing_kids1000 import run_wl_kids1000_fit
from cosmos2.wl.kids import standardize_kids1000, tomo_pairs
from cosmos2.wl.theory import compute_shear_predictions
from cosmos2.wl.backend import WeakLensingBackend


def test_standardize_kids1000_flatten_ordering() -> None:
    xi_plus = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    xi_minus = xi_plus + 100.0
    raw = {
        "xi_plus": xi_plus,
        "xi_minus": xi_minus,
        "theta": np.array([1.0, 2.0, 3.0]),
        "theta_units": "deg",
        "nz": np.ones((2, 4), dtype=float),
        "z_grid": np.linspace(0.0, 1.0, 4),
        "covariance": np.eye(18),
    }
    standardized = standardize_kids1000(raw)
    pairs = tomo_pairs(2)
    assert standardized["tomo_pairs"].shape == (len(pairs), 2)
    assert standardized["data_vector"].shape == (18,)
    block_len = 3
    # xi_plus flattening: (0,0), (0,1), (1,1)
    assert np.allclose(standardized["data_vector"][0:block_len], xi_plus[0, 0])
    assert np.allclose(standardized["data_vector"][block_len:2 * block_len], xi_plus[0, 1])
    assert np.allclose(standardized["data_vector"][2 * block_len:3 * block_len], xi_plus[1, 1])
    # First theta bin should be converted to radians
    assert math.isclose(standardized["theta_bins"][0], math.radians(1.0))


class _ToyModel:
    def __init__(self) -> None:
        self.parameters = {"H0": 70.0, "Omega_m0": 0.3, "sigma8_0": 0.8}

    def omega_m0(self) -> float:
        return 0.3

    def sigma8(self) -> float:
        return 0.8

    def DM(self, z):
        arr = np.asarray(z, dtype=float)
        return 3000.0 * arr

    def Hubble(self, z):
        arr = np.asarray(z, dtype=float)
        return np.full_like(arr, 70.0, dtype=float)

    def growth_factor(self, z):
        arr = np.asarray(z, dtype=float)
        return 1.0 / (1.0 + arr)


def test_compute_shear_predictions_produces_finite_vector() -> None:
    n_bins = 2
    n_theta = 3
    z_grid = np.linspace(0.01, 1.0, 5)
    n_of_z = np.ones((n_bins, z_grid.size), dtype=float)
    theta = np.deg2rad(np.array([1.0, 5.0, 10.0]))
    data_vector = np.zeros(n_bins * (n_bins + 1) // 2 * n_theta * 2, dtype=float)
    backend = WeakLensingBackend(_ToyModel())
    xi_plus, xi_minus, model_vector = compute_shear_predictions(
        backend,
        data_vector,
        n_of_z,
        z_grid,
        theta,
        shear_m=np.zeros(n_bins),
        ell_min=2,
        ell_max=50,
    )
    assert model_vector.shape == data_vector.shape
    assert np.all(np.isfinite(model_vector))
    assert np.all(np.isfinite(xi_plus))
    assert np.all(np.isfinite(xi_minus))


def test_run_wl_kids1000_fit_accepts_synthetic_dataset() -> None:
    model = _ToyModel()
    n_bins = 2
    n_theta = 3
    pairs = tomo_pairs(n_bins)
    vector_length = len(pairs) * n_theta * 2
    dataset = {
        "data_vector": np.zeros(vector_length, dtype=float),
        "covariance": np.eye(vector_length, dtype=float),
        "theta_bins": np.deg2rad(np.array([1.0, 5.0, 10.0])),
        "n_of_z": np.ones((n_bins, 5), dtype=float),
        "z_grid": np.linspace(0.01, 1.0, 5),
        "shear_m": np.zeros(n_bins),
        "meta": {"data_order": "xi_plus_then_xi_minus"},
    }
    chi2, extras = run_wl_kids1000_fit(model, dataset)
    assert math.isfinite(chi2)
    assert "predictions" in extras
