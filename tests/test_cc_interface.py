"""
Regression tests for the CC fitting interface.

These tests ensure the high-level helpers build cosmological models with the
correct baryon density and validate that missing uncertainty information is
handled gracefully.
"""

from unittest import mock

import numpy as np
import pytest

from cosmos.fits.cc.observables import chi2_cc
from cosmos.fits.cc.chi2 import chi_squared_cc


def _mock_dataset(err=None, cov=None):
    return {
        "type": "CC",
        "z": np.array([0.1]),
        "obs": np.array([70.0]),
        "err": None if err is None else np.array(err, dtype=float),
        "cov": cov,
    }


def test_chi2_cc_converts_obh2_to_omega_b_for_lcdm():
    dataset = _mock_dataset(err=[1.0])
    h0 = 70.0
    h = h0 / 100.0
    obh2 = 0.022
    expected_omega_b = obh2 / (h**2)

    with mock.patch("data_interface.cc_loader.load_cc_data", return_value=dataset), \
         mock.patch("data_interface.standardize.ensure_standard_dataset", side_effect=lambda data, _: data), \
         mock.patch("cosmos.fits.cc.observables.LCDM") as mock_lcdm:

        mock_model = mock.Mock()
        mock_model.H.return_value = dataset["obs"][0]
        mock_lcdm.return_value = mock_model

        params = {
            "H0": h0,
            "Om0": 0.3,
            "Ok0": 0.0,
            "Ol0": 0.7,
            "Or0": 5e-5,
            "Obh2": obh2,
        }

        chi2_cc(params, model_type="lcdm")

        assert mock_lcdm.call_count == 1
        kwargs = mock_lcdm.call_args.kwargs
        assert kwargs["h"] == pytest.approx(h)
        assert kwargs["omega_b"] == pytest.approx(expected_omega_b)


def test_chi2_cc_converts_obh2_to_omega_b_for_pbuf():
    dataset = _mock_dataset(err=[1.0])
    h0 = 67.4
    h = h0 / 100.0
    obh2 = 0.02237
    expected_omega_b = obh2 / (h**2)

    with mock.patch("data_interface.cc_loader.load_cc_data", return_value=dataset), \
         mock.patch("data_interface.standardize.ensure_standard_dataset", side_effect=lambda data, _: data), \
         mock.patch("cosmos.fits.cc.observables.PBUF") as mock_pbuf:

        mock_model = mock.Mock()
        mock_model.H.return_value = dataset["obs"][0]
        mock_pbuf.return_value = mock_model

        params = {
            "H0": h0,
            "Om0": 0.315,
            "Ok0": 0.0,
            "Or0": 5e-5,
            "Obh2": obh2,
            "alpha": 1e-3,
            "Rmax": 1e8,
            "k_sat": 1.0,
        }

        chi2_cc(params, model_type="pbuf")

        assert mock_pbuf.call_count == 1
        kwargs = mock_pbuf.call_args.kwargs
        assert kwargs["h"] == pytest.approx(h)
        assert kwargs["omega_b"] == pytest.approx(expected_omega_b)


def test_chi_squared_cc_requires_uncertainties():
    dataset = _mock_dataset(err=None, cov=None)
    mock_model = mock.Mock()
    mock_model.H.return_value = dataset["obs"][0]

    with pytest.raises(ValueError, match="covariance matrix or 1σ uncertainties"):
        chi_squared_cc(mock_model, data=dataset)
