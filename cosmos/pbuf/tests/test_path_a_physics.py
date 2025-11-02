"""
Physics regression tests for the Path A PBUF implementation.

These tests assert that the closure-based elastic formulation supplies the
missing late-time energy density while decoupling smoothly at high redshift.
"""

import numpy as np

from cosmos.pbuf.equations import E2_pbuf, omega_sigma_radfix, omega_sigma_total
from cosmos.pbuf.model import PBUF


def _params(**overrides):
    """
    Helper to assemble a minimal parameter dictionary for Path A helpers.
    """
    params = {
        "H0": 67.4,
        "Om0": 0.315,
        "Or0": 5.0e-5,
        "Ok0": 0.0,
        "alpha": 1.0,
        "Rmax": 0.75,
        "k_sat": 1.0,
        "eps0": 1.0,
        "n_alpha": 2.0,
        "n_eps": 1.0,
        "n_R": 0.0,
    }
    params.update(overrides)
    return params


def test_E2_matches_budget_sum():
    """
    E2 must equal the sum of the individual budget components.
    """
    params = _params()
    a_vals = np.array([1.0, 0.7, 0.4])
    elastic = omega_sigma_total(a_vals, params)
    expected = (
        params["Om0"] * a_vals**-3 +
        params["Or0"] * a_vals**-4 +
        params["Ok0"] * a_vals**-2 +
        elastic
    )
    computed = E2_pbuf(a_vals, params)
    np.testing.assert_allclose(
        computed,
        expected,
        rtol=1e-9,
        err_msg="E2_pbuf must equal the explicit Friedmann budget including Ωσ.",
    )


def test_closure_from_parameter_sum():
    """
    Ωσ(a=1) equals 1 − Ωm − Ωr − Ωk by construction.
    """
    params = _params()
    omega_sigma_today = float(omega_sigma_total(1.0, params))
    closure = params["Om0"] + params["Or0"] + params["Ok0"] + omega_sigma_today
    np.testing.assert_allclose(closure, 1.0, rtol=1e-12)


def test_elastic_suppressed_at_high_z():
    """
    The elastic density should fall well below its z=0 value at early times.
    """
    params = _params()
    omega_today = float(omega_sigma_total(1.0, params))
    omega_high = float(omega_sigma_total(1.0 / (1.0 + 10.0), params))
    assert omega_high < 1e-2 * omega_today, (
        "Elastic sector should decouple for z ≫ z_turn."
    )


def test_radfix_zero():
    """The legacy radiation tweak is now an explicit no-op."""
    params = _params()
    vals = omega_sigma_radfix(np.array([1.0, 0.5, 0.1]), params)
    assert np.all(vals == 0.0)


def test_model_closure_today_matches_helper():
    """
    Model.closure_today() must match the helper-based computation.
    """
    model = PBUF(
        omega_m=0.315,
        h=0.674,
        alpha=1.0,
        Rmax=0.75,
        k_sat=1.0,
        eps0=1.0,
        omega_k=0.0,
        omega_r=5.0e-5,
    )
    omega_sigma_today = omega_sigma_total(1.0, model.params)
    manual = model.omega_m + model.omega_r + model.omega_k + float(omega_sigma_today)
    reported = model.closure_today()
    np.testing.assert_allclose(reported, manual, rtol=1e-12)
