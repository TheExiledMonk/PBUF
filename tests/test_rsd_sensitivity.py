import os
from contextlib import contextmanager

import numpy as np

from cosmos.optim.dataset_evaluators import build_model
from cosmos.optim.parameter_defaults import (
    PBUF_PARAMETER_DEFAULTS,
    SIGMA8_PLANCK,
)
from cosmos.fits.rsd.chi2 import chi_squared_rsd


@contextmanager
def disable_phase6a():
    """
    Ensure Phase 6a guardrails do not interfere with targeted sensitivity tests.
    """
    original = os.environ.get("PBUF_DISABLE_PHASE6A")
    os.environ["PBUF_DISABLE_PHASE6A"] = "1"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("PBUF_DISABLE_PHASE6A", None)
        else:
            os.environ["PBUF_DISABLE_PHASE6A"] = original


def _baseline_params():
    params = dict(PBUF_PARAMETER_DEFAULTS)
    params.setdefault("H0", 67.0)
    params.setdefault("Obh2", 0.02237)
    return params


def _rsd_chi2_for(params, sigma8_0=None):
    model = build_model("pbuf", params)
    chi2 = chi_squared_rsd(model, sigma8_0=sigma8_0)
    diagnostics = getattr(model, "diagnostics", {})
    return chi2, diagnostics.get("rsd", {})


def test_rsd_chi2_marginalizes_sigma8():
    """
    χ² should be invariant to explicit σ₈ overrides once A_rsd is marginalized.
    """
    with disable_phase6a():
        params = _baseline_params()
        nominal, diag_nominal = _rsd_chi2_for(params, sigma8_0=SIGMA8_PLANCK)
        low, diag_low = _rsd_chi2_for(params, sigma8_0=0.65)
        high, diag_high = _rsd_chi2_for(params, sigma8_0=0.95)

    assert np.isclose(
        nominal, low
    ), f"χ² should be invariant under σ₈ override; {nominal} vs {low}"
    assert np.isclose(
        nominal, high
    ), f"χ² should be invariant under σ₈ override; {nominal} vs {high}"

    for diag in (diag_nominal, diag_low, diag_high):
        assert "A_rsd" in diag, "Missing A_rsd diagnostic for RSD marginalization."
        assert "A_rsd_var" in diag, "Missing A_rsd variance diagnostic."
        assert np.isfinite(diag["A_rsd"]), "A_rsd must be finite."
        assert diag["A_rsd_var"] > 0.0, "A_rsd variance must be positive."

    assert np.isclose(
        diag_nominal["A_rsd"], diag_low["A_rsd"]
    ), "A_rsd diagnostic should be insensitive to σ₈ overrides."
    assert np.isclose(
        diag_nominal["A_rsd"], diag_high["A_rsd"]
    ), "A_rsd diagnostic should be insensitive to σ₈ overrides."


def test_rsd_chi2_moves_with_elastic_knob():
    """
    Ensure elastic-sector parameters influence the RSD χ².
    """
    with disable_phase6a():
        params = _baseline_params()
        baseline, _ = _rsd_chi2_for(params)

        params_k_sat = dict(params)
        params_k_sat["k_sat"] = params_k_sat["k_sat"] * 0.7
        low_sat, _ = _rsd_chi2_for(params_k_sat)

        params_alpha = dict(params)
        params_alpha["alpha"] = params_alpha["alpha"] * 3.0
        high_alpha, _ = _rsd_chi2_for(params_alpha)

    delta_k = abs(baseline - low_sat)
    delta_alpha = abs(baseline - high_alpha)

    assert (
        delta_k > 1e-12
    ), f"χ² changed by only {delta_k} after reducing k_sat by 30%."
    assert (
        delta_alpha > 1e-12
    ), f"χ² changed by only {delta_alpha} after tripling alpha."
