import numpy as np
import pytest
from pathlib import Path

from tools.test_quantum_thermal_bridge import (
    EpsilonCurve,
    build_temperature_grid,
    compute_quantum_epsilon_curve,
    fit_exponential_params,
    load_microphysics,
    validate_curve,
    _exp_model,
)


def test_fit_exponential_recovers_params():
    temps = np.logspace(0, 6, 80)
    beta_true, t_star_true, power_true = 0.4, 2.0e3, 1.6
    eps = _exp_model(temps, beta_true, t_star_true, power_true)
    fit = fit_exponential_params(temps, eps, fit_min=1.0, fit_max=1.0e6, sample_points=60)
    assert fit.beta == pytest.approx(beta_true, rel=0.35)
    assert fit.t_star == pytest.approx(t_star_true, rel=0.35)
    assert fit.power == pytest.approx(power_true, rel=0.05)
    assert fit.rms_log_error < 0.05


def test_validate_curve_rejects_non_positive():
    temps = np.logspace(0, 2, 16)
    epsilon = np.linspace(1.0, 0.0, temps.size)
    dummy = np.zeros_like(temps)
    curve = EpsilonCurve(
        temps=temps,
        epsilon=epsilon,
        alpha=dummy,
        dln_epsilon=dummy,
        dln_alpha=dummy,
        g_star=dummy,
        g_starS=dummy,
        f_cut=dummy,
    )
    ok, reason = validate_curve(curve)
    assert not ok
    assert "epsilon0 <= 0" in (reason or "")


@pytest.mark.skipif(not Path("configs/quantum/micro_cache.json").exists(), reason="micro cache missing")
def test_quantum_curve_smoke_uses_cache():
    micro = load_microphysics(Path("configs/quantum/micro_cache.json"))
    temps = build_temperature_grid(2.7255, 1.0e6, 32)
    curve = compute_quantum_epsilon_curve(temps, micro)
    ok, reason = validate_curve(curve)
    assert ok, reason
    idx_today = int(np.argmin(np.abs(temps - 2.7255)))
    assert curve.epsilon[idx_today] == pytest.approx(micro.eps0_today, rel=1.0e-3)
