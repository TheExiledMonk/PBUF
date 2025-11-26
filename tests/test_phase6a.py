from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.models.common.phase6a import Phase6aConfig
from cosmos.models.pbuf.sanity import phase6a_elastic_curvature, phase6a_early_sanity


def _build_log_grid(a_min: float, a_max: float, n_points: int) -> np.ndarray:
    return np.logspace(np.log10(a_min), np.log10(a_max), n_points)


def _build_H_from_f(a_grid: np.ndarray, f_values: np.ndarray) -> np.ndarray:
    ln_a = np.log(a_grid)
    lnH = np.empty_like(ln_a)
    lnH[0] = 0.0
    for idx in range(a_grid.size - 1):
        step = ln_a[idx + 1] - ln_a[idx]
        lnH[idx + 1] = lnH[idx] + 0.5 * (f_values[idx] + f_values[idx + 1]) * step
    return np.exp(lnH)


def test_phase6a_early_sanity_radiation_passes() -> None:
    config = Phase6aConfig(a_min_global=1e-8, a_min_curv=1e-3, n_early_points=64, eps_early_max=1e-3)
    a_grid = _build_log_grid(config.a_min_global, config.a_min_curv, config.n_early_points)
    H_values = a_grid ** -2.0
    omega_sigma = np.zeros_like(a_grid)

    ok, reason = phase6a_early_sanity(a_grid, H_values, omega_sigma, config)
    assert ok
    assert reason is None


def test_phase6a_early_sanity_matter_passes() -> None:
    config = Phase6aConfig(a_min_global=1e-8, a_min_curv=5e-4, n_early_points=64, eps_early_max=1e-3)
    a_grid = _build_log_grid(config.a_min_global, config.a_min_curv, config.n_early_points)
    H_values = a_grid ** -1.5
    omega_sigma = np.zeros_like(a_grid)

    ok, reason = phase6a_early_sanity(a_grid, H_values, omega_sigma, config)
    assert ok
    assert reason is None


def test_phase6a_elastic_curvature_smooth_transition_passes() -> None:
    config = Phase6aConfig(
        a_min_curv=1e-3,
        n_elastic_points=200,
        df_max=0.6,
        f_el_min=0.01,
        f_el_max=0.3,
    )
    a_grid = _build_log_grid(config.a_min_curv, 1.0, config.n_elastic_points)
    f_values = -2.0 + 0.1 * np.sin(np.linspace(0.0, np.pi, a_grid.size))
    H_values = _build_H_from_f(a_grid, f_values)
    omega_sigma = np.full_like(a_grid, 0.2)
    omega_total = np.ones_like(a_grid)

    ok, reason, stats = phase6a_elastic_curvature(
        a_grid,
        H_values,
        omega_sigma,
        omega_total,
        config,
    )
    assert ok
    assert reason is None
    assert stats.max_weighted_abs_df <= config.df_max


def test_phase6a_elastic_curvature_sharp_kink_fails() -> None:
    config = Phase6aConfig(
        a_min_curv=1e-3,
        n_elastic_points=200,
        df_max=0.05,
        f_el_min=0.01,
        f_el_max=0.3,
    )
    a_grid = _build_log_grid(config.a_min_curv, 1.0, config.n_elastic_points)
    f_values = np.full(a_grid.size, -2.0)
    kink_index = a_grid.size // 2
    f_values[kink_index:] = -8.0
    H_values = _build_H_from_f(a_grid, f_values)
    omega_sigma = np.full_like(a_grid, 0.2)
    omega_total = np.ones_like(a_grid)

    ok, reason, stats = phase6a_elastic_curvature(
        a_grid,
        H_values,
        omega_sigma,
        omega_total,
        config,
    )
    assert not ok
    assert reason is not None and "df/dln a" in reason
    assert stats.max_weighted_abs_df > config.df_max
