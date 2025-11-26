"""Glue layer that exposes the Phase-7a sanity suite for PBUF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.model import PBUFModel
from cosmos.models.common.phase6a import Phase6aConfig
from cosmos.models.pbuf.phase7a import check_pbuf_phase7a_sanity
from cosmos.optim.sanity_base import SanityResult

ModelParams = Dict[str, float]


@dataclass
class Phase6aElasticStats:
    f: np.ndarray
    df_dln_a: np.ndarray
    weighted_abs_df: np.ndarray
    max_weighted_abs_df: float


def _compute_f(a_grid: np.ndarray, H_values: np.ndarray) -> np.ndarray:
    ln_a = np.log(a_grid)
    ln_H = np.log(H_values)
    return np.gradient(ln_H, ln_a)


def phase6a_early_sanity(
    a_grid: np.ndarray,
    H_values: np.ndarray,
    omega_sigma: np.ndarray,
    config: Phase6aConfig,
) -> Tuple[bool, str | None]:
    """Check early-time H(a) behavior (radiation/matter regimes)."""

    if a_grid.size < 3:
        return False, "Insufficient early-time samples"

    # Monotonicity: H should decrease with a (allow tiny numerical noise).
    if np.any(np.diff(H_values) > config.early_monotonic_tol):
        return False, "H(a) is not monotonically decreasing"

    f_vals = _compute_f(a_grid, H_values)
    targets = (-2.0, -1.5)  # radiation or matter slope
    deviations = []
    for target in targets:
        denom = max(abs(target), 1e-9)
        deviations.append(float(np.mean(np.abs((f_vals - target) / denom))))

    min_dev = min(deviations)
    if min_dev > config.eps_early_max:
        return False, f"Early-time slope deviates from radiation/matter (Δ={min_dev:.3g})"

    return True, None


def phase6a_elastic_curvature(
    a_grid: np.ndarray,
    H_values: np.ndarray,
    omega_sigma: np.ndarray,
    omega_total: np.ndarray,
    config: Phase6aConfig,
) -> Tuple[bool, str | None, Phase6aElasticStats]:
    """
    Probe elasticity curvature via df/dln a weighted by the elastic energy fraction.

    The weight uses omega_sigma / omega_total to mirror the original elastic gate.
    """

    f_vals = _compute_f(a_grid, H_values)
    ln_a = np.log(a_grid)
    df_dln_a = np.gradient(f_vals, ln_a)

    weight = omega_sigma / np.maximum(omega_total, 1e-12)
    weighted_abs_df = np.abs(df_dln_a) * weight
    max_weighted_abs_df = float(np.max(weighted_abs_df)) if weighted_abs_df.size else 0.0

    stats = Phase6aElasticStats(
        f=f_vals,
        df_dln_a=df_dln_a,
        weighted_abs_df=weighted_abs_df,
        max_weighted_abs_df=max_weighted_abs_df,
    )

    if max_weighted_abs_df > config.df_max:
        return False, f"|df/dln a| exceeds cap ({max_weighted_abs_df:.3g} > {config.df_max})", stats

    return True, None, stats


def check_pbuf_sanity(
    params: ModelParams,
    model: PBUFModel,
    lcdm_model_factory: Callable[..., LCDMModel] | None = None,
    *,
    use_thermal_table: bool = True,
) -> SanityResult:
    """Run the Phase-7a sanity suite for the supplied PBUF configuration."""

    return check_pbuf_phase7a_sanity(params, model, lcdm_model_factory=lcdm_model_factory)
