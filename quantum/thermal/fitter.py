"""
Runtime derivation of thermal exponential parameters (β, T*, p) from microphysics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from quantum.core.constants import FIELD_CONTENT_DEGREES, REGULATOR_COEFFICIENTS
from quantum.thermal.table import _g_star as quantum_g_star
from quantum.thermal.table import _g_star_s as quantum_g_star_s
from quantum.thermal.table import T_CMB0

DEFAULT_FIT_MIN = 1.0e3
DEFAULT_FIT_MAX = 1.0e9
DEFAULT_FIT_POINTS = 48
DEFAULT_FIT_SAMPLES = 128


@dataclass(frozen=True)
class MicrophysicsInputs:
    eps0_today: float
    alpha_qm: float
    regulator: str
    field_content: str
    f_coup: float | None
    mixing_strength: float | None


@dataclass(frozen=True)
class ThermalFitResult:
    beta: float
    t_star: float
    power: float
    rms_log_error: float


def _coupling_fraction(g: float) -> float:
    return (g * g) / (1.0 + g * g)


def build_temperature_grid(t_min: float, t_max: float, num_points: int) -> np.ndarray:
    if t_min <= 0.0 or t_max <= 0.0:
        raise ValueError("Temperature bounds must be positive.")
    if t_min >= t_max:
        raise ValueError("t_min must be smaller than t_max.")
    if num_points < 4:
        raise ValueError("At least 4 temperature samples are required.")
    return np.logspace(math.log10(t_min), math.log10(t_max), num=num_points)


def _effective_neff(temps: np.ndarray, base_neff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g_star_vals = np.array([quantum_g_star(float(temp)) for temp in temps], dtype=float)
    g_starS_vals = np.array([quantum_g_star_s(float(temp)) for temp in temps], dtype=float)
    g_ref = quantum_g_star_s(float(np.max(temps)))
    scale = base_neff / max(g_ref, 1.0e-6)
    return g_star_vals, g_starS_vals, g_starS_vals * scale


def _compute_quantum_epsilon_curve(
    temps: np.ndarray,
    micro: MicrophysicsInputs,
) -> np.ndarray:
    if micro.regulator not in REGULATOR_COEFFICIENTS:
        raise ValueError(f"Unknown regulator '{micro.regulator}'.")
    if micro.field_content not in FIELD_CONTENT_DEGREES:
        raise ValueError(f"Unknown field_content '{micro.field_content}'.")

    loop_coeff = REGULATOR_COEFFICIENTS[micro.regulator]
    base_neff = FIELD_CONTENT_DEGREES[micro.field_content]
    _, _, n_eff = _effective_neff(temps, base_neff)

    f_coup = micro.f_coup
    if f_coup is None and micro.mixing_strength is not None:
        f_coup = _coupling_fraction(micro.mixing_strength)
    if f_coup is None:
        raise ValueError("Missing coupling fraction (f_coup) and mixing_strength.")

    f_cut_vals = np.sqrt(1.0 / (loop_coeff * np.maximum(n_eff, 1.0e-12)))
    eps_raw = f_coup * np.power(f_cut_vals, 4) / max(micro.alpha_qm, 1.0e-12)

    idx_today = int(np.argmin(np.abs(temps - T_CMB0)))
    norm = micro.eps0_today / max(eps_raw[idx_today], 1.0e-12)
    epsilon = eps_raw * norm

    return epsilon


def _fit_exponential_params(
    temps: np.ndarray,
    eps: np.ndarray,
    *,
    fit_min: float,
    fit_max: float,
    sample_points: int,
) -> ThermalFitResult:
    t_low = fit_min or float(np.min(temps))
    t_high = fit_max or float(np.max(temps))
    if t_low <= 0.0 or t_high <= 0.0 or t_low >= t_high:
        raise ValueError("Invalid fit range for temperature.")
    if sample_points < 4:
        raise ValueError("sample_points must be >= 4")

    target_temps = np.logspace(math.log10(t_low), math.log10(t_high), num=sample_points)
    log_eps = np.log(np.clip(eps, 1.0e-60, None))
    log_source_t = np.log(temps)
    sample_eps = np.exp(np.interp(np.log(target_temps), log_source_t, log_eps))
    lower, upper = 1.0e-6, 0.98
    mask = (sample_eps > lower) & (sample_eps < upper)
    if np.count_nonzero(mask) < 4:
        mask = sample_eps > lower
    if np.count_nonzero(mask) < 4:
        mask = sample_eps > 0.0
    target_temps = target_temps[mask]
    sample_eps = sample_eps[mask]

    log_sample = np.log(np.clip(sample_eps, 1.0e-60, None))
    y_for_p = np.log(-log_sample)
    p_est, _ = np.polyfit(np.log(target_temps), y_for_p, 1)
    power = float(np.clip(p_est, 0.1, 5.0))

    def _residual(params: np.ndarray) -> np.ndarray:
        beta_val, t_star_val = params
        return (-beta_val * np.power(target_temps / t_star_val, power)) - log_sample

    beta0 = float(np.clip(np.median(-log_sample), 0.05, 2.0))
    t_star0 = float(np.median(target_temps)) if target_temps.size else math.sqrt(t_low * t_high)
    bounds_low = np.array([0.0, t_low * 0.1], dtype=float)
    bounds_high = np.array([5.0, t_high * 10.0], dtype=float)
    result = least_squares(
        _residual,
        np.array([beta0, t_star0], dtype=float),
        bounds=(bounds_low, bounds_high),
        max_nfev=20000,
    )
    beta, t_star = [float(val) for val in result.x]
    fitted_log = -beta * np.power(target_temps / t_star, power)
    rms = float(np.sqrt(np.mean(np.square(log_sample - fitted_log))))

    return ThermalFitResult(beta=beta, t_star=t_star, power=power, rms_log_error=rms)


def derive_thermal_params(
    micro: MicrophysicsInputs,
    *,
    t_min: float,
    t_max: float,
    samples: int = DEFAULT_FIT_SAMPLES,
    fit_min: float = DEFAULT_FIT_MIN,
    fit_max: float = DEFAULT_FIT_MAX,
    fit_points: int = DEFAULT_FIT_POINTS,
) -> ThermalFitResult:
    temps = build_temperature_grid(t_min, t_max, samples)
    eps = _compute_quantum_epsilon_curve(temps, micro)
    return _fit_exponential_params(temps, eps, fit_min=fit_min, fit_max=fit_max, sample_points=fit_points)


__all__ = [
    "MicrophysicsInputs",
    "ThermalFitResult",
    "derive_thermal_params",
    "DEFAULT_FIT_MIN",
    "DEFAULT_FIT_MAX",
    "DEFAULT_FIT_POINTS",
    "DEFAULT_FIT_SAMPLES",
]
