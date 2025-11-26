#!/usr/bin/env python3
"""
Single-script Quantum thermal bridge that derives ε₀(T) from the micro kernel,
fits the exponential surrogate, checks the Phase-7a island, and can optionally
run the thermal sanity suite.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from quantum.api import run_quantum_engine
from quantum.core.constants import FIELD_CONTENT_DEGREES, REGULATOR_COEFFICIENTS
from quantum.thermal.table import (
    ThermalModelConfig,
    ThermalTableSpec,
    generate_thermal_table,
    _g_star as quantum_g_star,
    _g_star_s as quantum_g_star_s,
)
from tools import thermal_validator

T_CMB0 = 2.7255

DEFAULT_VALIDATOR_BOUNDS = {
    "beta": (0.0, 1.0),
    "T_star": (1.0e3, 1.0e7),
    "power": (0.5, 4.0),
}


@dataclass(frozen=True)
class MicrophysicsSnapshot:
    regulator: str
    field_content: str
    alpha_qm: float
    eps0_today: float
    f_cut: float | None
    f_coup: float | None
    mixing_strength: float | None
    t_star: float | None
    power_index: float | None


@dataclass(frozen=True)
class EpsilonCurve:
    temps: np.ndarray
    epsilon: np.ndarray
    alpha: np.ndarray
    dln_epsilon: np.ndarray
    dln_alpha: np.ndarray
    g_star: np.ndarray
    g_starS: np.ndarray
    f_cut: np.ndarray


@dataclass(frozen=True)
class FitResult:
    beta: float
    t_star: float
    power: float
    rms_log_error: float
    sample_temps: np.ndarray
    sample_eps: np.ndarray


def _load_json(path: Path) -> MutableMapping[str, object]:
    return json.loads(path.read_text())


def load_microphysics(micro_path: Path | None = None) -> MicrophysicsSnapshot:
    """
    Load cached Quantum outputs or run the engine if no cache is available.
    """
    micro_payload: Mapping[str, object] | None = None
    if micro_path is not None and micro_path.exists():
        micro_payload = _load_json(micro_path)
    if micro_payload is None:
        micro_payload = run_quantum_engine()

    def _maybe(name: str, alt: str | None = None) -> float | None:
        val = micro_payload.get(name)
        if val is None and alt is not None:
            val = micro_payload.get(alt)
        return None if val is None else float(val)

    return MicrophysicsSnapshot(
        regulator=str(micro_payload.get("regulator", "hard_cutoff")),
        field_content=str(micro_payload.get("field_content", "SM_full")),
        alpha_qm=float(_maybe("alpha_qm", "alpha_QM") or 0.02),
        eps0_today=float(_maybe("eps0_base", "eps0") or 1.0),
        f_cut=_maybe("f_cut"),
        f_coup=_maybe("f_coup"),
        mixing_strength=_maybe("mixing_strength"),
        t_star=_maybe("T_star"),
        power_index=_maybe("power_index"),
    )


def _coupling_fraction(g: float) -> float:
    return (g * g) / (1.0 + g * g)


def build_temperature_grid(t_min: float, t_max: float, num_points: int) -> np.ndarray:
    if t_min <= 0.0 or t_max <= 0.0:
        raise ValueError("Temperature bounds must be positive.")
    if t_min >= t_max:
        raise ValueError("t_min must be smaller than t_max.")
    return np.logspace(math.log10(t_min), math.log10(t_max), num=num_points)


def _effective_neff(temps: np.ndarray, base_neff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g_star_vals = np.array([quantum_g_star(float(temp)) for temp in temps], dtype=float)
    g_starS_vals = np.array([quantum_g_star_s(float(temp)) for temp in temps], dtype=float)
    g_ref = quantum_g_star_s(float(np.max(temps)))
    scale = base_neff / max(g_ref, 1.0e-6)
    return g_star_vals, g_starS_vals, g_starS_vals * scale


def compute_quantum_epsilon_curve(
    temps: np.ndarray,
    micro: MicrophysicsSnapshot,
) -> EpsilonCurve:
    if micro.regulator not in REGULATOR_COEFFICIENTS:
        raise ValueError(f"Unknown regulator '{micro.regulator}'.")
    if micro.field_content not in FIELD_CONTENT_DEGREES:
        raise ValueError(f"Unknown field_content '{micro.field_content}'.")

    loop_coeff = REGULATOR_COEFFICIENTS[micro.regulator]
    base_neff = FIELD_CONTENT_DEGREES[micro.field_content]
    g_star, g_starS, n_eff = _effective_neff(temps, base_neff)

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
    alpha_vals = micro.alpha_qm * epsilon

    log_t = np.log(temps)
    log_eps = np.log(np.clip(epsilon, 1.0e-60, None))
    dln_eps = np.gradient(log_eps, log_t)
    dln_alpha = dln_eps.copy()

    return EpsilonCurve(
        temps=temps,
        epsilon=epsilon,
        alpha=alpha_vals,
        dln_epsilon=dln_eps,
        dln_alpha=dln_alpha,
        g_star=g_star,
        g_starS=g_starS,
        f_cut=f_cut_vals,
    )


def validate_curve(curve: EpsilonCurve) -> Tuple[bool, str | None]:
    eps = curve.epsilon
    if not np.all(np.isfinite(eps)):
        idx = int(np.argmax(~np.isfinite(eps)))
        return False, f"epsilon0 contains non-finite entry at idx {idx}"
    if np.any(eps <= 0.0):
        idx = int(np.argmax(eps <= 0.0))
        return False, f"epsilon0 <= 0 at idx {idx}"
    monotone = np.all(np.diff(eps) <= 1.0e-12)
    if not monotone:
        return False, "epsilon0(T) is not monotone decreasing in T"
    return True, None


def _exp_model(temp: np.ndarray, beta: float, t_star: float, power: float) -> np.ndarray:
    return np.exp(-beta * np.power(temp / t_star, power))


def fit_exponential_params(
    temps: np.ndarray,
    eps: np.ndarray,
    *,
    fit_min: float | None = None,
    fit_max: float | None = None,
    sample_points: int = 48,
) -> FitResult:
    if sample_points < 4:
        raise ValueError("sample_points must be >= 4")

    t_low = fit_min or float(np.min(temps))
    t_high = fit_max or float(np.max(temps))
    if t_low <= 0.0 or t_high <= 0.0 or t_low >= t_high:
        raise ValueError("Invalid fit range for temperature.")

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
    result = least_squares(_residual, np.array([beta0, t_star0], dtype=float), bounds=(bounds_low, bounds_high), max_nfev=20000)
    beta, t_star = [float(val) for val in result.x]
    fitted_log = -beta * np.power(target_temps / t_star, power)
    rms = float(np.sqrt(np.mean(np.square(log_sample - fitted_log))))

    return FitResult(
        beta=beta,
        t_star=t_star,
        power=power,
        rms_log_error=rms,
        sample_temps=target_temps,
        sample_eps=sample_eps,
    )


def _is_inside_island(params: FitResult, bounds: Mapping[str, Sequence[float]]) -> bool:
    beta_min, beta_max = bounds.get("beta", DEFAULT_VALIDATOR_BOUNDS["beta"])
    t_star_min, t_star_max = bounds.get("T_star", DEFAULT_VALIDATOR_BOUNDS["T_star"])
    p_min, p_max = bounds.get("power", DEFAULT_VALIDATOR_BOUNDS["power"])
    return (
        beta_min <= params.beta <= beta_max
        and t_star_min <= params.t_star <= t_star_max
        and p_min <= params.power <= p_max
    )


def _build_macro_table(
    fit: FitResult,
    micro: MicrophysicsSnapshot,
    t_min: float,
    t_max: float,
    num_points: int,
    dense_points: int = 24,
    *,
    table_version: int = 12,
    method_version: int = 12,
) -> Mapping[str, object]:
    cfg = ThermalModelConfig(
        mode="exp",
        beta=fit.beta,
        t_star=fit.t_star,
        power=fit.power,
        alpha_qm=micro.alpha_qm,
        eps_min=1.0e-4,
    )
    spec = ThermalTableSpec(
        model=cfg,
        t_min=t_min,
        t_max=t_max,
        num_points=num_points,
        dense_points=dense_points,
        table_version=int(table_version),
        method_version=int(method_version),
        regulator=micro.regulator,
        field_content=micro.field_content,
        f_cut_T=micro.f_cut or 1.0,
        f_coup_T=micro.f_coup or 1.0,
        notes="quantum thermal bridge fit",
    )
    table = generate_thermal_table(spec)
    payload = {"metadata": table.metadata, "rows": [asdict(row) for row in table.rows]}
    return payload


def _table_to_curves(payload: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    rows = payload.get("rows", [])
    temps = np.array([float(row["T_K"]) for row in rows], dtype=float)
    eps = np.array([float(row["epsilon0_T"]) for row in rows], dtype=float)
    alpha = np.array([float(row["alpha_T"]) for row in rows], dtype=float)
    dln_eps = np.array([float(row["dln_epsilon0_dlnT"]) for row in rows], dtype=float)
    dln_alpha = np.array([float(row["dln_alpha_dlnT"]) for row in rows], dtype=float)
    a_vals = np.array([float(row["a"]) for row in rows], dtype=float)
    return {
        "T": temps,
        "a": a_vals,
        "epsilon0": eps,
        "alpha": alpha,
        "dln_epsilon0_dlnT": dln_eps,
        "dln_alpha_dlnT": dln_alpha,
    }


def run_phase7a_check(table_payload: Mapping[str, object], thresholds_path: Path) -> Tuple[bool, List[str]]:
    curves = _table_to_curves(table_payload)
    thresholds = thermal_validator.load_phase7a_thresholds(thresholds_path)
    ok, reasons = thermal_validator.check_phase7a(curves, thresholds, metadata=table_payload.get("metadata"))
    return ok, reasons


def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def _format_bounds(bounds: Mapping[str, Sequence[float]]) -> str:
    beta_min, beta_max = bounds.get("beta", DEFAULT_VALIDATOR_BOUNDS["beta"])
    t_star_min, t_star_max = bounds.get("T_star", DEFAULT_VALIDATOR_BOUNDS["T_star"])
    p_min, p_max = bounds.get("power", DEFAULT_VALIDATOR_BOUNDS["power"])
    return f"β∈[{beta_min}, {beta_max}], T*∈[{t_star_min}, {t_star_max}] K, p∈[{p_min}, {p_max}]"


def _render_summary(
    micro: MicrophysicsSnapshot,
    curve: EpsilonCurve,
    fit: FitResult,
    island_ok: bool,
    island_bounds: Mapping[str, Sequence[float]],
    phase7a_ok: bool | None,
    phase7a_reasons: Sequence[str],
) -> None:
    _print_header("Quantum Thermal Test")
    print(f"Regulator:    {micro.regulator}")
    print(f"Field:        {micro.field_content}")
    print()
    print("Micro outputs:")
    print(f"  alpha_QM  = {micro.alpha_qm:.5g}")
    print(f"  epsilon0  = {micro.eps0_today:.5g}")
    print(f"  f_cut     = {micro.f_cut:.5g}" if micro.f_cut is not None else "  f_cut     = (derived)")
    print(f"  f_coup    = {micro.f_coup:.5g}" if micro.f_coup is not None else "  f_coup    = (derived)")
    print()
    ok_curve, reason = validate_curve(curve)
    status = "OK" if ok_curve else f"FAIL ({reason})"
    print(f"Quantum-derived epsilon0(T): {status}")
    print("Fitted exponential parameters:")
    print(f"  beta      = {fit.beta:.4g}")
    print(f"  T_star    = {fit.t_star:.4g} K")
    print(f"  power     = {fit.power:.4g}")
    print(f"  RMS[ln ε] = {fit.rms_log_error:.3g}")
    print()
    print(f"Validator island check: {'PASS' if island_ok else 'FAIL'}")
    print(f"  Bounds: { _format_bounds(island_bounds)}")
    print(f"  Inside: {'yes' if island_ok else 'no'}")
    print()
    if phase7a_ok is not None:
        print(f"Phase-7a thermal sanity: {'PASS' if phase7a_ok else 'FAIL'}")
        if phase7a_reasons:
            for reason_text in phase7a_reasons:
                print(f"    - {reason_text}")
        print()
    print("Summary:")
    print(f"  epsilon0_T: min={np.min(curve.epsilon):.3g}, max={np.max(curve.epsilon):.3g}")
    print(f"  alpha_T:    min={np.min(curve.alpha):.3g}, max={np.max(curve.alpha):.3g}")
    print(f"  max |dln(eps)/dlnT| = {np.max(np.abs(curve.dln_epsilon)):.3g}")
    print(f"  g_star range: [{np.min(curve.g_star):.3g}, {np.max(curve.g_star):.3g}]")
    print()
    verdict = "PASS" if ok_curve and island_ok and (phase7a_ok in (None, True)) else "FAIL"
    print(f"Overall: {verdict} (Quantum thermal bridge)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantum thermal bridge test harness")
    parser.add_argument("--config", type=Path, default=Path("configs/quantum/config.json"), help="JSON config with thermal grid + bounds")
    parser.add_argument("--micro-cache", type=Path, default=Path("configs/quantum/micro_cache.json"), help="Cached quantum microphysics JSON")
    parser.add_argument("--fit-min", type=float, default=1.0e3, help="Lower T bound (K) for the exponential fit")
    parser.add_argument("--fit-max", type=float, default=1.0e9, help="Upper T bound (K) for the exponential fit")
    parser.add_argument("--fit-points", type=int, default=48, help="Number of log-spaced samples used during fitting")
    parser.add_argument("--run-phase7a", action="store_true", help="Generate macro LUT and run Phase-7a thermal checks")
    parser.add_argument("--phase7a-thresholds", type=Path, default=Path("configs/phase7a/pbuf.json"), help="Path to Phase-7a thresholds JSON")
    return parser.parse_args()


def _load_script_config(path: Path) -> Mapping[str, object]:
    if path.exists():
        return _load_json(path)
    return {
        "regulator": "hard_cutoff",
        "field_content": "SM_full",
        "T_min": T_CMB0,
        "T_max": 1.0e12,
        "n_T": 128,
        "validator_bounds": dict(DEFAULT_VALIDATOR_BOUNDS),
    }


def main() -> None:
    args = _parse_args()
    config = _load_script_config(args.config)
    micro = load_microphysics(args.micro_cache)

    regulator = str(config.get("regulator", micro.regulator))
    field_content = str(config.get("field_content", micro.field_content))
    t_min = float(config.get("T_min", T_CMB0))
    t_max = float(config.get("T_max", 1.0e12))
    n_T = int(config.get("n_T", 128))
    validator_bounds = config.get("validator_bounds", DEFAULT_VALIDATOR_BOUNDS)

    temps = build_temperature_grid(t_min, t_max, n_T)
    micro = MicrophysicsSnapshot(
        regulator=regulator,
        field_content=field_content,
        alpha_qm=micro.alpha_qm,
        eps0_today=micro.eps0_today,
        f_cut=micro.f_cut,
        f_coup=micro.f_coup,
        mixing_strength=micro.mixing_strength,
        t_star=micro.t_star,
        power_index=micro.power_index,
    )

    curve = compute_quantum_epsilon_curve(temps, micro)
    fit = fit_exponential_params(curve.temps, curve.epsilon, fit_min=args.fit_min, fit_max=args.fit_max, sample_points=args.fit_points)
    island_ok = _is_inside_island(fit, validator_bounds)

    phase7a_ok: bool | None = None
    phase7a_reasons: List[str] = []
    if args.run_phase7a:
        macro_payload = _build_macro_table(
            fit,
            micro,
            t_min=t_min,
            t_max=t_max,
            num_points=n_T,
            dense_points=int(config.get("dense_points", 24)),
            table_version=int(config.get("table_version", 12)),
            method_version=int(config.get("method_version", 12)),
        )
        phase7a_ok, phase7a_reasons = run_phase7a_check(macro_payload, args.phase7a_thresholds)

    _render_summary(micro, curve, fit, island_ok, validator_bounds, phase7a_ok, phase7a_reasons)


if __name__ == "__main__":
    main()
