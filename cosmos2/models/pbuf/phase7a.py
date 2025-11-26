"""Phase-7a sanity guards for the PBUF cosmology (ported from cosmos_old)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, TYPE_CHECKING

import numpy as np

from .distances import H as pbuf_H_of_a, omega_total_at_a
from .elastic import omega_sigma_of_a
from .sanity_base import SanityResult
from .thermal_table import ThermalTable

if TYPE_CHECKING:
    from .model import PBUFModel  # pragma: no cover
else:
    PBUFModel = Any

ParamDict = Dict[str, float]
SanityFn = Callable[[ParamDict], Tuple[bool, str | None]]


@dataclass(frozen=True)
class Phase7aConfig:
    """Thresholds for the Phase-7a sanity suite."""

    alpha_max_abs: float = 0.1
    alpha_step_max: float = 0.01
    epsilon0_max: float = 2.0
    epsilon0_step_max: float = 0.02
    k_sat_step_max: float = 0.02
    Rmax_min: float = 1.0e5
    Rmax_max: float = 1.0e10
    Rmax_step_factor: float = 3.0

    df_max: float = 2.0
    curv_ratio_max: float = 5.0e6
    curv_ratio_fraction: float = 0.1

    a_min: float = 1.0e-9
    a_max: float = 1.0
    n_a: int = 500

    early_lcdm_tol: float = 1.0e-3
    closure_tol: float = 1.0e-5
    alpha_deriv_max: float = 5.0
    epsilon_deriv_max: float = 3.0
    H_monotonic_rel_tol: float = 0.0


def load_phase7a_config(model_name: str) -> Phase7aConfig:
    """Load the Phase-7a config for the supplied model."""

    path = Path("configs/phase7a") / f"{model_name.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase-7a config not found at {path}")
    payload = json.loads(path.read_text())
    return Phase7aConfig(
        alpha_max_abs=float(payload.get("alpha_max_abs", 0.1)),
        alpha_step_max=float(payload.get("alpha_step_max", 0.01)),
        epsilon0_max=float(payload.get("epsilon0_max", 2.0)),
        epsilon0_step_max=float(payload.get("epsilon0_step_max", 0.02)),
        k_sat_step_max=float(payload.get("k_sat_step_max", 0.02)),
        Rmax_min=float(payload.get("Rmax_min", 1.0e5)),
        Rmax_max=float(payload.get("Rmax_max", 1.0e10)),
        Rmax_step_factor=float(payload.get("Rmax_step_factor", 3.0)),
        df_max=float(payload.get("df_max", 2.0)),
        curv_ratio_max=float(payload.get("curv_ratio_max", 5.0e6)),
        curv_ratio_fraction=float(payload.get("curv_ratio_fraction", 0.1)),
        a_min=float(payload.get("a_min", 1.0e-9)),
        a_max=float(payload.get("a_max", 1.0)),
        n_a=int(payload.get("n_a", 500)),
        early_lcdm_tol=float(payload.get("early_lcdm_tol", 1.0e-3)),
        closure_tol=float(payload.get("closure_tol", 1.0e-5)),
        alpha_deriv_max=float(payload.get("alpha_deriv_max", 5.0)),
        epsilon_deriv_max=float(payload.get("epsilon_deriv_max", 3.0)),
        H_monotonic_rel_tol=float(payload.get("H_monotonic_rel_tol", 0.0)),
    )


_PBUF_PHASE7A_CONFIG = load_phase7a_config("pbuf")


def make_phase7a_checker(
    thermal_table: ThermalTable,
    metadata: Dict[str, Any] | None = None,
    model_factory: Callable[..., PBUFModel] | None = None,
) -> SanityFn:
    def checker(params: ParamDict) -> tuple[bool, str | None]:
        sanitized = {key: float(value) for key, value in params.items()}
        factory = model_factory or _default_pbuf_factory
        model = factory(
            thermal_table=thermal_table,
            thermal_metadata=metadata,
            **sanitized,
        )
        result = check_pbuf_phase7a_sanity(
            sanitized,
            model,
            lcdm_model_factory=_lcdm_factory,
        )
        if result.ok:
            return True, None
        return False, "; ".join(result.reasons)

    return checker


def _default_pbuf_factory(**kwargs: Any) -> PBUFModel:
    try:
        from .model import PBUFModel as _PBUFModel  # type: ignore
    except Exception as exc:  # pragma: no cover - missing until model lands
        raise RuntimeError("PBUFModel is not available for Phase-7a checks yet.") from exc
    return _PBUFModel(**kwargs)


def _lcdm_factory(**kwargs: float):  # pragma: no cover - thin wrapper
    from cosmos2.models.lcdm.model import LCDMModel

    return LCDMModel(**kwargs)


def check_pbuf_phase7a_sanity(
    params: ParamDict,
    model: PBUFModel,
    lcdm_model_factory: Callable[..., Any] | None = None,
) -> SanityResult:
    """Execute the Phase-7a sanity suite for PBUF."""

    result = SanityResult()
    config = _PBUF_PHASE7A_CONFIG
    table = getattr(model, "thermal_table", None)
    if table is None:
        result.add_error("Phase-7a: missing thermal table")
        return result

    _check_thermal_lut(result, config, params, table)
    a_grid, H_values, omega_sigma, omega_total = _build_global_grid(config, model, table)
    alpha_grid = _sample_table_field(table, "alpha_T", a_grid)
    eps_grid = _sample_table_field(table, "epsilon0_T", a_grid)
    k_sat_grid = eps_grid - alpha_grid
    dln_alpha_grid = _sample_table_field(table, "dln_alpha_dlnT", a_grid)
    dln_eps_grid = _sample_table_field(table, "dln_epsilon0_dlnT", a_grid)
    _check_derivatives(
        result,
        config,
        a_grid,
        alpha_grid,
        eps_grid,
        k_sat_grid,
        dln_alpha_grid,
        dln_eps_grid,
    )
    _check_hubble_grid(result, config, a_grid, H_values)
    _check_omega_constraints(result, config, params, a_grid, omega_sigma, omega_total, model)

    if lcdm_model_factory is not None:
        _check_early_time_lcdm(result, config, params, model, table, lcdm_model_factory)

    return result


def _check_thermal_lut(
    result: SanityResult,
    config: Phase7aConfig,
    params: ParamDict,
    table: ThermalTable,
) -> None:
    a_vals = table.a
    alpha = table.alpha
    eps = table.eps
    k_sat = eps - alpha

    for label, values in (("alpha", alpha), ("epsilon0", eps), ("k_sat", k_sat), ("a", a_vals)):
        if not np.all(np.isfinite(values)):
            idx = int(np.argmax(~np.isfinite(values)))
            result.add_error(
                f"Phase-7a thermal: {label} contains non-finite entry at idx {idx} (a≈{_fmt_a(a_vals[idx])})"
            )
            return
    Rmax_val = float(params.get("Rmax", table.metadata.get("Rmax", 0.0)))

    if not (config.Rmax_min <= Rmax_val <= config.Rmax_max):
        result.add_error(
            f"Phase-7a thermal: R_max {Rmax_val:.3g} outside [{config.Rmax_min:.3g}, {config.Rmax_max:.3g}]"
        )

    _check_range(
        result,
        values=alpha,
        mask=np.abs(alpha) > config.alpha_max_abs,
        value_name="alpha",
        limit=f"[-{config.alpha_max_abs:.3g}, {config.alpha_max_abs:.3g}]",
        a_vals=a_vals,
    )
    _check_range(
        result,
        values=eps,
        mask=(eps <= 0.0) | (eps > config.epsilon0_max),
        value_name="epsilon0",
        limit=f"(0, {config.epsilon0_max:.3g}]",
        a_vals=a_vals,
    )
    _check_range(
        result,
        values=k_sat,
        mask=(k_sat < 0.0) | (k_sat > 1.0),
        value_name="k_sat",
        limit="[0.0, 1.0]",
        a_vals=a_vals,
    )
    _check_derivatives(
        result,
        config,
        a_vals,
        alpha,
        eps,
        k_sat,
        table.dln_alpha,
        table.dln_eps,
    )

    if a_vals.size >= 2:
        ratios = np.ones(a_vals.size - 1)
        if np.any(ratios >= config.Rmax_step_factor):
            idx = int(np.argmax(ratios >= config.Rmax_step_factor))
            result.add_error(
                f"Phase-7a thermal: R_max ratio at idx {idx} (a≈{_fmt_a(a_vals[idx])}) "
                f"= {ratios[idx]:.3g} >= {config.Rmax_step_factor:.3g}"
            )

    _check_smoothness(result, config, a_vals, alpha, eps, k_sat)


def _check_range(
    result: SanityResult,
    *,
    values: np.ndarray,
    mask: np.ndarray,
    value_name: str,
    limit: str,
    a_vals: np.ndarray,
) -> None:
    if not mask.any():
        return
    idx = int(np.argmax(mask))
    result.add_error(
        f"Phase-7a thermal: {value_name}[{idx}] (a≈{_fmt_a(a_vals[idx])})={values[idx]:.3g} outside {limit}"
    )


def _check_smoothness(
    result: SanityResult,
    config: Phase7aConfig,
    a_vals: np.ndarray,
    alpha: np.ndarray,
    eps: np.ndarray,
    k_sat: np.ndarray,
) -> None:
    if a_vals.size < 2:
        return

    _check_step(result, config.alpha_step_max, alpha, "alpha", a_vals)
    _check_step(result, config.epsilon0_step_max, eps, "epsilon0", a_vals)
    _check_step(result, config.k_sat_step_max, k_sat, "k_sat", a_vals)


def _check_step(
    result: SanityResult,
    limit: float,
    values: np.ndarray,
    name: str,
    a_vals: np.ndarray,
) -> None:
    diffs = np.abs(np.diff(values))
    if diffs.size == 0:
        return
    mask = diffs >= limit
    if not mask.any():
        return
    idx = int(np.argmax(mask))
    value = diffs[idx]
    result.add_error(
        f"Phase-7a thermal: Δ{name}[{idx}->{idx+1}] between a≈{_fmt_a(a_vals[idx])} "
        f"and a≈{_fmt_a(a_vals[idx+1])} is {value:.3g} ≥ {limit:.3g}"
    )


def _check_derivatives(
    result: SanityResult,
    config: Phase7aConfig,
    a_vals: np.ndarray,
    alpha: np.ndarray,
    eps: np.ndarray,
    k_sat: np.ndarray,
    dln_alpha: np.ndarray,
    dln_eps: np.ndarray,
) -> None:
    if not np.all(np.isfinite(dln_alpha)):
        idx = int(np.argmax(~np.isfinite(dln_alpha)))
        result.add_error(
            f"Phase-7a thermal: dln alpha/dln T not finite at idx {idx} (a≈{_fmt_a(a_vals[idx])})"
        )
        return
    if not np.all(np.isfinite(dln_eps)):
        idx = int(np.argmax(~np.isfinite(dln_eps)))
        result.add_error(
            f"Phase-7a thermal: dln epsilon/dln T not finite at idx {idx} (a≈{_fmt_a(a_vals[idx])})"
        )
        return

    mask_alpha = np.abs(dln_alpha) >= config.alpha_deriv_max
    if mask_alpha.any():
        idx = int(np.argmax(mask_alpha))
        result.add_error(
            f"Phase-7a thermal: |dln alpha/dln T|={np.abs(dln_alpha[idx]):.3g} ≥ {config.alpha_deriv_max:.3g} at idx {idx} (a≈{_fmt_a(a_vals[idx])})"
        )
        return

    mask_epsilon = np.abs(dln_eps) >= config.epsilon_deriv_max
    if mask_epsilon.any():
        idx = int(np.argmax(mask_epsilon))
        result.add_error(
            f"Phase-7a thermal: |dln epsilon/dln T|={np.abs(dln_eps[idx]):.3g} ≥ {config.epsilon_deriv_max:.3g} at idx {idx} (a≈{_fmt_a(a_vals[idx])})"
        )
        return

    dln_k_sat = np.full_like(k_sat, np.nan)
    positive_mask = k_sat > 0.0
    if positive_mask.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = (eps[positive_mask] * dln_eps[positive_mask]) - (
                alpha[positive_mask] * dln_alpha[positive_mask]
            )
            dln_k_sat[positive_mask] = numerator / k_sat[positive_mask]
        bad_mask = ~np.isfinite(dln_k_sat[positive_mask])
        if bad_mask.any():
            idxs = np.nonzero(positive_mask)[0]
            bad_idx = int(idxs[np.argmax(bad_mask)])
            result.add_error(
                f"Phase-7a thermal: dln k_sat/dln T not finite at idx {bad_idx} (a≈{_fmt_a(a_vals[bad_idx])})"
            )
            return


def _build_global_grid(
    config: Phase7aConfig,
    model: PBUFModel,
    table: ThermalTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_grid = np.logspace(np.log10(config.a_min), np.log10(config.a_max), config.n_a)
    params = getattr(model, "params", None) or getattr(model, "_params", None)
    H_values = np.array([pbuf_H_of_a(a, params, table) for a in a_grid], dtype=float)
    omega_sigma = np.array([omega_sigma_of_a(a, params, table) for a in a_grid], dtype=float)
    alpha_value = getattr(model, "alpha", 0.0)
    omega_total = np.array(
        [omega_total_at_a(a, params, table, alpha=alpha_value) for a in a_grid],
        dtype=float,
    )
    return a_grid, H_values, omega_sigma, omega_total


def _check_hubble_grid(
    result: SanityResult,
    config: Phase7aConfig,
    a_grid: np.ndarray,
    H_values: np.ndarray,
) -> None:
    if a_grid.size < 2:
        return

    if np.any(H_values <= 0.0):
        idx = int(np.argmin(H_values))
        result.add_error(
            f"Phase-7a H: non-positive H(a)={H_values[idx]:.3g} at idx {idx} (a≈{_fmt_a(a_grid[idx])})"
        )
        return

    tol = max(0.0, config.H_monotonic_rel_tol)
    rel_mask = a_grid[:-1] < 0.99
    if tol > 0.0:
        increases = H_values[1:] > H_values[:-1] * (1.0 + tol)
    else:
        increases = np.diff(H_values) > 0.0
    mask_segments = rel_mask if tol > 0.0 else np.ones_like(rel_mask, dtype=bool)
    bad_monotonic = increases & mask_segments
    if bad_monotonic.any():
        idx = int(np.argmax(bad_monotonic))
        reason = (
            f"Phase-7a H: H(a) increases too much between idx {idx} and {idx+1} "
            f"(a≈{_fmt_a(a_grid[idx])}->{_fmt_a(a_grid[idx+1])}), ΔH={(H_values[idx + 1] - H_values[idx]):.3g}"
        )
        if tol > 0.0:
            reason += f" (tol={tol:.3g} rel)"
        result.add_error(reason)
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        lnH = np.log(H_values)
        lna = np.log(a_grid)
        f_grid = np.gradient(lnH, lna)
    df = np.diff(f_grid)
    mask_df = np.abs(df) > config.df_max
    if mask_df.any():
        idx = int(np.argmax(mask_df))
        result.add_error(
            f"Phase-7a H: Δ(d ln H/d ln a) too large at idx {idx+1} (a≈{_fmt_a(a_grid[idx+1])}), |Δf|={np.abs(df[idx]):.3g} > {config.df_max:.3g}"
        )
        return

    H_prime = np.gradient(H_values, a_grid)
    H_double = np.gradient(H_prime, a_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = H_double / H_prime
    valid = np.isfinite(ratio) & np.isfinite(H_prime) & (np.abs(H_prime) > 0.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return
    bad = valid & (np.abs(ratio) > config.curv_ratio_max)
    bad_count = int(np.count_nonzero(bad))
    bad_fraction = bad_count / float(valid_count)
    if bad_fraction > config.curv_ratio_fraction:
        idx = int(np.argmax(bad))
        result.add_error(
            f"Phase-7a H: |H''/H'|={np.abs(ratio[idx]):.3g} at idx {idx} (a≈{_fmt_a(a_grid[idx])}) "
            f"above {config.curv_ratio_max:.3g} on {bad_fraction:.1%} of valid grid"
        )


def _check_omega_constraints(
    result: SanityResult,
    config: Phase7aConfig,
    params: ParamDict,
    a_grid: np.ndarray,
    omega_sigma: np.ndarray,
    omega_total: np.ndarray,
    model: PBUFModel | None = None,
) -> None:
    if omega_sigma.size == 0:
        return

    mask_negative = omega_sigma < -1e-12
    if mask_negative.any():
        idx = int(np.argmax(mask_negative))
        result.add_error(
            f"Phase-7a Omega: Ωσ<0 at idx {idx} (a≈{_fmt_a(a_grid[idx])}), Ωσ={omega_sigma[idx]:.3g}"
        )
        return

    mask_total = omega_sigma > omega_total + 1e-6
    if mask_total.any():
        idx = int(np.argmax(mask_total))
        result.add_error(
            f"Phase-7a Omega: Ωσ>Ω_total at idx {idx} (a≈{_fmt_a(a_grid[idx])}), Ωσ={omega_sigma[idx]:.3g}, Ω_tot={omega_total[idx]:.3g}"
        )
        return

    idx_one = int(np.argmax(a_grid >= 1.0))
    omega_sigma_1 = float(omega_sigma[idx_one])
    alpha_value = getattr(model, "alpha", 0.0)
    omega_sum = (
        float(params.get("Omega_m0", 0.0))
        + float(params.get("Omega_b0", 0.0))
        + float(params.get("Omega_r0", 0.0))
        + float(alpha_value)
        + omega_sigma_1
    )
    if abs(omega_sum - 1.0) > config.closure_tol:
        result.add_error(
            f"Phase-7a closure: sum of Ω at a≈1 is {omega_sum:.3g} (|Δ|>{config.closure_tol:.3g})"
        )


def _check_early_time_lcdm(
    result: SanityResult,
    config: Phase7aConfig,
    params: ParamDict,
    model: PBUFModel,
    table: ThermalTable,
    lcdm_model_factory: Callable[..., Any],
) -> None:
    a_grid = np.logspace(-9, -6, 40)
    params_obj = getattr(model, "params", None) or getattr(model, "_params", None)
    H_pbuf = np.array([pbuf_H_of_a(a, params_obj, table) for a in a_grid], dtype=float)

    H0 = float(params.get("H0", 0.0))
    om = float(params.get("Omega_m0", 0.0))
    ob = float(params.get("Omega_b0", 0.0))
    orad = float(params.get("Omega_r0", 0.0))
    alpha_curv = getattr(model, "alpha", 0.0)
    ol = 1.0 - (om + ob + orad + alpha_curv)
    ok_term = alpha_curv
    H_lcdm = np.array(
        [
            H0
            * math.sqrt(
                max(
                    om / a**3
                    + ob / a**3
                    + orad / a**4
                    + ok_term / a**2
                    + ol,
                    0.0,
                )
            )
            for a in a_grid
        ],
        dtype=float,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = H_pbuf / H_lcdm
    mask = np.abs(ratio - 1.0) > config.early_lcdm_tol
    if mask.any():
        idx = int(np.argmax(mask))
        result.add_error(
            f"Phase-7a early LCDM: H_PBUF/H_LCDM={ratio[idx]:.6g} at idx {idx} (a≈{_fmt_a(a_grid[idx])}) deviates >{config.early_lcdm_tol:.3g}"
        )


def _fmt_a(value: float) -> str:
    return f"{value:.3e}"


def _sample_table_field(table: ThermalTable, field: str, a_vals: np.ndarray) -> np.ndarray:
    try:
        return np.array([table.fast_get(field, at_scale_factor=a) for a in a_vals], dtype=float)
    except Exception:
        return np.array([table.get(field, at_scale_factor=a) for a in a_vals], dtype=float)


__all__ = [
    "Phase7aConfig",
    "check_pbuf_phase7a_sanity",
    "make_phase7a_checker",
    "load_phase7a_config",
]
