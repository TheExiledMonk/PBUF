"""Central parameter authority that builds consistent parameter snapshots for LCDM and PBUF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math
import numpy as np

from cosmos2.models.lcdm.utils import C_LIGHT as C_LIGHT_KM_S
from cosmos2.models.pbuf.thermal_table import ThermalTable

GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
MPC_TO_KM = 3.0856775814913673e19  # 1 Mpc in km
SECONDS_PER_GYR = 3.1536e16
_PLOT_POINTS = 64
_GROWTH_POINTS = 128
_AGE_GRID_MIN = 1.0e-4


@dataclass(frozen=True)
class ModelState:
    """Lightweight container that captures the inputs needed by the CPA."""

    model_name: str
    model: Any
    fitted_params: Mapping[str, float]
    config: Mapping[str, Any] | None = None
    thermal_table: ThermalTable | None = None


@dataclass(frozen=True)
class ParameterSnapshot:
    """Structured snapshot that exposes fitted, constant, and derived parameters."""

    fitted: Mapping[str, float]
    constants: Mapping[str, float]
    derived: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted": _serialize_value(self.fitted),
            "constants": _serialize_value(self.constants),
            "derived": _serialize_value(self.derived),
        }

    def to_predictions(self) -> dict[str, Any]:
        derived = self.derived
        return {
            "H0": float(self.fitted.get("H0", 0.0)),
            "Omega_m0": float(self.fitted.get("Omega_m0", 0.0)),
            "Omega_k0": float(derived.get("Omega_k0", 0.0)),
            "S8": float(derived.get("S8", 0.0)) if derived.get("S8") is not None else None,
            "sigma8": float(derived.get("sigma8", 0.0)) if derived.get("sigma8") is not None else None,
            "r_d": float(derived.get("r_d", 0.0)) if derived.get("r_d") is not None else None,
            "plot_data": derived.get("plot_data", {}),
        }


def get_parameter_snapshot(state: ModelState) -> ParameterSnapshot:
    model_name = state.model_name.strip().lower()
    fitted = _coerce_to_floats(state.fitted_params)
    if model_name == "pbuf":
        return _build_pbuf_snapshot(fitted, state)
    return _build_lcdm_snapshot(fitted, state)


# -----------------------
# Snapshot builders
# -----------------------

def _build_lcdm_snapshot(fitted: dict[str, float], state: ModelState) -> ParameterSnapshot:
    model = state.model
    derived = _derive_from_model(model, fitted)
    constants = {
        "Omega_r0": float(_get_parameter(state, "Omega_r0", fallback=fitted.get("Omega_r0", 0.0))),
        "c_km_per_s": float(C_LIGHT_KM_S),
        "G": float(GRAVITATIONAL_CONSTANT),
        "h": float(fitted.get("H0", 0.0)) / 100.0,
        "sigma8_fixed": float(derived.get("sigma8", 0.0)) if derived.get("sigma8") is not None else 0.0,
    }
    return ParameterSnapshot(fitted=fitted, constants=constants, derived=derived)


def _build_pbuf_snapshot(fitted: dict[str, float], state: ModelState) -> ParameterSnapshot:
    model = state.model
    derived = _derive_from_model(model, fitted)
    table = state.thermal_table or getattr(model, "_thermal", None)
    alpha = float(_get_parameter(state, "alpha", fallback=fitted.get("alpha", derived.get("Omega_k0", 0.0))))
    epsilon0 = _get_thermal_constant(table, "epsilon0_T")
    k_sat = (epsilon0 - alpha) if epsilon0 is not None else None
    constants = {
        "Omega_r0": float(_get_parameter(state, "Omega_r0", fallback=fitted.get("Omega_r0", 0.0))),
    }
    derived = dict(derived)
    derived["Omega_k0"] = alpha
    derived.setdefault("Omega_b0", float(fitted.get("Omega_b0", 0.0)))
    derived["alpha"] = alpha
    if epsilon0 is not None:
        derived["epsilon0"] = float(epsilon0)
    if k_sat is not None:
        derived["k_sat"] = float(k_sat)
    return ParameterSnapshot(fitted=fitted, constants=constants, derived=derived)


# -----------------------
# Shared helpers
# -----------------------

def _derive_from_model(model: Any, fitted: dict[str, float]) -> dict[str, Any]:
    z_grid = np.linspace(0.0, 2.0, _PLOT_POINTS, dtype=float)
    a_grid = 1.0 / (1.0 + z_grid)
    H_grid = _safe_array(model.Hubble, z_grid)
    DM_grid = _safe_array(model.DM, z_grid)
    fs8_grid = _safe_array(model.fs8, z_grid)
    growth_a = np.logspace(math.log10(_AGE_GRID_MIN), 0.0, _GROWTH_POINTS, dtype=float)
    growth_z = 1.0 / growth_a - 1.0
    growth_factor = _safe_array(model.growth_factor, growth_z)
    age_info = _compute_age_and_q(model, growth_a)
    sigma8_val = _safe_scalar(model.sigma8)
    s8_val = _safe_scalar(model.S8)
    plot_data = {
        "z": z_grid.tolist(),
        "H_z": H_grid.tolist(),
        "DM_z": DM_grid.tolist(),
        "fs8_z": fs8_grid.tolist(),
    }
    growth_curve = {
        "a": growth_a.tolist(),
        "D": growth_factor.tolist(),
    }
    derived = {
        "Omega_m0": float(fitted.get("Omega_m0", 0.0)),
        "Omega_b0": float(fitted.get("Omega_b0", 0.0)),
        "Omega_k0": float(fitted.get("Omega_k0", 0.0)),
        "S8": float(s8_val) if s8_val is not None else None,
        "sigma8": float(sigma8_val) if sigma8_val is not None else None,
        "r_d": float(_safe_scalar(model.sound_horizon)),
        "age_gyr": age_info["age_gyr"],
        "z_acc": age_info["z_acc"],
        "q0": age_info["q0"],
        "plot_data": plot_data,
        "growth_curve": growth_curve,
    }
    return derived


def _compute_age_and_q(model: Any, a_grid: np.ndarray) -> dict[str, Any]:
    z_grid = 1.0 / a_grid - 1.0
    H_grid = _safe_array(model.Hubble, z_grid)
    H_si = H_grid * 1e3 / MPC_TO_KM
    H_safe = np.clip(H_grid, 1e-30, None)
    integrand = 1.0 / (a_grid * np.clip(H_si, 1e-30, None))
    age_seconds = float(np.trapz(integrand, a_grid))
    age = age_seconds / SECONDS_PER_GYR
    dH_da = np.gradient(H_grid, a_grid)
    q_vals = - (a_grid / np.clip(H_safe, 1e-30, None)) * dH_da - 1.0
    q0 = float(q_vals[-1])
    negative = np.where(q_vals < 0.0)[0]
    z_acc = None
    if negative.size:
        idx = int(negative[0])
        z_acc = float(max(0.0, (1.0 / a_grid[idx]) - 1.0))
    return {"age_gyr": age, "z_acc": z_acc, "q0": q0}


def _get_thermal_constant(table: ThermalTable | None, field: str) -> float | None:
    if table is None:
        return None
    try:
        return float(table.fast_get(field, at_scale_factor=1.0))
    except Exception:
        try:
            return float(table.get(field, at_scale_factor=1.0))
        except Exception:
            return None


def _safe_array(func: Any, inputs: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(func(inputs), dtype=float)
        if values.size != inputs.size:
            values = np.asarray(func(inputs.tolist()), dtype=float)
        return np.nan_to_num(values, nan=0.0, posinf=np.inf, neginf=-np.inf)
    except Exception:
        return np.zeros(inputs.size, dtype=float)


def _safe_scalar(func: Any) -> float | None:
    try:
        return float(func())
    except Exception:
        try:
            return float(func)
        except Exception:
            return None


def _coerce_to_floats(mapping: Mapping[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in mapping.items() if value is not None}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_serialize_value(item) for item in list(value)]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _get_parameter(state: ModelState, key: str, fallback: float = 0.0) -> float:
    candidate = state.fitted_params.get(key)
    if candidate is not None:
        try:
            return float(candidate)
        except Exception:
            pass
    model_params = getattr(state.model, "parameters", {})
    if isinstance(model_params, Mapping) and key in model_params:
        try:
            return float(model_params[key])
        except Exception:
            pass
    return fallback


__all__ = ["ModelState", "ParameterSnapshot", "get_parameter_snapshot"]
