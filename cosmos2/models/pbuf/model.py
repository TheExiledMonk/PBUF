"""PBUF model backed by the ported per-model helpers (cosmos2)."""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Any, Dict, Mapping, Sequence
from pathlib import Path

import numpy as np

from cosmos2.kernels.pbuf_grids import build_grids_njit
from cosmos2.kernels.pbuf_observables import (
    dh_from_H_kernel,
    distance_modulus_from_DM_kernel,
    dv_from_DM_H_kernel,
)
from cosmos2.models.lcdm.common import CMBOutput
from cosmos2.pbuf.microphysics import ensure_thermal_table, get_last_bootstrap_metadata

from . import cmb as cmb_module
from . import distances
from . import utils as pbuf_utils
from .normalization import normalize_parameters
from .params import PBUFParams, coerce_pbuf_parameters
from .sanity import check_pbuf_sanity
from .thermal_table import ThermalTable
from .growth import growth_ode_rhs, make_growth_rhs_njit
from .growth_table import GrowthTable

C_LIGHT = pbuf_utils.C_LIGHT
_GROWTH_RHS_ENV = os.environ.get("PBUF_GROWTH_RHS", "").strip().lower()
_FORCE_PYTHON_GROWTH = _GROWTH_RHS_ENV in {"python", "py", "force_python"}
_GRID_ENV = os.environ.get("PBUF_BACKGROUND_GRID", "").strip().lower()
_FORCE_PYTHON_GRID = _GRID_ENV in {"python", "py", "force_python"}


def _to_array(z: float | Sequence[float]) -> tuple[np.ndarray, bool]:
    arr = np.atleast_1d(np.asarray(z, dtype=float))
    return arr, np.isscalar(z)


def _distance_modulus_from_DM(DM: np.ndarray, z: np.ndarray) -> np.ndarray:
    mu = np.empty_like(z, dtype=float)
    for i in range(z.shape[0]):
        mu[i] = distance_modulus_from_DM_kernel(float(DM[i]), float(z[i]))
    return mu


def _dv_from_DM_H(z: np.ndarray, DM: np.ndarray, H: np.ndarray) -> np.ndarray:
    dv = np.empty_like(z, dtype=float)
    for i in range(z.shape[0]):
        dv[i] = dv_from_DM_H_kernel(float(z[i]), float(DM[i]), float(H[i]))
    return dv


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Lightweight cumulative trapezoid with a zero baseline."""

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    mids = 0.5 * (y[:-1] + y[1:])
    return np.concatenate([[0.0], np.cumsum(dx * mids)])


class PBUFModel:
    """
    Minimal PBUF implementation that mirrors the legacy CosmologyModel surface.
    """

    def __init__(
        self,
        *,
        thermal_table_path: str | None = None,
        thermal_table: ThermalTable | None = None,
        thermal_metadata: Mapping[str, Any] | None = None,
        normalization_mode: str | None = None,
        n_grid: int | None = None,
        **params: Any,
    ) -> None:
        raw_clean = coerce_pbuf_parameters(params, normalization_mode=normalization_mode)
        self._raw_params = PBUFParams(**raw_clean)
        self._thermal = thermal_table or (ThermalTable(thermal_table_path) if thermal_table_path else ensure_thermal_table())
        self.thermal_table = self._thermal  # Expose for sanity gates expecting 'thermal_table'
        self.micro_bootstrap_metadata: Dict[str, Any] = dict(thermal_metadata or get_last_bootstrap_metadata() or {})

        # Resolve alpha/closure and rescale omega_sigma if needed.
        normalized, norm_meta, resolved_alpha = normalize_parameters(self._raw_params, self._thermal, self.micro_bootstrap_metadata)
        self._alpha = float(resolved_alpha)
        self._normalization_metadata = norm_meta
        self._params = normalized
        self._parameters = dict(asdict(self._params))
        self._parameters["alpha_resolved"] = self._alpha
        self._parameters["normalization_metadata"] = dict(self._normalization_metadata)

        # Derive sigma8 directly from Omega_m0 (σ₈ ≈ 1 - Ωₘ₀).
        omega_m0_val = float(self._params.Omega_m0)
        self._sigma8 = float(1.0 - omega_m0_val)
        self._sigma8_today = self._sigma8
        self._r_d: float | None = None

        self._n_grid = int(n_grid) if n_grid is not None else 4000
        self._a_grid: np.ndarray | None = None
        self._H_grid: np.ndarray | None = None
        self._DM_grid: np.ndarray | None = None
        self._growth_table: GrowthTable | None = None
        self._growth_cache: Dict[tuple[str, float], float] = {}
        self._build_grids()

    # --------------------------
    # CosmologyModel properties
    # --------------------------
    @property
    def parameters(self) -> Dict[str, float]:
        return dict(self._parameters)

    def omega_m0(self) -> float:
        return float(self._params.Omega_m0)

    def sigma8(self) -> float:
        return float(self._sigma8)

    def cmb(self, data: Any) -> CMBOutput:
        return cmb_module.compute_cmb_output(self._params, self._thermal)

    def distance_modulus(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        DM = np.interp(a, self._a_grid, self._DM_grid)
        d_L = DM * (1.0 + z_arr)
        mu = _distance_modulus_from_DM(d_L, z_arr)
        return float(mu[0]) if scalar else mu

    def DV(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        DM = np.interp(a, self._a_grid, self._DM_grid)
        H = np.interp(a, self._a_grid, self._H_grid)
        dv = _dv_from_DM_H(z_arr, DM, H)
        return float(dv[0]) if scalar else dv

    def DM(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        vals = np.interp(a, self._a_grid, self._DM_grid)
        return float(vals[0]) if scalar else vals

    def DA(self, z: float | Sequence[float]) -> float | np.ndarray:
        vals = self.DM(z)
        z_arr, scalar = _to_array(z)
        vals = np.asarray(vals, dtype=float) / (1.0 + z_arr)
        return float(vals[0]) if scalar else vals

    def DH(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        hubble = np.asarray(self.Hubble(z_arr), dtype=float)
        vals = np.array([dh_from_H_kernel(val) for val in hubble], dtype=float)
        return float(vals[0]) if scalar else vals

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        vals = np.interp(a, self._a_grid, self._H_grid)
        return float(vals[0]) if scalar else vals

    def sound_horizon(self) -> float:
        if self._r_d is None:
            integrator = lambda f, a, b: pbuf_utils.simpson_integral(f, a, b, n=4096)
            self._r_d = cmb_module.sound_horizon_drag(self._params, self._thermal, integrator)
        return float(self._r_d)

    def growth_factor(self, z: float | Sequence[float]) -> float | np.ndarray:
        return self._evaluate_growth_prediction(
            z,
            "growth_factor",
            lambda solver, a: solver.growth_factor(a),
        )

    def growth_rate(self, z: float | Sequence[float]) -> float | np.ndarray:
        return self._evaluate_growth_prediction(
            z,
            "growth_rate",
            lambda solver, a: solver.growth_rate(a),
        )

    def fs8(self, z: float | Sequence[float]) -> float | np.ndarray:
        sigma8_today = float(self._sigma8_today)
        return self._evaluate_growth_prediction(
            z,
            "fs8",
            lambda solver, a: solver.fs8(a, sigma8_today),
        )

    def S8(self, gamma: float = 0.5) -> float:
        return float(self._sigma8 * (self.omega_m0() / 0.3) ** gamma)

    def is_valid(self) -> bool:
        from cosmos2.models.lcdm.model import LCDMModel

        def _lcdm_factory(**kwargs: float) -> LCDMModel:
            return LCDMModel(**kwargs)

        result = check_pbuf_sanity(
            self._numeric_parameters(),
            self,
            lcdm_model_factory=_lcdm_factory,
        )
        return result.ok

    # --------------------------
    # Internal helpers
    # --------------------------
    def _numeric_parameters(self) -> Dict[str, float]:
        sanitized: Dict[str, float] = {}
        for key, value in self.parameters.items():
            try:
                sanitized[key] = float(value)
            except (TypeError, ValueError):
                continue
        return sanitized

    def _build_grids(self) -> None:
        n_grid = max(self._n_grid, 32)
        a_min = 1.0e-6
        self._a_grid = np.logspace(np.log10(a_min), 0.0, n_grid, dtype=np.float64)
        # Try nopython grid builder; fall back to Python vectorized path on failure or when forced off.
        try:
            mode = getattr(self._params, "omega_normalization", "flat_today")
            mode_flag = 0 if mode == "free" else 1 if mode == "flat_today" else -1
            if not _FORCE_PYTHON_GRID and mode_flag >= 0:
                sigma_rescale = float(getattr(self._params, "sigma_rescale", 1.0))
                a_table, log_a, T_arr, eps_arr, alpha_arr, dln_eps, dln_alpha, g_star, g_starS = self._thermal.numba_payload()
                H_vals, DM_vals = build_grids_njit(
                    self._a_grid,
                    float(self._params.H0),
                    float(self._params.Omega_m0),
                    float(self._params.Omega_r0),
                    float(self._params.alpha),
                    float(self._params.Rmax),
                    sigma_rescale,
                    mode_flag,
                    a_table,
                    alpha_arr,
                    eps_arr,
                )
                self._H_grid = H_vals
                self._DM_grid = DM_vals
                return
        except Exception:
            pass

        # Background H(a)
        H_vals = np.array([distances.H(a, self._params, self._thermal) for a in self._a_grid], dtype=float)
        self._H_grid = H_vals

        # Transverse comoving distance grid: integrate c/(a^2 H(a)) from a to 1.
        integrand = C_LIGHT / (self._a_grid * self._a_grid * np.maximum(H_vals, 1e-12))
        cumulative = _cumtrapz(integrand, self._a_grid)
        total = float(cumulative[-1])
        self._DM_grid = total - cumulative

    def _ensure_growth_table(self) -> GrowthTable:
        if self._growth_table is None:
            def rhs(a: float, y: np.ndarray) -> np.ndarray:
                return growth_ode_rhs(a, y, self._params, self._thermal)
            rhs_njit = None
            if not _FORCE_PYTHON_GROWTH:
                rhs_njit = make_growth_rhs_njit(self._params, self._thermal)
            self._growth_table = GrowthTable(rhs, rhs_njit=rhs_njit)
        return self._growth_table

    @staticmethod
    def _safe_scale_factor(z_value: float) -> float:
        z_safe = max(float(z_value), -0.999999999)
        return 1.0 / (1.0 + z_safe)

    def _evaluate_growth_prediction(
        self,
        z_values: float | np.ndarray | Sequence[float],
        actor_name: str,
        actor: Any,
    ) -> float | np.ndarray:
        values = np.asarray(z_values, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.ravel()
        solver = self._ensure_growth_table()
        result = np.empty_like(flat, dtype=float)
        for idx, raw in enumerate(flat):
            if not np.isfinite(raw):
                result[idx] = np.nan
                continue
            a = self._safe_scale_factor(raw)
            cache_key = (actor_name, a)
            cached = self._growth_cache.get(cache_key)
            if cached is not None:
                result[idx] = cached
                continue
            value = actor(solver, a)
            self._growth_cache[cache_key] = float(value)
            result[idx] = float(value)
        shaped = result.reshape(values.shape)
        return float(shaped.item()) if scalar_input else shaped
