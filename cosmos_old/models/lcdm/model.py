"""LCDM model implementation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict

import math
import numpy as np

from cosmos.interfaces import CMBOutput, CosmologyModel
from cosmos.models.common.distance_utils import (
    distance_modulus_from_luminosity_distance,
    luminosity_distance,
    luminosity_distance_non_decreasing,
    transverse_comoving_distance,
)
from cosmos.models.common.growth import GrowthTable
from cosmos.models.lcdm import cmb as cmb_module
from cosmos.models.lcdm import utils as lcdm_utils
from cosmos.models.lcdm.distances import H_z as H_z_func
from cosmos.models.lcdm.distances import comoving_distance
from cosmos.models.lcdm.params import LCDMParams
from cosmos.optim.sanity_base import SanityResult


class LCDMModel(CosmologyModel):
    """Standard ΛCDM cosmology with the common CMB interface."""

    def __init__(self, **params: float) -> None:
        coerced = self._coerce_params(params)
        lcdm_params = LCDMParams(**coerced)

        if lcdm_params.Omega_lambda0 is None:
            Omega_lambda = 1.0 - lcdm_params.Omega_m0 - lcdm_params.Omega_r0 - lcdm_params.Omega_k0
            lcdm_params = lcdm_params.with_lambda(Omega_lambda)

        self._params = lcdm_params
        self._parameters = dict(asdict(self._params))
        self._distance_cache: dict[float, tuple[float, float, float]] = {}
        self._distance_steps = 4096
        self._sound_steps = 4096
        self._distance_integrator = self._make_distance_integrator()
        self._sound_integrator = self._make_simpson_integrator(self._sound_steps)
        self._sound_cache: float | None = None
        self._growth_table: GrowthTable | None = None
        self._sigma8_today = float(lcdm_params.sigma8_0)
        self._expansion_cache: dict[float, tuple[float, float]] = {}
        self._growth_cache: dict[tuple[str, float], float] = {}
        self._distance_sanity_checked = False
        self._distance_sanity_ok = True
        self._distance_sanity_reasons: tuple[str, ...] = ()

    @property
    def params(self) -> LCDMParams:
        return self._params

    def cmb(self, data: Any) -> CMBOutput:
        return cmb_module.compute_cmb_output(self._params)

    def chi2_cmb(self, dataset: Any) -> tuple[CMBOutput, np.ndarray, float]:
        """Compute the CMB χ² for this LCDM instance with diagnostic logging."""

        output = self.cmb(dataset)
        predicted = np.array(
            [
                output.R,
                output.l_A,
                output.theta_star,
            ]
        )
        residual = predicted - dataset.observed
        weighted = dataset.inv_covariance @ residual
        chi2 = float(residual.T @ weighted)

        print("LCDMModel.chi2_cmb called with params:", asdict(self._params))
        print("Computed distances:", output.D_M_Mpc, output.r_s_Mpc, output.l_A, output.theta_star)
        print("Returned chi2:", chi2)

        return output, residual, chi2

    @property
    def parameters(self) -> Dict[str, float]:
        return dict(self._parameters)

    def omega_m0(self) -> float:
        """Present-day total matter fraction."""

        return float(self._params.Omega_m0)

    def sigma8(self) -> float:
        """Return σ₈ today using the cached growth table."""

        sigma8_today = float(self._sigma8_today)
        solver = self._ensure_growth_table()
        return float(solver.sigma8(1.0, sigma8_today))

    def S8(self, gamma: float = 0.5) -> float:
        """Compute S₈ with the same definition used in WL-style constraints."""

        om = self.omega_m0()
        if om <= 0.0:
            raise ValueError("Model returned non-positive Ω_m0; cannot build S₈.")
        s8 = self.sigma8()
        return float(s8 * (om / 0.3) ** float(gamma))

    def is_valid(self) -> bool:
        """Run the LCDM sanity suite and return whether it passed."""

        from cosmos.models.lcdm.sanity import check_lcdm_sanity

        sanitized = {key: float(value) for key, value in self.parameters.items()}
        result = check_lcdm_sanity(sanitized, self)
        return result.ok

    def distance_modulus(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        values = np.asarray(z, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.ravel()

        if not self._ensure_distance_sanity():
            fill = np.full(flat.shape, np.inf, dtype=float)
            return float(fill[0]) if scalar_input else fill.reshape(values.shape)

        mu_arr = np.empty_like(flat)
        dL_arr = np.empty_like(flat)
        for idx, zval in enumerate(flat):
            mu, dL, _ = self._distance_entry(float(zval))
            mu_arr[idx] = mu
            dL_arr[idx] = dL

        if not luminosity_distance_non_decreasing(flat, dL_arr):
            fill = np.full(flat.shape, np.inf, dtype=float)
            return float(fill[0]) if scalar_input else fill.reshape(values.shape)

        result = mu_arr.reshape(values.shape)
        return float(result.item()) if scalar_input else result

    def DV(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        values = np.asarray(z, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.ravel()

        if not self._ensure_distance_sanity():
            fill = np.full(flat.shape, np.inf, dtype=float)
            return float(fill[0]) if scalar_input else fill.reshape(values.shape)

        dv_arr = np.empty_like(flat)
        for idx, zval in enumerate(flat):
            dv_arr[idx] = self._dv_scalar(float(zval))

        result = dv_arr.reshape(values.shape)
        return float(result.item()) if scalar_input else result

    def DM(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_scalar_array(z, self._transverse_comoving_distance)

    def D_M(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        """Alias for DM to mirror the cosmos2 API."""
        return self.DM(z)

    def DA(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        def _compute(z_val: float) -> float:
            D_M = self._transverse_comoving_distance(z_val)
            if not math.isfinite(D_M):
                return math.inf
            denom = 1.0 + z_val
            if denom <= 0.0:
                return math.inf
            return D_M / denom

        return self._evaluate_scalar_array(z, _compute)

    def DH(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        def _compute(z_val: float) -> float:
            H_val = self._hubble_rate(z_val)
            if H_val <= 0.0 or not math.isfinite(H_val):
                return math.inf
            return lcdm_utils.C_LIGHT / H_val

        return self._evaluate_scalar_array(z, _compute)

    def Hubble(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_scalar_array(z, self._hubble_rate)

    def _distance_for_redshift(self, z_val: float) -> tuple[float, float]:
        mu, dL, _ = self._distance_entry(z_val)
        return mu, dL

    def _distance_entry(self, z_val: float) -> tuple[float, float, float]:
        if z_val < 0.0:
            bad = math.inf  # type: ignore[attr-defined]
            return bad, bad, bad

        cache_key = float(z_val)
        cached = self._distance_cache.get(cache_key)
        if cached is not None:
            return cached

        D_M = self._transverse_comoving_distance(z_val)
        D_L = luminosity_distance(D_M, z_val)
        mu = distance_modulus_from_luminosity_distance(D_L)
        entry = (mu, D_L, D_M)
        self._distance_cache[cache_key] = entry
        return entry

    def sound_horizon(self) -> float:
        if self._sound_cache is not None:
            return self._sound_cache

        result = cmb_module.sound_horizon_drag(self._params, self._sound_integrator)
        self._sound_cache = float(result)
        return self._sound_cache

    def _dv_scalar(self, z_val: float) -> float:
        if z_val < 0.0:
            return math.inf

        D_M = self._transverse_comoving_distance(z_val)
        if not math.isfinite(D_M):
            return math.inf

        D_A = D_M / (1.0 + z_val)
        H_val = self._hubble_rate(z_val)
        if H_val <= 0.0:
            return math.inf

        factor = z_val * (1.0 + z_val) ** 2 * D_A ** 2 * lcdm_utils.C_LIGHT / H_val
        if factor <= 0.0:
            return math.inf
        return float(factor ** (1.0 / 3.0))

    def _transverse_comoving_distance(self, z_val: float) -> float:
        chi = self._comoving_distance(z_val)
        return transverse_comoving_distance(chi, self._params.H0, self._params.Omega_k0)

    def _hubble_rate(self, z_val: float) -> float:
        hubble, _ = self._expansion_entry(z_val)
        return hubble

    def _comoving_distance(self, z_val: float) -> float:
        def integrand(zp: float) -> float:
            H_val, _ = self._expansion_entry(zp)
            if H_val <= 0.0:
                return math.inf
            return lcdm_utils.C_LIGHT / H_val

        return self._distance_integrator(integrand, 0.0, z_val)

    def _expansion_entry(self, z_val: float) -> tuple[float, float]:
        key = float(z_val)
        cached = self._expansion_cache.get(key)
        if cached is not None:
            return cached

        hubble = H_z_func(key, self._params)
        scale_factor = self._safe_scale_factor(key)
        entry = (hubble, scale_factor)
        self._expansion_cache[key] = entry
        return entry

    def _evaluate_scalar_array(
        self,
        z_values: float | np.ndarray | list | tuple,
        func: Callable[[float], float],
    ) -> float | np.ndarray:
        values = np.asarray(z_values, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.ravel()

        if not self._ensure_distance_sanity():
            fill = np.full(flat.shape, math.inf, dtype=float)
            return float(fill[0]) if scalar_input else fill.reshape(values.shape)

        result = np.empty_like(flat)
        for idx, zval in enumerate(flat):
            result[idx] = func(float(zval))

        shaped = result.reshape(values.shape)
        return float(shaped.item()) if scalar_input else shaped

    def _ensure_distance_sanity(self) -> bool:
        if self._distance_sanity_checked:
            return self._distance_sanity_ok

        from cosmos.models.lcdm import sanity as lcdm_sanity

        result = SanityResult()
        result.merge(lcdm_sanity.check_closure_lcdm(self.parameters, self))
        result.merge(lcdm_sanity.check_expansion_lcdm(self.parameters, self))

        self._distance_sanity_checked = True
        self._distance_sanity_ok = result.ok
        if not result.ok:
            self._distance_sanity_reasons = tuple(result.reasons)

        return self._distance_sanity_ok

    def _make_distance_integrator(self):
        return self._make_simpson_integrator(self._distance_steps)

    def _make_simpson_integrator(self, steps: int):
        n = int(steps)
        if n <= 0:
            raise ValueError("Integrators require a positive step count.")
        if n % 2 != 0:
            n += 1

        def integrator(func, lower, upper):
            return lcdm_utils.simpson_integral(func, lower, upper, n=n)

        return integrator

    def _ensure_growth_table(self) -> GrowthTable:
        if self._growth_table is None:
            from cosmos.models.lcdm.growth import growth_ode_rhs

            def rhs(a: float, y: np.ndarray) -> np.ndarray:
                return growth_ode_rhs(a, y, self._params)

            self._growth_table = GrowthTable(rhs)
        return self._growth_table

    @staticmethod
    def _safe_scale_factor(z_value: float) -> float:
        z_safe = max(float(z_value), -0.999999999)
        return 1.0 / (1.0 + z_safe)

    def _evaluate_growth_prediction(
        self,
        z_values: float | np.ndarray | list | tuple,
        actor_name: str,
        actor: Callable[[GrowthTable, float], float],
    ) -> float | np.ndarray:
        values = np.asarray(z_values, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.ravel()
        solver = self._ensure_growth_table()
        result = np.empty_like(flat)
        for idx, raw in enumerate(flat):
            if not np.isfinite(raw):
                result[idx] = np.nan
                continue
            _, a = self._expansion_entry(raw)
            cache_key = (actor_name, a)
            cached = self._growth_cache.get(cache_key)
            if cached is not None:
                result[idx] = cached
                continue
            value = actor(solver, a)
            self._growth_cache[cache_key] = value
            result[idx] = value
        shaped = result.reshape(values.shape)
        return float(shaped.item()) if scalar_input else shaped

    def growth_factor(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_growth_prediction(
            z,
            "growth_factor",
            lambda solver, a: solver.growth_factor(a),
        )

    def growth_rate(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_growth_prediction(
            z,
            "growth_rate",
            lambda solver, a: solver.growth_rate(a),
        )

    def fs8(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        sigma8_today = float(self._sigma8_today)
        return self._evaluate_growth_prediction(
            z,
            "fs8",
            lambda solver, a: solver.fs8(a, sigma8_today),
        )

    def _coerce_params(self, params: Dict[str, float]) -> Dict[str, float]:
        required = ["H0", "Omega_m0", "Omega_r0", "Omega_k0", "Omega_b0"]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required LCDM parameters: {missing}")
        normalized: Dict[str, float] = {}
        for key in required:
            normalized[key] = float(params[key])
        if "Omega_lambda0" in params:
            normalized["Omega_lambda0"] = float(params["Omega_lambda0"])
        normalized["sigma8_0"] = float(params.get("sigma8_0", 0.811))
        return normalized

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"LCDMModel({asdict(self._params)})"
