"""Core PBUF model implementation."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
# `typing` is intentionally below to avoid re-exporting to protocols.
from typing import Any, Callable, Dict, Literal, Optional

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
from cosmos.models.pbuf import cmb as cmb_module
from cosmos.models.pbuf import utils as pbuf_utils
from cosmos.models.pbuf.distances import H as H_of_a
from cosmos.models.pbuf.distances import H_z as H_z_func
from cosmos.models.pbuf.distances import omega_total_at_a
from cosmos.models.pbuf.elastic import omega_sigma_of_a, omega_sigma_raw_of_a
from cosmos.models.pbuf.microphysics import ensure_thermal_table, get_last_bootstrap_metadata
from cosmos.models.pbuf.params import PBUFParams, coerce_pbuf_parameters
from cosmos.models.pbuf.thermal_table import ThermalTable
from cosmos.optim.sanity_base import SanityResult


NormalizationMode = Literal["free", "flat_today"]

# PBUF derives σ₈ from the growth ODE with D(1)=1 normalization.
# This constant is used only as a scaling factor, not in any physical calculation.
# The value 0.811 is the Planck 2018 result.
_SIGMA8_TODAY = 0.811


class PBUFModel(CosmologyModel):
    """Implements the PBUF cosmology with explicit R_max control."""

    def __init__(
        self,
        *,
        thermal_table_path: Optional[str] = None,
        thermal_table: Optional[ThermalTable] = None,
        thermal_metadata: Optional[Dict[str, Any]] = None,
        normalization_mode: Optional[NormalizationMode] = None,
        **params: Any,
    ) -> None:
        alpha_override = params.pop("alpha", None)
        raw = self._coerce_params(params, normalization_mode)
        self._raw_params = PBUFParams(**raw)
        self._normalization_metadata: Dict[str, Any] = {}

        if thermal_table is not None:
            self._thermal = thermal_table
        else:
            if thermal_table_path is not None:
                self._thermal = ThermalTable(Path(thermal_table_path))
            else:
                self._thermal = ensure_thermal_table()
        metadata = thermal_metadata if thermal_metadata is not None else get_last_bootstrap_metadata()
        if metadata is None:
            metadata = {}
        self.micro_bootstrap_metadata = metadata
        if alpha_override is not None:
            meta = dict(self.micro_bootstrap_metadata)
            meta["alpha"] = float(alpha_override)
            self.micro_bootstrap_metadata = meta
        resolved_params = self._apply_normalization(self._raw_params, self._thermal)
        resolved_params = replace(resolved_params, alpha=self.alpha)
        self._params = resolved_params
        self._parameters = dict(asdict(self._params))
        self._distance_cache: dict[float, tuple[float, float, float]] = {}
        self._distance_steps = 4096
        self._distance_integrator = self._make_distance_integrator()
        self._sound_steps = 4096
        self._sound_integrator = self._make_sound_integrator()
        self._sound_cache: float | None = None
        self._growth_table: GrowthTable | None = None
        self._sigma8_today = _SIGMA8_TODAY
        self._expansion_cache: dict[float, tuple[float, float]] = {}
        self._growth_cache: dict[tuple[str, float], float] = {}
        self._distance_sanity_checked = False
        self._distance_sanity_ok = True
        self._distance_sanity_reasons: tuple[str, ...] = ()

    @property
    def params(self) -> PBUFParams:
        """Expose the frozen parameter dataclass."""

        return self._params

    @property
    def raw_params(self) -> PBUFParams:
        """Return the input parameters prior to any runtime normalization."""

        return self._raw_params

    @property
    def thermal_table(self) -> ThermalTable:
        """Expose the loaded thermal table."""

        return self._thermal

    def ensure_quantum_and_thermal_table(self) -> None:
        """Ensure that the quantum bootstrap and thermal table are available."""

        if getattr(self, "_thermal", None) is not None:
            return

        self._thermal = ensure_thermal_table()
        self.micro_bootstrap_metadata = get_last_bootstrap_metadata()

    @property
    def normalization_metadata(self) -> Dict[str, Any]:
        """Return diagnostic information about the active normalization."""

        return dict(self._normalization_metadata)

    def cmb(self, data: Any) -> CMBOutput:
        """Return CMB distance priors for the stored parameters."""

        return cmb_module.compute_cmb_output(self._params, self._thermal)

    @property
    def parameters(self) -> Dict[str, float]:
        return dict(self._parameters)

    @property
    def alpha(self) -> float:
        metadata = self.micro_bootstrap_metadata or {}
        for key in ("alpha_qm", "alpha"):
            value = metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        table_meta = getattr(self._thermal, "metadata", {}) if hasattr(self, "_thermal") else {}
        for key in ("alpha_qm", "alpha"):
            value = table_meta.get(key)
            if value is not None:
                return float(value)
        if hasattr(self, "_raw_params"):
            try:
                return float(self._raw_params.alpha)
            except Exception:
                pass
        return 0.0

    def _numeric_parameters(self) -> Dict[str, float]:
        """Return only numeric parameters so sanities skip the normalization mode string."""
        sanitized: Dict[str, float] = {}
        for key, value in self.parameters.items():
            try:
                sanitized[key] = float(value)
            except (TypeError, ValueError):
                continue
        return sanitized

    def omega_m0(self) -> float:
        """Present-day total matter density fraction in this PBUF instance."""

        return float(self._params.Omega_m0)

    def sigma8(self) -> float:
        """Return σ₈ today via the cached growth solution (D(1)=1)."""

        sigma8_today = float(self._sigma8_today)
        solver = self._ensure_growth_table()
        return float(solver.sigma8(1.0, sigma8_today))

    def S8(self, gamma: float = 0.5) -> float:
        """Return S₈ = σ₈(Ωₘ/0.3)^γ for the supplied γ exponent."""

        om = self.omega_m0()
        if om <= 0.0:
            raise ValueError("Model returned non-positive Ω_m0; cannot build S₈.")
        s8 = self.sigma8()
        return float(s8 * (om / 0.3) ** float(gamma))

    def is_valid(self) -> bool:
        """Return whether this PBUF instance satisfies the Phase-6a sanity checks."""

        from cosmos.models.lcdm.model import LCDMModel
        from cosmos.models.pbuf.sanity import check_pbuf_sanity

        def _lcdm_factory(**kwargs: float) -> LCDMModel:
            return LCDMModel(**kwargs)

        sanitized = self._numeric_parameters()
        result = check_pbuf_sanity(
            sanitized,
            self,
            lcdm_model_factory=_lcdm_factory,
        )
        return result.ok

    def distance_modulus(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        values = np.asarray(z, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.ravel()

        if not self._ensure_distance_sanity():
            if scalar_input:
                return float(np.inf)
            return np.full(flat.shape, np.inf).reshape(values.shape)

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
            if scalar_input:
                return float(np.inf)
            return np.full(flat.shape, np.inf).reshape(values.shape)

        dv_arr = np.empty_like(flat)
        for idx, zval in enumerate(flat):
            dv_arr[idx] = self._dv_scalar(float(zval))

        result = dv_arr.reshape(values.shape)
        return float(result.item()) if scalar_input else result

    def DM(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_scalar_array(z, self._transverse_comoving_distance)

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
            return pbuf_utils.C_LIGHT / H_val

        return self._evaluate_scalar_array(z, _compute)

    def Hubble(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_scalar_array(z, self._hubble_rate)

    def _distance_for_redshift(self, z_val: float) -> tuple[float, float]:
        mu, dL, _ = self._distance_entry(z_val)
        return mu, dL

    def _distance_entry(self, z_val: float) -> tuple[float, float, float]:
        if z_val < 0.0:
            bad = math.inf
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

        result = cmb_module.sound_horizon_drag(self._params, self._thermal, self._sound_integrator)
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

        factor = z_val * (1.0 + z_val) ** 2 * D_A ** 2 * pbuf_utils.C_LIGHT / H_val
        if factor <= 0.0:
            return math.inf
        return float(factor ** (1.0 / 3.0))

    def _transverse_comoving_distance(self, z_val: float) -> float:
        chi = self._comoving_distance(z_val)
        return transverse_comoving_distance(chi, self._params.H0, self.alpha)

    def _hubble_rate(self, z_val: float) -> float:
        hubble, _ = self._expansion_entry(z_val)
        return hubble

    def _comoving_distance(self, z_val: float) -> float:
        def integrand(zp: float) -> float:
            H_val, _ = self._expansion_entry(zp)
            if H_val <= 0.0:
                return math.inf
            return pbuf_utils.C_LIGHT / H_val

        return self._distance_integrator(integrand, 0.0, z_val)

    def _expansion_entry(self, z_val: float) -> tuple[float, float]:
        key = float(z_val)
        cached = self._expansion_cache.get(key)
        if cached is not None:
            return cached

        hubble = H_z_func(key, self._params, self._thermal)
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

        from cosmos.models.pbuf import sanity as pbuf_sanity

        result = pbuf_sanity.check_pbuf_sanity(self._numeric_parameters(), self)

        self._distance_sanity_checked = True
        self._distance_sanity_ok = result.ok
        if not result.ok:
            self._distance_sanity_reasons = tuple(result.reasons)

        return self._distance_sanity_ok

    def _make_distance_integrator(self):
        return self._make_simpson_integrator(self._distance_steps)

    def _make_sound_integrator(self):
        return self._make_simpson_integrator(self._sound_steps)

    def _make_simpson_integrator(self, steps: int):
        n = int(steps)
        if n <= 0:
            raise ValueError("Simpson integrators require a positive even number of steps.")
        if n % 2 != 0:
            n += 1

        def integrator(func, lower, upper):
            return pbuf_utils.simpson_integral(func, lower, upper, n=n)

        return integrator

    def _ensure_growth_table(self) -> GrowthTable:
        if self._growth_table is None:
            from cosmos.models.pbuf.growth import growth_ode_rhs

            def rhs(a: float, y: np.ndarray) -> np.ndarray:
                return growth_ode_rhs(a, y, self._params, self._thermal)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_params(self, params: Dict[str, Any], normalization_mode: Optional[str]) -> Dict[str, Any]:
        return coerce_pbuf_parameters(params, normalization_mode=normalization_mode)

    def _apply_normalization(self, params: PBUFParams, table: ThermalTable) -> PBUFParams:
        mode = params.omega_normalization
        if mode == "free":
            self._normalization_metadata = {"mode": "free", "sigma_rescale": 1.0}
            return params

        if mode == "flat_today":
            sigma_target = 1.0 - params.Omega_m0 - params.Omega_r0 - self.alpha
            if sigma_target <= 0.0:
                raise ValueError("Cannot enforce flat_today normalization because Ω_sigma target ≤ 0.")

            omega_raw = omega_sigma_raw_of_a(1.0, params, table)
            if omega_raw <= 0.0:
                raise ValueError("Cannot normalize Ω_sigma because the raw Ω_sigma(a=1) ≤ 0.")

            rescale = sigma_target / omega_raw
            resolved = replace(params, sigma_rescale=rescale)
            omega_total = omega_total_at_a(1.0, resolved, table, alpha=self.alpha)
            omega_sigma = omega_sigma_of_a(1.0, resolved, table)
            self._normalization_metadata = {
                "mode": "flat_today",
                "sigma_rescale": rescale,
                "omega_total_a1": omega_total,
                "omega_sigma_a1": omega_sigma,
                "omega_sigma_target": sigma_target,
                "omega_sigma_raw_a1": omega_raw,
            }
            return resolved

        raise RuntimeError(f"Unsupported omega_normalization mode '{mode}'.")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"PBUFModel({asdict(self._params)})"
