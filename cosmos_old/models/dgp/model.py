"""Simple DGP braneworld cosmology that reuses shared distance and growth utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict

import math

import numpy as np

from cosmos.interfaces import CMBOutput, CosmologyModel
from cosmos.models.common.distance_utils import (
    C_LIGHT,
    distance_modulus_from_luminosity_distance,
    luminosity_distance,
    transverse_comoving_distance,
)
from cosmos.models.common.growth import GrowthTable
from cosmos.models.dgp.params import DGPParams
from cosmos.models.lcdm import utils as lcdm_utils


class DGPModel(CosmologyModel):
    """Dvali–Gabadadze–Porrati cosmology for comparisons with ΛCDM+MG."""

    def __init__(self, **params: float) -> None:
        coerced = self._coerce_params(params)
        self._params = DGPParams(**coerced)
        self._sqrt_Omega_rc = math.sqrt(max(self._params.Omega_rc, 0.0))
        self._epsilon_branch = 1 if self._params.epsilon_branch >= 0 else -1
        self._parameters: Dict[str, float] = dict(asdict(self._params))
        self._parameters["r_c_Mpc"] = self.rc()
        self._distance_steps = 4096
        self._distance_integrator = self._make_simpson_integrator(self._distance_steps)
        self._growth_table: GrowthTable | None = None
        self._growth_cache: Dict[tuple[str, float], float] = {}
        self._sigma8_today = float(self._params.sigma8_0)

    @property
    def params(self) -> DGPParams:
        return self._params

    @property
    def parameters(self) -> Dict[str, float]:
        return dict(self._parameters)

    def cmb(self, data: Any) -> CMBOutput:  # type: ignore[override]
        raise NotImplementedError("DGPModel does not yet provide CMB outputs.")

    def is_valid(self) -> bool:
        return True

    def omega_m0(self) -> float:
        return float(self._params.Omega_m0)

    def sigma8(self) -> float:
        solver = self._ensure_growth_table()
        return float(solver.sigma8(1.0, self._sigma8_today))

    def S8(self, gamma: float = 0.5) -> float:
        om = self.omega_m0()
        if om <= 0.0:
            raise ValueError("Model returned non-positive Ω_m0; cannot build S₈.")
        return float(self.sigma8() * (om / 0.3) ** float(gamma))

    def sound_horizon(self) -> float:
        raise NotImplementedError("DGPModel does not expose a calibrated sound horizon yet.")

    def distance_modulus(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_distance_prediction(z, lambda zval: self._distance_entry(zval)[0])

    def DM(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_distance_prediction(z, lambda zval: self._distance_entry(zval)[2])

    def DA(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_distance_prediction(z, lambda zval: self._distance_entry(zval)[3])

    def DH(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_distance_prediction(z, lambda zval: self._distance_entry(zval)[4])

    def DV(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_distance_prediction(z, lambda zval: self._distance_entry(zval)[5])

    def Hubble(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._evaluate_distance_prediction(z, lambda zval: self._hubble_scalar(zval))

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
        return self._evaluate_growth_prediction(
            z,
            "fs8",
            lambda solver, a: solver.fs8(a, self._sigma8_today),
        )

    def E(self, a: float) -> float:
        return self._E(float(a))

    def dE_da(self, a: float) -> float:
        return self._dE_da(float(a))

    def dlnH_dlna(self, a: float) -> float:
        return self._dlnH_dlna(float(a))

    def mu(self, a: float) -> float:
        return self._mu_scalar(float(a))

    def rc(self) -> float:
        if self._sqrt_Omega_rc <= 0.0:
            return float("inf")
        return float(1.0 / (2.0 * self._params.H0 * self._sqrt_Omega_rc))

    def _coerce_params(self, params: Dict[str, float]) -> Dict[str, float]:
        required = ["H0", "Omega_m0", "Omega_r0", "Omega_k0", "Omega_b0", "Omega_rc"]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required DGP parameters: {missing}")
        normalized: Dict[str, float] = {key: float(params[key]) for key in required}
        normalized["epsilon_branch"] = int(params.get("epsilon_branch", 1))
        normalized["sigma8_0"] = float(params.get("sigma8_0", 0.811))
        return normalized

    def _make_simpson_integrator(self, steps: int):
        n = int(steps)
        if n <= 0:
            raise ValueError("Integrators require a positive step count.")
        if n % 2 != 0:
            n += 1

        def integrator(func: Callable[[float], float], lower: float, upper: float) -> float:
            return lcdm_utils.simpson_integral(func, lower, upper, n=n)

        return integrator

    def _distance_entry(self, z: float) -> tuple[float, float, float, float, float, float]:
        z_safe = float(z)
        chi = self._line_of_sight_distance(z_safe)
        DM = transverse_comoving_distance(chi, self._params.H0, self._params.Omega_k0)
        DA = DM / (1.0 + max(z_safe, -0.999999999))
        DL = luminosity_distance(DM, z_safe)
        mu = distance_modulus_from_luminosity_distance(DL)
        DH = C_LIGHT / self._hubble_scalar(z_safe)
        DV = self._volume_distance(z_safe, DA, DH)
        return mu, DL, DM, DA, DH, DV

    def _line_of_sight_distance(self, z: float) -> float:
        if z >= 0.0:
            integrand = lambda zp: C_LIGHT / self._hubble_scalar(zp)
            return self._distance_integrator(integrand, 0.0, z)
        integrand = lambda zp: C_LIGHT / self._hubble_scalar(zp)
        return -self._distance_integrator(integrand, z, 0.0)

    def _volume_distance(self, z: float, DA: float, DH: float) -> float:
        z_eff = max(z, 1e-8)
        if not math.isfinite(DA) or not math.isfinite(DH):
            return math.inf
        return ((1.0 + z_eff) ** 2 * DA**2 * DH * z_eff) ** (1.0 / 3.0)

    def _hubble_scalar(self, z: float) -> float:
        a = self._safe_scale_factor(z)
        return float(self._params.H0 * self._E(a))

    def _make_grid_evaluator(
        self,
        z_values: float | np.ndarray | list | tuple,
        func: Callable[[float], float],
    ) -> float | np.ndarray:
        values = np.asarray(z_values, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.ravel()
        result = np.empty_like(flat)
        for idx, raw in enumerate(flat):
            result[idx] = func(float(raw))
        shaped = result.reshape(values.shape)
        return float(shaped.item()) if scalar_input else shaped

    def _evaluate_distance_prediction(
        self,
        z_values: float | np.ndarray | list | tuple,
        func: Callable[[float], float],
    ) -> float | np.ndarray:
        return self._make_grid_evaluator(z_values, func)

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
                result[idx] = math.nan
                continue
            a = self._safe_scale_factor(raw)
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

    def _ensure_growth_table(self) -> GrowthTable:
        if self._growth_table is None:
            def rhs(a: float, y: np.ndarray) -> np.ndarray:
                return self._growth_rhs(a, y)

            self._growth_table = GrowthTable(rhs)
        return self._growth_table

    def _growth_rhs(self, a: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        D, D_prime = y
        E_a = self._E(a)
        dE_da = self._dE_da(a)
        term1 = -(3.0 / a + dE_da / E_a) * D_prime
        mu = self._mu_scalar(a)
        term2 = 1.5 * self._params.Omega_m0 / (a ** 5 * E_a**2) * mu * D
        return np.array([D_prime, term1 + term2])

    def _mu_scalar(self, a: float) -> float:
        if self._sqrt_Omega_rc <= 0.0:
            return 1.0
        beta = self._beta(a)
        if beta == 0.0:
            return float("inf")
        return 1.0 + 1.0 / (3.0 * beta)

    def _beta(self, a: float) -> float:
        if self._sqrt_Omega_rc <= 0.0:
            return 1.0
        E = self._E(a)
        dlnH = self._dlnH_dlna(a)
        return 1.0 - self._epsilon_branch * E / self._sqrt_Omega_rc * (1.0 + dlnH / 3.0)

    def _dlnH_dlna(self, a: float) -> float:
        E_a = self._E(a)
        if E_a == 0.0:
            return 0.0
        return a * self._dE_da(a) / E_a

    def _E(self, a: float) -> float:
        F = self._F(a)
        sqrt_F = math.sqrt(max(F, 0.0))
        return sqrt_F + self._epsilon_branch * self._sqrt_Omega_rc

    def _F(self, a: float) -> float:
        return (
            self._params.Omega_m0 / a**3
            + self._params.Omega_r0 / a**4
            + self._params.Omega_k0 / a**2
            + self._params.Omega_rc
        )

    def _dE_da(self, a: float) -> float:
        F = self._F(a)
        if F <= 0.0:
            return 0.0
        return 0.5 * self._dF_da(a) / math.sqrt(F)

    def _dF_da(self, a: float) -> float:
        return (
            -3.0 * self._params.Omega_m0 / a**4
            - 4.0 * self._params.Omega_r0 / a**5
            - 2.0 * self._params.Omega_k0 / a**3
        )

    @staticmethod
    def _safe_scale_factor(z_value: float) -> float:
        z_safe = max(float(z_value), -0.999999999)
        return max(1e-12, 1.0 / (1.0 + z_safe))

    def __repr__(self) -> str:
        return f"DGPModel({asdict(self._params)})"
