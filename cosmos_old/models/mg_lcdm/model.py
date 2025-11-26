"""MG-ΛCDM cosmology that reuses the LCDM background but modifies growth and lensing."""

from __future__ import annotations

import warnings
from dataclasses import asdict
from typing import Any, Callable, Dict, Tuple

import numpy as np

from cosmos.interfaces import CMBOutput, CosmologyModel
from cosmos.models.common.growth import GrowthTable
from cosmos.models.lcdm.distances import E
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.mg_lcdm.params import MGLCDMParams


class MGLCDMModel(CosmologyModel):
    """Phenomenological modified-gravity variant of ΛCDM."""

    def __init__(self, **params: float) -> None:
        coerced = self._coerce_params(params)
        base_params = {
            key: coerced[key]
            for key in ("H0", "Omega_m0", "Omega_r0", "Omega_k0", "Omega_b0")
        }
        if "Omega_lambda0" in coerced:
            base_params["Omega_lambda0"] = coerced["Omega_lambda0"]
        base_params["sigma8_0"] = coerced["sigma8_0"]

        self._base_model = LCDMModel(**base_params)
        self._lcdm_params = self._base_model.params

        mg_params = MGLCDMParams(**{**coerced, "Omega_lambda0": float(self._lcdm_params.Omega_lambda0)})
        self._params = mg_params
        self._parameters = dict(asdict(self._params))
        self._mu0 = float(self._params.mu0)
        self._Sigma0 = float(self._params.Sigma0)
        self._sigma8_today = float(self._params.sigma8_0)

        self._growth_table: GrowthTable | None = None
        self._growth_cache: Dict[Tuple[str, float], float] = {}

        self._omega_de_today = self._omega_de_scalar(1.0)
        self._warned_negative_mu = False
        self._warned_negative_sigma = False

    @property
    def parameters(self) -> Dict[str, float]:
        return dict(self._parameters)

    def cmb(self, data: Any) -> CMBOutput:
        return self._base_model.cmb(data)

    def omega_m0(self) -> float:
        return self._base_model.omega_m0()

    def sigma8(self) -> float:
        solver = self._ensure_growth_table()
        return float(solver.sigma8(1.0, self._sigma8_today))

    def S8(self, gamma: float = 0.5) -> float:
        om = self.omega_m0()
        if om <= 0.0:
            raise ValueError("Model returned non-positive Ω_m0; cannot build S₈.")
        return float(self.sigma8() * (om / 0.3) ** float(gamma))

    def is_valid(self) -> bool:
        return self._base_model.is_valid()

    def distance_modulus(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._base_model.distance_modulus(z)

    def DV(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._base_model.DV(z)

    def DM(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._base_model.DM(z)

    def DA(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._base_model.DA(z)

    def DH(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._base_model.DH(z)

    def Hubble(self, z: float | np.ndarray | list | tuple) -> float | np.ndarray:
        return self._base_model.Hubble(z)

    def sound_horizon(self) -> float:
        return self._base_model.sound_horizon()

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

    def omega_m(self, a: float | np.ndarray) -> float | np.ndarray:
        Omega_m, _, _, _ = self._omega_components(a)
        return Omega_m

    def omega_r(self, a: float | np.ndarray) -> float | np.ndarray:
        _, Omega_r, _, _ = self._omega_components(a)
        return Omega_r

    def omega_k(self, a: float | np.ndarray) -> float | np.ndarray:
        _, _, Omega_k, _ = self._omega_components(a)
        return Omega_k

    def omega_de(self, a: float | np.ndarray) -> float | np.ndarray:
        _, _, _, Omega_de = self._omega_components(a)
        return Omega_de

    def mu(self, a: float | np.ndarray) -> float | np.ndarray:
        return self._evaluate_scale_dependent(a, lambda scale: self._mu_scalar(scale))

    def Sigma(self, a: float | np.ndarray) -> float | np.ndarray:
        return self._evaluate_scale_dependent(a, lambda scale: self._Sigma_scalar(scale))

    def _ensure_growth_table(self) -> GrowthTable:
        if self._growth_table is None:
            def rhs(a: float, y: np.ndarray) -> np.ndarray:
                return self._growth_rhs(a, y)

            self._growth_table = GrowthTable(rhs)
        return self._growth_table

    def _growth_rhs(self, a: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        D, D_prime = y
        eps = 1e-5
        E_a = self._E(a)
        a_minus = max(a - eps, 1e-8)
        E_a_plus = self._E(a + eps)
        E_a_minus = self._E(a_minus)
        dE_da = (E_a_plus - E_a_minus) / (2.0 * eps)
        term1 = -(3.0 / a + dE_da / E_a) * D_prime
        term2 = 1.5 * self._lcdm_params.Omega_m0 / (a**5 * E_a**2) * self._mu_scalar(a) * D
        return np.array([D_prime, term1 + term2])

    def _E(self, a: float) -> float:
        return float(E(a, self._lcdm_params))

    def _omega_components(self, a_values: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float, float]:
        arr = np.asarray(a_values, dtype=float)
        scalar = arr.ndim == 0
        safe = np.where(arr <= 0.0, np.nextafter(0.0, 1.0), arr)
        params = self._lcdm_params
        Omega_lambda0 = float(
            params.Omega_lambda0
            if params.Omega_lambda0 is not None
            else 1.0 - params.Omega_m0 - params.Omega_r0 - params.Omega_k0
        )
        Om = params.Omega_m0 / safe**3
        Or = params.Omega_r0 / safe**4
        Ok = params.Omega_k0 / safe**2
        Ol = Omega_lambda0
        E2 = Om + Or + Ok + Ol
        E2_safe = np.where(E2 <= 0.0, np.finfo(float).tiny, E2)
        Omega_m = Om / E2_safe
        Omega_r = Or / E2_safe
        Omega_k = Ok / E2_safe
        Omega_de = 1.0 - (Omega_m + Omega_r + Omega_k)
        if scalar:
            return (
                float(Omega_m),
                float(Omega_r),
                float(Omega_k),
                float(Omega_de),
            )
        return Omega_m, Omega_r, Omega_k, Omega_de

    def _omega_de_scalar(self, a: float) -> float:
        value = self.omega_de(a)
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)

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

    @staticmethod
    def _safe_scale_factor(z_value: float) -> float:
        z_safe = max(float(z_value), -0.999999999)
        return 1.0 / (1.0 + z_safe)

    def _evaluate_scale_dependent(
        self,
        raw_values: float | np.ndarray,
        callback: Callable[[float], float],
    ) -> float | np.ndarray:
        values = np.asarray(raw_values, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.ravel()
        result = np.empty_like(flat)
        for idx, raw in enumerate(flat):
            if not np.isfinite(raw):
                result[idx] = np.nan
                continue
            result[idx] = callback(float(raw))
        shaped = result.reshape(values.shape)
        return float(shaped.item()) if scalar_input else shaped

    def _mu_scalar(self, a: float) -> float:
        if self._mu0 == 0.0 or self._omega_de_today <= 0.0:
            return 1.0
        ratio = self._omega_de_scalar(a) / self._omega_de_today
        value = 1.0 + self._mu0 * ratio
        if value <= 0.0 and not self._warned_negative_mu:
            warnings.warn(
                "MG μ(a) became non-positive; results may be unphysical.",
                stacklevel=2,
            )
            self._warned_negative_mu = True
        return float(value)

    def _Sigma_scalar(self, a: float) -> float:
        if self._Sigma0 == 0.0 or self._omega_de_today <= 0.0:
            return 1.0
        ratio = self._omega_de_scalar(a) / self._omega_de_today
        value = 1.0 + self._Sigma0 * ratio
        if value <= 0.0 and not self._warned_negative_sigma:
            warnings.warn(
                "MG Σ(a) became non-positive; results may be unphysical.",
                stacklevel=2,
            )
            self._warned_negative_sigma = True
        return float(value)

    def __repr__(self) -> str:
        return f"MGLCDMModel({asdict(self._params)})"

    def _coerce_params(self, params: Dict[str, float]) -> Dict[str, float]:
        required = ["H0", "Omega_m0", "Omega_r0", "Omega_k0", "Omega_b0"]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required MG-LCDM parameters: {missing}")
        normalized: Dict[str, float] = {key: float(params[key]) for key in required}
        if "Omega_lambda0" in params:
            normalized["Omega_lambda0"] = float(params["Omega_lambda0"])
        normalized["sigma8_0"] = float(params.get("sigma8_0", 0.811))
        normalized["mu0"] = float(params.get("mu0", 0.0))
        normalized["Sigma0"] = float(params.get("Sigma0", 0.0))
        return normalized
