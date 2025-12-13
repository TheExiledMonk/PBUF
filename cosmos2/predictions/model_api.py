"""Adapter that exposes a minimal prediction-ready API for arbitrary models."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from cosmos2.kernels.common.growth import solve_growth
from cosmos2.models.lcdm.utils import C_LIGHT as C_LIGHT_KM_S
from cosmos2.parameters.central_authority import MPC_TO_KM, SECONDS_PER_GYR


class PredictionModelAdapter:
    """Wrap a model instance to expose the prediction API surface."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._constants = {
            "c_km_per_s": float(C_LIGHT_KM_S),
            "Tcmb": 2.7255,
        }
        self._lensing_adapter: PredictionLensing | None = None
        self._background_accessor: PredictionBackground | None = None
        self._matter_accessor: PredictionMatter | None = None
        self._growth_accessor: PredictionGrowth | None = None
        self._elastic_accessor: PredictionElastic | None = None

    @property
    def raw_model(self) -> Any:
        return self._model

    @property
    def parameters(self) -> dict[str, float | str]:
        raw_params = getattr(self._model, "parameters", {})
        normalized: dict[str, float | str] = {}
        if isinstance(raw_params, dict):
            for key, value in raw_params.items():
                try:
                    normalized[key] = float(value)
                except (TypeError, ValueError):
                    normalized[key] = value
        return normalized

    @property
    def constants(self) -> dict[str, float]:
        return dict(self._constants)

    @property
    def alpha(self) -> float:
        """Return the model's elastic amplitude α when available."""
        value = self.parameters.get("alpha")
        if value is not None:
            return float(value)
        candidate = getattr(self._model, "alpha", None)
        if callable(candidate):
            return float(candidate())
        if candidate is not None:
            return float(candidate)
        raise AttributeError("Underlying model does not expose alpha.")

    @property
    def background(self) -> "PredictionBackground":
        if self._background_accessor is None:
            self._background_accessor = PredictionBackground(self)
        return self._background_accessor

    @property
    def growth(self) -> "PredictionGrowth":
        if self._growth_accessor is None:
            self._growth_accessor = PredictionGrowth(self)
        return self._growth_accessor

    @property
    def elastic(self) -> "PredictionElastic":
        if self._elastic_accessor is None:
            self._elastic_accessor = PredictionElastic(self)
        return self._elastic_accessor

    @property
    def matter(self) -> "PredictionMatter":
        if self._matter_accessor is None:
            self._matter_accessor = PredictionMatter(self)
        return self._matter_accessor

    def H(self, a: float | Sequence[float]) -> np.ndarray:
        return self._evaluate_over_scale_factor(a, lambda a_val: float(self._model.Hubble(self._scale_to_redshift(a_val))))

    def temperature(self, a: float | Sequence[float]) -> np.ndarray:
        candidate = getattr(self._model, "temperature", None)
        if callable(candidate):
            return self._evaluate_over_scale_factor(a, candidate)
        # fallback to CMB scaling
        return self._evaluate_over_scale_factor(a, lambda a_val: float(self._constants["Tcmb"] / a_val))

    def H_of_z(self, z: float | Sequence[float]) -> np.ndarray:
        """Return H(z) by converting redshift to scale factor."""
        arr = np.asarray(z, dtype=float)
        if arr.size == 0:
            return arr.copy()
        safe_z = np.clip(arr, -0.999999, np.inf)
        a_vals = np.clip(1.0 / np.clip(1.0 + safe_z, 1e-9, np.inf), 1e-9, np.inf)
        return self.H(a_vals)

    def comoving_distance(self, z: float | Sequence[float]) -> np.ndarray:
        """Return the transverse comoving distance D_M(z) if the model exposes one."""
        arr = np.asarray(z, dtype=float)
        if arr.size == 0:
            return arr.copy()
        candidate = getattr(self._model, "comoving_distance", None)
        if callable(candidate):
            return np.asarray(candidate(arr), dtype=float)
        candidate = getattr(self._model, "DM", None)
        if callable(candidate):
            return np.asarray(candidate(arr), dtype=float)
        raise AttributeError("Underlying model does not expose a comoving distance.")

    def elastic_stiffness(self, a: float | Sequence[float]) -> np.ndarray:
        """Return ε₀(T) (dimensionless stiffness) exposed by the underlying model."""

        arr = self._normalize_scale_factor(a)
        if arr.size == 0:
            return arr.copy()

        flat = arr.ravel()

        def _evaluate(func: Callable[[float], float], inputs: np.ndarray) -> np.ndarray:
            values = np.empty_like(inputs, dtype=float)
            for idx, value in enumerate(inputs):
                values[idx] = float(func(float(value)))
            return values

        candidate = getattr(self._model, "elastic_stiffness", None)
        if callable(candidate):
            evaluated = _evaluate(candidate, flat)
            return evaluated.reshape(arr.shape)

        epsilon_of_T = getattr(self._model, "epsilon0_of_T", None)
        if callable(epsilon_of_T):
            temperature_vals = self.temperature(arr).ravel()
            evaluated = _evaluate(epsilon_of_T, temperature_vals)
            return evaluated.reshape(arr.shape)

        # Try PBUF-specific thermal table lookup as a final fallback.
        thermal_table = getattr(self._model, "thermal_table", None)
        if thermal_table is not None:
            try:
                from cosmos2.models.pbuf.elastic import epsilon_of_a
            except Exception:  # pragma: no cover - best-effort fallback
                epsilon_of_a = None
            if epsilon_of_a is not None:
                evaluated = _evaluate(lambda val: epsilon_of_a(val, thermal_table), flat)
                return evaluated.reshape(arr.shape)

        raise AttributeError("Underlying model does not expose an elastic stiffness surface.")

    def sound_speed(self, a: float | Sequence[float]) -> np.ndarray:
        candidate = getattr(self._model, "sound_speed", None)
        if callable(candidate):
            return self._evaluate_over_scale_factor(a, candidate)
        # default approximation: c/sqrt(3)
        return self._evaluate_over_scale_factor(a, lambda _: float(self._constants["c_km_per_s"] / np.sqrt(3)))

    def sound_horizon(self) -> float:
        candidate = getattr(self._model, "sound_horizon", None)
        if callable(candidate):
            return float(candidate())
        raise AttributeError("Underlying model does not expose 'sound_horizon'.")

    def Omega_m_of_z(self, z: float | Sequence[float]) -> np.ndarray:
        """Return Ω_m(z) using the background helper."""

        return self.background.Omega_m_of_z(z)

    def omega_m0(self) -> float:
        candidate = getattr(self._model, "omega_m0", None)
        if callable(candidate):
            return float(candidate())
        value = self.parameters.get("Omega_m0")
        if value is None:
            raise AttributeError("Underlying model does not expose Omega_m0.")
        return float(value)

    def sigma8_today(self) -> float:
        candidate = getattr(self._model, "sigma8_today", None)
        if callable(candidate):
            return float(candidate())
        candidate = getattr(self._model, "sigma8", None)
        if callable(candidate):
            return float(candidate())
        value = self.parameters.get("sigma8_0") or self.parameters.get("sigma8")
        if value is None:
            raise AttributeError("Underlying model does not expose sigma8 today.")
        return float(value)

    @property
    def lensing(self) -> "PredictionLensing":
        """Expose the model's lensing backend with a safe fallback."""
        if self._lensing_adapter is None:
            self._lensing_adapter = PredictionLensing(self)
        return self._lensing_adapter

    # ------------------
    # Internal helpers
    # ------------------
    def _normalize_scale_factor(self, a: float | Sequence[float]) -> np.ndarray:
        arr = np.asarray(a, dtype=float)
        if arr.size == 0:
            return np.array([], dtype=float)
        return np.clip(arr, 1e-9, np.inf)

    def _scale_to_redshift(self, a: np.ndarray) -> np.ndarray:
        return np.clip((1.0 / a) - 1.0, -0.999999, np.inf)

    def _compute_deceleration(self, a: np.ndarray, H_vals: np.ndarray) -> np.ndarray:
        if H_vals.size == 0:
            return H_vals
        derivative = np.gradient(H_vals, a, edge_order=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = -1.0 - (a / np.where(H_vals == 0.0, np.nan, H_vals)) * derivative
        return np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    def _evaluate_over_scale_factor(
        self, a: float | Sequence[float], func: Callable[[float], float]
    ) -> np.ndarray:
        arr = self._normalize_scale_factor(a)
        flattened = arr.flatten()
        values = [float(func(float(val))) for val in flattened]
        return np.array(values, dtype=float).reshape(arr.shape)


class PredictionBackground:
    """Helper that exposes the background API expected by prediction modules."""

    def __init__(self, adapter: PredictionModelAdapter) -> None:
        self._adapter = adapter

    def _normalize_scale_factor(self, a: float | Sequence[float]) -> np.ndarray:
        return self._adapter._normalize_scale_factor(a)

    def __call__(self, a: float | Sequence[float]) -> dict[str, np.ndarray]:
        a_arr = self._adapter._normalize_scale_factor(a)
        H_vals = self._adapter.H(a_arr)
        q_vals = self._adapter._compute_deceleration(a_arr, H_vals)
        return {
            "a": a_arr,
            "z": self._adapter._scale_to_redshift(a_arr),
            "H": H_vals,
            "q": q_vals,
        }

    def H(self, z: float | Sequence[float]) -> np.ndarray:
        return self._adapter.H_of_z(z)

    def D_A(self, z: float | Sequence[float]) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        if arr.size == 0:
            return arr.copy()
        model = self._adapter.raw_model
        candidate = (
            getattr(model, "DA", None)
            or getattr(model, "D_A", None)
            or getattr(model, "angular_diameter_distance", None)
        )
        if not callable(candidate):
            raise AttributeError("Underlying model does not expose an angular diameter distance.")
        values = candidate(arr)
        return np.asarray(values, dtype=float)

    def comoving_distance(self, z: float | Sequence[float]) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        if arr.size == 0:
            return arr.copy()
        return self._adapter.comoving_distance(arr)

    def c_value(self) -> float:
        constants = self._adapter.constants
        value = constants.get("c_km_per_s")
        if value is None:
            raise AttributeError("Speed of light constant unavailable.")
        return float(value)

    def E(self, a: float | Sequence[float]) -> np.ndarray:
        arr = self._normalize_scale_factor(a)
        if arr.size == 0:
            return arr.copy()
        H_vals = self._adapter.H(arr)
        if H_vals.shape != arr.shape:
            raise RuntimeError("H(a) grid size mismatch while evaluating E(a).")
        parameters = self._adapter.parameters
        H0 = float(parameters.get("H0", 67.4))
        if H0 <= 0.0:
            raise ValueError("Model reports non-positive H0.")
        return np.asarray(H_vals, dtype=float) / H0

    def Omega_m_of_a(self, a: float | Sequence[float]) -> np.ndarray:
        arr = self._normalize_scale_factor(a)
        if arr.size == 0:
            return arr.copy()
        H_vals = self._adapter.H(arr)
        if H_vals.shape != arr.shape:
            raise RuntimeError("H(a) grid size mismatch while evaluating Omega_m(a).")
        parameters = self._adapter.parameters
        H0 = float(parameters.get("H0", 67.4))
        if H0 <= 0.0:
            raise ValueError("Model reports non-positive H0.")
        omega_m0 = parameters.get("Omega_m0")
        if omega_m0 is None:
            raise AttributeError("Underlying model does not expose Omega_m0.")
        E_vals = np.asarray(H_vals, dtype=float) / H0
        safe_E = np.clip(E_vals, 1e-12, np.inf)
        power = np.power(arr, -3.0, dtype=float)
        ratio = float(omega_m0) * power / (safe_E * safe_E)
        return np.asarray(ratio, dtype=float)

    def Omega_m_of_z(self, z: float | Sequence[float]) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        a_vals = np.clip(1.0 / np.clip(1.0 + arr, 1e-9, np.inf), 1e-9, np.inf)
        return self.Omega_m_of_a(a_vals)

    def Omega_m_a(self, a: float | Sequence[float]) -> np.ndarray:
        arr = self._normalize_scale_factor(a)
        if arr.size == 0:
            return arr.copy()
        return self.Omega_m_of_a(arr)

    def get_time_conversion_to_Gyr(self) -> float:
        """Return a scalar factor that converts 1/H(z) into gigayears."""

        factor = MPC_TO_KM / SECONDS_PER_GYR
        return float(factor)


class PredictionGrowth:
    """Growth solver helper for prediction modules."""

    def __init__(self, adapter: PredictionModelAdapter) -> None:
        self._adapter = adapter

    def solve_growth(self, a: float | Sequence[float]) -> np.ndarray:
        arr = self._adapter._normalize_scale_factor(a)
        if arr.size == 0:
            return arr.copy()
        E_vals = np.asarray(self._adapter.background.E(arr), dtype=float)
        if E_vals.shape != arr.shape:
            raise RuntimeError("E(a) grid size mismatch while solving growth.")
        omega_m0 = float(self._adapter.omega_m0())
        sorted_idx = np.argsort(arr)
        sorted_a = arr[sorted_idx]
        sorted_E = E_vals[sorted_idx]
        D_sorted, _ = solve_growth(sorted_a, sorted_E, omega_m0=omega_m0)
        D = np.empty_like(arr)
        D[sorted_idx] = D_sorted
        return D


class PredictionElastic:
    """Elastic surface helper exposing omega_sigma for prediction modules."""

    def __init__(self, adapter: PredictionModelAdapter) -> None:
        self._adapter = adapter
        self._evaluator = self._resolve_evaluator()

    def omega_sigma(self, a: float | Sequence[float]) -> np.ndarray:
        arr = self._adapter._normalize_scale_factor(a)
        if arr.size == 0:
            return arr.copy()
        values = self._evaluator(arr)
        if values.shape != arr.shape:
            values = values.reshape(arr.shape)
        return values

    def _resolve_evaluator(self) -> Callable[[np.ndarray], np.ndarray]:
        raw = self._adapter.raw_model
        elastic_surface = getattr(raw, "elastic", None)
        if elastic_surface is not None:
            omega = getattr(elastic_surface, "omega_sigma", None)
            if callable(omega):
                return lambda arr: self._evaluate_floatwise(omega, arr)

        direct = getattr(raw, "omega_sigma", None)
        if callable(direct):
            return lambda arr: self._evaluate_floatwise(direct, arr)

        params = getattr(raw, "_params", None)
        table = getattr(raw, "thermal_table", None)
        if params is not None and table is not None:
            try:
                from cosmos2.models.pbuf.elastic import omega_sigma_of_a
            except Exception:
                pass
            else:
                return lambda arr: self._evaluate_floatwise(
                    lambda value: omega_sigma_of_a(value, params, table), arr
                )

        return lambda arr: np.zeros_like(arr, dtype=float)

    @staticmethod
    def _evaluate_floatwise(func: Callable[[float], float], arr: np.ndarray) -> np.ndarray:
        flat = arr.ravel()
        result = np.empty_like(flat, dtype=float)
        for idx, value in enumerate(flat):
            result[idx] = float(func(float(value)))
        return result.reshape(arr.shape)


class PredictionMatter:
    """Surface that proxies matter observables exposed by the underlying model."""

    def __init__(self, adapter: PredictionModelAdapter) -> None:
        self._adapter = adapter

    def _backend(self) -> Any:
        raw = self._adapter.raw_model
        matter = getattr(raw, "matter", None)
        return matter if matter is not None else raw

    def power_spectrum(self, k_array: Sequence[float], z: float) -> np.ndarray:
        backend = self._backend()
        compute = getattr(backend, "power_spectrum", None)
        if not callable(compute):
            raise AttributeError("Underlying model does not expose matter.power_spectrum().")
        grid = np.asarray(k_array, dtype=float)
        result = compute(grid, float(z))
        return np.asarray(result, dtype=float)

    def pk_config(self) -> dict[str, Any]:
        backend = self._backend()
        candidate = getattr(backend, "pk_config", None)
        if not callable(candidate):
            raise AttributeError("Underlying model does not expose matter.pk_config().")
        config = candidate()
        if isinstance(config, dict):
            return dict(config)
        if config is None:
            return {}
        return dict(config)


class PredictionLensing:
    """Wrapped lensing backend that provides safe fallbacks."""

    _MIN_ELL = 2.0

    def __init__(self, adapter: PredictionModelAdapter) -> None:
        self._adapter = adapter
        self.backend_source: str | None = None

    def compute_cmb_kappa(self, ell: Sequence[float]) -> np.ndarray:
        arr = np.asarray(ell, dtype=float)
        if arr.size == 0:
            self.backend_source = "empty"
            return arr.copy()

        raw = self._adapter.raw_model
        backend = getattr(raw, "lensing", None)
        if backend is not None:
            compute = getattr(backend, "compute_cmb_kappa", None)
            if callable(compute):
                try:
                    values = np.asarray(compute(arr), dtype=float)
                except Exception:
                    self.backend_source = "fallback"
                    return self._approximate(arr)
                if values.shape != arr.shape:
                    raise RuntimeError("Lensing backend returned data with unexpected shape.")
                self.backend_source = getattr(backend, "backend_name", "model")
                return values

        self.backend_source = "fallback"
        return self._approximate(arr)

    def _approximate(self, ell: np.ndarray) -> np.ndarray:
        safe_ell = np.clip(ell, self._MIN_ELL, np.inf)
        sigma8 = self._safe_sigma8()
        omega_m = self._safe_omega_m()
        h0 = self._safe_H0()
        amplitude = (omega_m * sigma8) ** 2 * ((h0 / 70.0) ** 4)
        scale = 300.0
        profile = amplitude * np.exp(-safe_ell / 1500.0) / (1.0 + (safe_ell / scale) ** 2)
        return np.clip(profile, 0.0, np.inf)

    def _safe_sigma8(self) -> float:
        try:
            return max(float(self._adapter.sigma8_today()), 1e-4)
        except Exception:
            return max(float(self._adapter.parameters.get("sigma8_0", 0.8)), 1e-4)

    def _safe_omega_m(self) -> float:
        try:
            return max(float(self._adapter.omega_m0()), 1e-4)
        except Exception:
            return max(float(self._adapter.parameters.get("Omega_m0", 0.3)), 1e-4)

    def _safe_H0(self) -> float:
        return max(float(self._adapter.parameters.get("H0", 70.0)), 1e-3)
