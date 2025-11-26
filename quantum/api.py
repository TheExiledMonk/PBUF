"""Bridge between Cosmos and the full Quantum + E₀ engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .engine import run_quantum_engine as _legacy_engine
from .paths import quantum_root, resolve_path, repo_root
from .thermal.fitter import (
    DEFAULT_FIT_MAX,
    DEFAULT_FIT_MIN,
    DEFAULT_FIT_POINTS,
    DEFAULT_FIT_SAMPLES,
    MicrophysicsInputs,
    ThermalFitResult,
    derive_thermal_params,
)


CONFIG_ROOT = repo_root() / "configs" / "quantum"
CONFIG_PATH = CONFIG_ROOT / "config.json"
DEFAULT_T_MIN = 2.7255
DEFAULT_T_MAX = 1.0e12
DEFAULT_POINTS = 256
DEFAULT_DENSE_POINTS = 24
DEFAULT_TABLE_VERSION = 12
DEFAULT_METHOD_VERSION = 12
DEFAULT_REGULATOR = "hard_cutoff"
DEFAULT_FIELD_CONTENT = "SM_full"
DEFAULT_THERMAL_MODE = "exp"


@dataclass(frozen=True)
class ThermalSpec:
    """Lightweight configuration describing the Quantum→Cosmos handoff."""

    regulator: str = DEFAULT_REGULATOR
    field_content: str = DEFAULT_FIELD_CONTENT
    thermal_mode: str = DEFAULT_THERMAL_MODE
    temperature_points: int = DEFAULT_POINTS
    dense_points: int = DEFAULT_DENSE_POINTS
    t_min: float = DEFAULT_T_MIN
    t_max: float = DEFAULT_T_MAX
    table_version: int = DEFAULT_TABLE_VERSION
    method_version: int = DEFAULT_METHOD_VERSION
    fit_samples: int = DEFAULT_FIT_SAMPLES
    fit_min: float = DEFAULT_FIT_MIN
    fit_max: float = DEFAULT_FIT_MAX
    fit_points: int = DEFAULT_FIT_POINTS
    engine_override_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ThermalSpec":
        iterations = payload.get("temperature_points") or payload.get("iterations") or DEFAULT_POINTS
        dense = int(payload.get("dense_points", DEFAULT_DENSE_POINTS))
        t_min = float(payload.get("t_min", DEFAULT_T_MIN))
        t_max = float(payload.get("t_max", DEFAULT_T_MAX))
        if t_max <= t_min:
            raise ValueError("t_max must exceed t_min.")

        fit_samples = int(payload.get("fit_samples", DEFAULT_FIT_SAMPLES))
        fit_min = float(payload.get("fit_min", DEFAULT_FIT_MIN))
        fit_max = float(payload.get("fit_max", DEFAULT_FIT_MAX))
        fit_points = int(payload.get("fit_points", DEFAULT_FIT_POINTS))
        if fit_samples < 4:
            raise ValueError("fit_samples must be >= 4.")
        if fit_points < 4:
            raise ValueError("fit_points must be >= 4.")

        override_raw = payload.get("engine_override_path")
        override_path = resolve_path(override_raw) if override_raw else None

        return cls(
            temperature_points=max(int(iterations), 32),
            dense_points=max(dense, 0),
            t_min=t_min,
            t_max=t_max,
            table_version=int(payload.get("table_version", DEFAULT_TABLE_VERSION)),
            method_version=int(payload.get("method_version", DEFAULT_METHOD_VERSION)),
            regulator=str(payload.get("regulator", DEFAULT_REGULATOR)),
            field_content=str(payload.get("field_content", DEFAULT_FIELD_CONTENT)),
            thermal_mode=str(payload.get("thermal_mode", DEFAULT_THERMAL_MODE)).lower(),
            fit_samples=fit_samples,
            fit_min=fit_min,
            fit_max=fit_max,
            fit_points=fit_points,
            engine_override_path=override_path,
        )


def _load_spec() -> ThermalSpec:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Quantum config not found at {CONFIG_PATH}")
    return ThermalSpec.from_dict(json.loads(CONFIG_PATH.read_text()))


def _serialize_path(path: Optional[Path]) -> Optional[str]:
    return str(path) if path is not None else None


def _build_microphysics(raw: Dict[str, Any], spec: ThermalSpec, fit: ThermalFitResult, regulator: str, field_content: str) -> Dict[str, Any]:
    derived = raw.get("derived_parameters") or {}
    metadata = raw.get("run_metadata") or {}
    stats = metadata.get("stats") or {}

    micro = {
        "eps0_base": float(raw["eps0"]),
        "eps0_error": raw.get("eps0_error"),
        "alpha_qm": float(raw["alpha_QM"]),
        "alpha_error": raw.get("alpha_error"),
        "f_cut": derived.get("f_cut"),
        "f_coup": derived.get("f_coup"),
        "mixing_strength": derived.get("mixing_strength"),
        "regulator": regulator,
        "field_content": field_content,
        "thermal_mode": spec.thermal_mode,
        "beta": fit.beta,
        "T_star": fit.t_star,
        "power_index": fit.power,
        "temperature_points": spec.temperature_points,
        "iterations": spec.temperature_points,
        "dense_points": spec.dense_points,
        "t_min": spec.t_min,
        "t_max": spec.t_max,
        "table_version": spec.table_version,
        "method_version": spec.method_version,
        "engine_override_path": _serialize_path(spec.engine_override_path),
        "engine_source": raw.get("source"),
        "run_metadata": metadata,
        "time_window": stats.get("time_window"),
        "k_eps": stats.get("k_eps"),
        "threads": stats.get("threads"),
    }
    return micro


def run_quantum_engine() -> Dict[str, Any]:
    """
    Execute the full Quantum engine and enrich the result with thermal knobs.
    """

    spec = _load_spec()
    raw = _legacy_engine(spec.engine_override_path)
    derived = raw.get("derived_parameters") or {}
    regulator = str(derived.get("regulator", spec.regulator))
    field_content = str(derived.get("field_set", spec.field_content))
    fit = derive_thermal_params(
        MicrophysicsInputs(
            eps0_today=float(raw["eps0"]),
            alpha_qm=float(raw["alpha_QM"]),
            regulator=regulator,
            field_content=field_content,
            f_coup=derived.get("f_coup"),
            mixing_strength=derived.get("mixing_strength"),
        ),
        t_min=spec.t_min,
        t_max=spec.t_max,
        samples=spec.fit_samples,
        fit_min=spec.fit_min,
        fit_max=spec.fit_max,
        fit_points=spec.fit_points,
    )
    return _build_microphysics(raw, spec, fit, regulator, field_content)


__all__ = ["run_quantum_engine"]
