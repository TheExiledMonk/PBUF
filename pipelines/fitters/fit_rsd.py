#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Growth-rate (fσ₈) fitter mirroring the SN/BAO/CMB pipelines.

The fitter orchestrates loading the derived RSD catalogue, evaluating the
background model via the shared kernels in ``core/``, and optionally
optimising the supplied parameter vector. All predictions defer to the
existing cosmology modules; no new parameter names are introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas may be unavailable in some envs
    pd = None  # type: ignore[assignment]

try:
    from scipy.optimize import minimize
    from scipy.stats import chi2 as chi2_dist
except ImportError:  # pragma: no cover - SciPy is optional at runtime
    minimize = None  # type: ignore[assignment]
    chi2_dist = None  # type: ignore[assignment]

from core import gr_models, pbuf_models
from utils import logging as log
from utils.io import read_latest_result, read_yaml, write_json_atomic
from utils import parameters as param_utils

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DATASET_TOKEN = "RSD_joint"
TRANSFORM_VERSION_FALLBACK = "rsd_prepare_v1"

DEFAULT_PARAMS = {
    "lcdm": {
        "H0": 67.4,
        "Om0": 0.315,
        "Or0": 0.0,
        "Ok0": 0.0,
        "sigma8": 0.811,
    },
    "pbuf": {
        "H0": 67.4,
        "Om0": 0.315,
        "Or0": 0.0,
        "Ok0": 0.0,
        "alpha": 5.0e-4,
        "Rmax": 1.0e9,
        "eps0": 0.7,
        "n_eps": 0.0,
        "k_sat": 1.0,
        "sigma8": 0.811,
    },
}

FREE_PARAMS = {
    "lcdm": ("Om0", "sigma8"),
    "pbuf": ("Om0", "sigma8", "k_sat"),
}

MODEL_REGISTRY = {"lcdm": gr_models, "pbuf": pbuf_models}
SIGMA8_KEYS = ("sigma8", "sigma8_0", "sigma8_today", "Sigma8", "sigma_8")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _run_id(tag: str) -> str:
    return f"{tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # pragma: no cover - git may be unavailable
        return "n/a"


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised cumulative trapezoidal rule."""
    if y.size != x.size:
        raise ValueError("Input arrays for cumulative trapezoid must have matching lengths.")
    if y.size < 2:
        return np.zeros_like(y, dtype=float)
    dx = np.diff(x, axis=0)
    ave = 0.5 * (y[1:] + y[:-1])
    csum = np.cumsum(dx * ave, axis=0)
    return np.concatenate([np.zeros_like(y[:1], dtype=float), csum], axis=0)


def _coerce_float(name: str, value: object) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Parameter '{name}' expects a numeric value, received '{value}'.") from exc
    raise TypeError(f"Parameter '{name}' expects a numeric value, received type {type(value).__name__}.")


def _jsonify(obj: object) -> object:
    """Recursively convert numpy/Python scalars into JSON-serialisable objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------
def _normalise_param_payload(payload: Mapping[str, object]) -> Tuple[Dict[str, object], Tuple[str, ...]]:
    """
    Convert the user-supplied JSON payload into a parameter dictionary and free list.

    Entries may be either raw numbers or dictionaries with ``{"value": x, "fixed": bool}``.
    """

    params: MutableMapping[str, object] = {}
    fixed: set[str] = set()
    explicit_free: set[str] = set()

    for key, raw in payload.items():
        if isinstance(raw, Mapping):
            if "value" not in raw:
                raise KeyError(f"Parameter '{key}' dictionary must include a 'value' field.")
            params[key] = raw["value"]
            flag = raw.get("fixed")
            if flag is True:
                fixed.add(str(key))
            elif flag is False:
                explicit_free.add(str(key))
        else:
            params[key] = raw

    for key, value in list(params.items()):
        if isinstance(value, (int, float, np.floating, str)):
            try:
                params[key] = _coerce_float(str(key), value)
            except (TypeError, ValueError):
                params[key] = value

    free_names: list[str] = []
    for key, value in params.items():
        if key in fixed:
            continue
        if isinstance(value, (int, float, np.floating)):
            free_names.append(str(key))
        elif key in explicit_free:
            raise TypeError(f"Parameter '{key}' marked free but value is non-numeric.")

    return dict(params), tuple(free_names)


def _parse_params(path: Path) -> Tuple[Dict[str, object], Tuple[str, ...]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise TypeError(f"Parameter file '{path}' must contain a JSON object.")
    for key in ("parameters", "params"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    return _normalise_param_payload(payload)


def _load_bounds(
    free_names: Sequence[str],
    settings_bounds: Mapping[str, Sequence[float]],
    overrides: Mapping[str, Sequence[float]] | None = None,
) -> Tuple[Tuple[float | None, float | None], ...]:
    bounds_map: Dict[str, Sequence[float]] = dict(settings_bounds)
    if overrides:
        bounds_map.update({str(k): v for k, v in overrides.items()})
    sigma8_defaults = {key: 0.0 for key in SIGMA8_KEYS}

    bounds: list[Tuple[float | None, float | None]] = []
    for name in free_names:
        spec = bounds_map.get(name)
        if spec is None:
            lower = sigma8_defaults.get(name)
            bounds.append((lower, None) if lower is not None else (None, None))
        elif len(spec) >= 2:
            bounds.append((float(spec[0]), float(spec[1])))
        elif len(spec) == 1:
            bounds.append((float(spec[0]), None))
        else:
            bounds.append((None, None))
    return tuple(bounds)


def _initial_parameters(
    model: str,
    params_path: Path | None,
) -> Tuple[Dict[str, object], Tuple[str, ...], Path | None]:
    model_key = model.upper()
    canonical_block = param_utils.canonical_parameters(model_key)
    params: Dict[str, object] = dict(canonical_block)
    for key, value in DEFAULT_PARAMS[model].items():
        params.setdefault(key, value)
    free_names: List[str] = list(FREE_PARAMS.get(model, ()))
    params_file: Path | None = None

    if params_path is not None:
        overrides, explicit_free = _parse_params(params_path)
        params.update(overrides)
        params_file = params_path
        if explicit_free:
            free_names = list(explicit_free)

    if not free_names:
        free_names = [
            name
            for name, value in params.items()
            if isinstance(value, (int, float, np.floating))
        ]

    calibration = read_latest_result(model=model_key, kind="CMB")
    if calibration:
        log.info("Loaded CMB calibration (%s)", calibration.get("_source_path"))

    for key, value in list(params.items()):
        if isinstance(value, (int, float, np.floating, str)):
            try:
                params[key] = _coerce_float(key, value)
            except (TypeError, ValueError):
                params[key] = value

    return params, tuple(free_names), params_file


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _load_rsd_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    csv_path = data_dir / "rsd_index.csv"
    cov_path = data_dir / "rsd_index.cov.npy"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing growth-rate catalogue: {csv_path}")
    if not cov_path.exists():
        raise FileNotFoundError(f"Missing covariance matrix: {cov_path}")

    if pd is not None:
        table = pd.read_csv(csv_path)
        z = table["z"].to_numpy(dtype=float)
        fs8 = table["fs8"].to_numpy(dtype=float)
        sigma = table["sigma_fs8"].to_numpy(dtype=float)
    else:  # pragma: no cover - fallback path
        raw = np.genfromtxt(csv_path, delimiter=",", names=True)
        columns = {name.lower(): raw[name] for name in raw.dtype.names}
        z = np.asarray(columns.get("z"), dtype=float)
        fs8 = np.asarray(columns.get("fs8"), dtype=float)
        sigma = np.asarray(columns.get("sigma_fs8"), dtype=float)

    if z.ndim != 1:
        raise ValueError("Growth-rate catalogue must be one-dimensional.")

    cov = np.load(cov_path)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 1:
        cov = np.diag(cov)
    if cov.shape != (z.size, z.size):
        raise ValueError(f"Covariance matrix shape {cov.shape} incompatible with data length {z.size}.")

    meta_path = data_dir / "rsd_index.meta.json"
    meta: Dict[str, object] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            log.warning("Failed to parse %s; continuing without metadata.", meta_path)

    meta.setdefault("transform_version", meta.get("transform_version", TRANSFORM_VERSION_FALLBACK))
    return z, fs8, {"sigma": sigma, "cov": cov, "meta": meta}


# ---------------------------------------------------------------------------
# Growth predictions
# ---------------------------------------------------------------------------
def _extract_sigma8(params: Mapping[str, object]) -> float:
    for key in SIGMA8_KEYS:
        if key in params:
            return _coerce_float(key, params[key])
    keys = ", ".join(SIGMA8_KEYS)
    raise KeyError(f"Parameter set must include sigma8 (one of: {keys}).")


def _growth_factors(
    z_eval: np.ndarray,
    params: Mapping[str, object],
    model_module,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute linear growth factor D₊(a) and f = d ln D / d ln a.

    A standard integral approximation is used:
        D(a) ∝ H(a) ∫₀ᵃ da' / (a'^3 H(a')³)
    Normalised such that D(a=1) = 1.
    """

    a_eval = 1.0 / (1.0 + np.asarray(z_eval, dtype=float))
    if a_eval.size == 0:
        raise ValueError("At least one evaluation redshift is required for growth computation.")
    min_eval = float(np.min(a_eval))
    grid_start = min(1.0e-4, 0.5 * min_eval)
    grid_start = max(grid_start, 1.0e-6)
    grid_points = 4096 if grid_start < 1.0e-4 else 2048
    grid = np.linspace(grid_start, 1.0, grid_points)
    z_grid = 1.0 / np.clip(grid, 1.0e-12, None) - 1.0
    om0 = max(_coerce_float("Om0", params.get("Om0")), 0.0)

    e2 = np.asarray(model_module.E2(z_grid, params), dtype=float)
    e = np.sqrt(np.clip(e2, 1.0e-12, None))

    integrand = 1.0 / (grid ** 3 * e ** 3)
    integral = _cumtrapz(integrand, grid)

    prefactor = 2.5 * om0 * e
    d_plus = prefactor * integral
    d_plus = np.clip(d_plus, 0.0, None)
    normalisation = d_plus[-1] if d_plus[-1] != 0.0 else 1.0
    d_plus = d_plus / normalisation

    d_da = np.gradient(d_plus, grid, edge_order=2)
    safe_d = np.clip(d_plus, 1.0e-12, None)
    f_growth = np.clip(grid * d_da / safe_d, 0.0, None)

    d_interp = np.interp(a_eval, grid, d_plus, left=d_plus[0], right=1.0)
    f_interp = np.interp(a_eval, grid, f_growth, left=f_growth[0], right=f_growth[-1])
    f_interp = np.clip(f_interp, 0.0, None)
    return d_interp, f_interp


def _fsigma8_prediction(z: np.ndarray, params: Mapping[str, object], model_module) -> np.ndarray:
    if hasattr(model_module, "fsigma8_of_z"):
        try:
            return np.asarray(model_module.fsigma8_of_z(z, params), dtype=float)
        except TypeError:
            return np.asarray(model_module.fsigma8_of_z(z, **params), dtype=float)  # type: ignore[arg-type]

    sigma8_0 = _extract_sigma8(params)
    d_plus, f_growth = _growth_factors(z, params, model_module)
    return np.asarray(f_growth * sigma8_0 * d_plus, dtype=float)


# ---------------------------------------------------------------------------
# χ² utilities
# ---------------------------------------------------------------------------
def _chi2(residuals: np.ndarray, cov: np.ndarray) -> float:
    inv = np.linalg.pinv(cov, hermitian=True)
    return float(residuals.T @ inv @ residuals)


def _chi2_metrics(chi2_val: float, n_points: int, n_free: int) -> Dict[str, float | int | None]:
    dof = max(int(n_points - n_free), 0)
    chi2_dof = chi2_val / dof if dof > 0 else math.nan
    aic = chi2_val + 2.0 * n_free
    bic = chi2_val + n_free * math.log(max(n_points, 1))
    if chi2_dist is not None and dof > 0:
        p_value = float(chi2_dist.sf(chi2_val, dof))
    else:  # pragma: no cover - SciPy fallback
        mean = dof
        variance = 2.0 * dof if dof > 0 else math.nan
        if variance <= 0.0 or not np.isfinite(variance):
            p_value = None
        else:
            z = (chi2_val - mean) / math.sqrt(variance)
            p_value = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {
        "chi2": chi2_val,
        "dof": dof,
        "chi2_dof": chi2_dof,
        "AIC": aic,
        "BIC": bic,
        "p_value": p_value,
        "ndata": int(n_points),
        "nfree": int(n_free),
    }


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def _vector_from_params(params: Mapping[str, object], names: Sequence[str]) -> np.ndarray:
    return np.array([_coerce_float(name, params[name]) for name in names], dtype=float)


def _apply_vector(
    base: Mapping[str, object],
    names: Sequence[str],
    values: Sequence[float],
) -> Dict[str, object]:
    updated = dict(base)
    for key, value in zip(names, values):
        updated[key] = float(value)
    return updated


def _optimise_parameters(
    params: Mapping[str, object],
    free_names: Sequence[str],
    bounds: Sequence[Tuple[float | None, float | None]],
    model_module,
    z: np.ndarray,
    fs8_obs: np.ndarray,
    cov: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, float, Dict[str, object]]:
    if minimize is None:  # pragma: no cover - SciPy optional
        raise RuntimeError("SciPy is required for --fit optimisation but is not installed.")

    x0 = _vector_from_params(params, free_names)

    def objective(vector: np.ndarray) -> float:
        trial = _apply_vector(params, free_names, vector)
        prediction = _fsigma8_prediction(z, trial, model_module)
        resid = fs8_obs - prediction
        return _chi2(resid, cov)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-9, "gtol": 1e-8, "maxiter": 2000},
    )

    best_vec = result.x if result.success else x0
    best_params = _apply_vector(params, free_names, best_vec)
    best_prediction = _fsigma8_prediction(z, best_params, model_module)
    best_residuals = fs8_obs - best_prediction
    best_chi2 = _chi2(best_residuals, cov)
    optimisation = {
        "success": bool(result.success),
        "status": int(getattr(result, "status", -1)),
        "message": getattr(result, "message", ""),
        "n_iter": getattr(result, "nit", None),
        "n_eval": getattr(result, "nfev", None),
    }
    return best_params, best_prediction, best_residuals, best_chi2, optimisation


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fit growth-rate fσ₈(z) measurements with PBUF/ΛCDM models.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--params", help="Optional JSON file with parameter overrides.")
    parser.add_argument("--data-dir", required=True, help="Directory with rsd_index.csv and rsd_index.cov.npy.")
    parser.add_argument("--out", required=True, help="Output directory for the proof JSON.")
    parser.add_argument("--fit", action="store_true", help="Run optimiser using free parameters in the JSON file.")
    args = parser.parse_args()

    model_name = args.model.lower()
    model_label = model_name.upper()
    model_module = MODEL_REGISTRY[model_name]

    params_path = Path(args.params).expanduser().resolve() if args.params else None
    params, free_names, params_file = _initial_parameters(model_name, params_path)
    canonical_block = param_utils.canonical_parameters(model_label)
    if not free_names and args.fit:
        log.warning("No free parameters detected; --fit flag will be ignored.")

    data_dir = Path(args.data_dir).expanduser().resolve()
    z, fs8_obs, dataset_payload = _load_rsd_dataset(data_dir)
    cov = dataset_payload["cov"]
    sigma = dataset_payload["sigma"]
    meta = dataset_payload["meta"]

    settings = read_yaml("config/settings.yml")
    settings_bounds = settings.get("bounds", {})
    bounds_overrides: Mapping[str, Sequence[float]] | None = None
    bounds = _load_bounds(free_names, settings_bounds, bounds_overrides)

    log.info("Loaded %d growth-rate points spanning %.3f ≤ z ≤ %.3f", z.size, float(z.min()), float(z.max()))

    prediction = _fsigma8_prediction(z, params, model_module)
    residuals = fs8_obs - prediction
    chi2_val = _chi2(residuals, cov)
    optimisation_meta: Dict[str, object] | None = None

    if args.fit and free_names:
        log.info("Running optimisation over parameters: %s", ", ".join(free_names))
        params, prediction, residuals, chi2_val, optimisation_meta = _optimise_parameters(
            params,
            free_names,
            bounds,
            model_module,
            z,
            fs8_obs,
            cov,
        )
        log.info("Optimisation completed (success=%s).", optimisation_meta.get("success", False))

    metrics = _chi2_metrics(chi2_val, z.size, len(free_names))

    dataset_tag = str(meta.get("release_tag", DATASET_TOKEN)).upper()
    mode_tag = "MOCK" if settings.get("mock", False) else "REAL"
    run_tag = f"RSD_{dataset_tag}_{mode_tag}"
    run_id = _run_id(run_tag)

    run_root = Path(args.out).expanduser().resolve()
    run_dir = run_root / f"RSD_{model_label}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    params.pop("Or0", None)
    parameter_payload = param_utils.build_parameter_payload(
        model_label,
        fitted=params,
        free_names=free_names,
        canonical=canonical_block,
    )

    meta_map = meta if isinstance(meta, Mapping) else {}
    release_tag = meta_map.get("release_tag")
    dataset_name = f"RSD_{dataset_tag}"
    records = meta_map.get("records")
    primary_record = None
    if isinstance(records, list):
        primary_record = next((rec for rec in records if isinstance(rec, Mapping) and rec.get("kind") == "raw"), None)
        if primary_record is None and records:
            first = records[0]
            primary_record = first if isinstance(first, Mapping) else None

    dataset_path = str((data_dir / "rsd_index.csv").resolve())
    covariance_path = str((data_dir / "rsd_index.cov.npy").resolve())

    dataset_payload = {
        "name": "RSD Growth-rate fσ₈",
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "covariance": covariance_path,
        "tag": dataset_tag,
        "release_tag": release_tag,
        "transform_version": meta_map.get("transform_version", TRANSFORM_VERSION_FALLBACK),
        "prepared_at": meta_map.get("prepared_at"),
        "source": meta_map.get("raw_directory") or (primary_record.get("path") if isinstance(primary_record, Mapping) else None),
        "raw_directory": meta_map.get("raw_directory"),
        "derived_directory": meta_map.get("derived_directory"),
        "citation": meta_map.get("citation"),
        "notes": meta_map.get("notes"),
        "covariance_recipe": meta_map.get("covariance_recipe"),
    }
    dataset_payload = {k: v for k, v in dataset_payload.items() if v not in (None, "", [])}

    output = {
        "run_id": run_id,
        "timestamp": _timestamp(),
        "mock": settings.get("mock", False),
        "dataset": dataset_payload,
        "model": model_label,
        "parameters": parameter_payload,
        "metrics": metrics,
        "observables": {
            "z": z.tolist(),
            "fs8_observed": fs8_obs.tolist(),
            "sigma_fs8": sigma.tolist(),
            "fs8_model": prediction.tolist(),
            "residuals": residuals.tolist(),
        },
        "provenance": {
            "commit": _git_commit(),
            "settings": str(Path("config/settings.yml").resolve()),
            "data_dir": str(data_dir),
            "params_file": str(params_file.resolve()) if params_file else "defaults",
            "sigma_source": dataset_path,
            "covariance_source": covariance_path,
        },
    }

    if optimisation_meta is not None:
        output["optimisation"] = optimisation_meta

    json_path = run_dir / "fit_results.json"
    write_json_atomic(json_path, _jsonify(output))
    log.info("Growth-rate fit stored in %s (χ² = %.4f, dof = %d).", json_path, chi2_val, metrics["dof"])


if __name__ == "__main__":
    main()
