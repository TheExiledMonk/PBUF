#!/usr/bin/env python3
"""
Chronometer H(z) fitter mirroring the SN/BAO/CMB pipelines.

The fitter only orchestrates data I/O and optimisation; all cosmology
predictions are delegated to the model kernels in ``core/``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore[assignment]

try:
    from scipy.optimize import minimize
    from scipy.stats import chi2 as chi2_dist
except ImportError:  # pragma: no cover - SciPy is optional at runtime
    minimize = None  # type: ignore[assignment]
    chi2_dist = None  # type: ignore[assignment]

from core import gr_models, pbuf_models
from utils.io import read_latest_result, read_yaml, write_json_atomic
from utils import parameters as param_utils


# ---------------------------------------------------------------------------
# Model registry & defaults
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Dict[str, float]] = {
    "lcdm": {
        "H0": 67.4,
        "Om0": 0.315,
        "Or0": 0.0,
        "Ok0": 0.0,
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
    },
}

MODEL_REGISTRY = {"lcdm": gr_models, "pbuf": pbuf_models}
FREE_PARAMS: Dict[str, Tuple[str, ...]] = {
    "lcdm": ("H0", "Om0"),
    "pbuf": ("H0", "Om0"),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # pragma: no cover - git may be absent in some envs
        return "n/a"


def _chi2_p_value(chi2_value: float, dof: int) -> float:
    if dof <= 0:
        return float("nan")
    if chi2_dist is not None:
        return float(chi2_dist.sf(chi2_value, dof))
    # Wilson–Hilferty approximation as a fallback.
    mean = dof
    variance = 2.0 * dof
    if variance <= 0:
        return float("nan")
    z = (chi2_value - mean) / math.sqrt(variance)
    # survival function of normal distribution using erfc
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _sanitize_token(value: str | None, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        text = fallback
    token = "".join(ch if ch.isalnum() else "_" for ch in text)
    token = token.strip("_") or fallback
    return token.upper()


def _run_id(tag: str) -> str:
    return f"{tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _save_vectors(run_dir: Path, vectors: Mapping[str, np.ndarray]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for key, array in vectors.items():
        path = run_dir / f"{key}.npy"
        np.save(path, array)
        paths[key] = str(path.resolve())
    return paths


def _coerce_float(name: str, value: object) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Parameter '{name}' expects a numeric value, received '{value}'.") from exc
    raise TypeError(f"Parameter '{name}' expects a numeric value, received type {type(value).__name__}.")


def _parse_params(
    model: str,
    params_path: str | None,
) -> Tuple[Dict[str, object], Tuple[str, ...]]:
    """
    Merge defaults with user-supplied parameters.

    Returns
    -------
    params : dict
        Parameter dictionary passed to the model kernel.
    free_names : tuple[str, ...]
        Ordered list of parameter names treated as free in optimisation.
    """

    params: MutableMapping[str, object] = dict(DEFAULT_PARAMS.get(model, {}))
    fixed: set[str] = set()
    explicit_free: set[str] = set()

    if params_path:
        payload = json.loads(Path(params_path).read_text())
        if not isinstance(payload, Mapping):
            raise TypeError(f"Parameter file '{params_path}' must contain a JSON object.")
        for key, raw_val in payload.items():
            if isinstance(raw_val, Mapping):
                if "value" not in raw_val:
                    raise KeyError(f"Parameter '{key}' dictionary must include a 'value' field.")
                params[key] = raw_val["value"]
                fixed_flag = raw_val.get("fixed")
                if fixed_flag is True:
                    fixed.add(key)
                elif fixed_flag is False:
                    explicit_free.add(key)
            else:
                params[key] = raw_val

    for key, value in list(params.items()):
        if isinstance(value, (int, float, np.floating, str)):
            try:
                params[key] = _coerce_float(key, value)
            except (TypeError, ValueError):
                # Retain non-numeric values (e.g. policy dictionaries) verbatim.
                params[key] = value

    allowed = set(FREE_PARAMS.get(model, tuple(params.keys())))
    free_names = tuple(
        name
        for name in params.keys()
        if name not in fixed
        and isinstance(params[name], (int, float, np.floating))
        and (name in allowed or name in explicit_free)
    )
    return dict(params), free_names


def _load_bounds(
    free_names: Sequence[str],
    settings_bounds: Mapping[str, Sequence[float]],
    bounds_path: str | None,
) -> Tuple[Tuple[float | None, float | None], ...]:
    bounds_map: Dict[str, Sequence[float]] = dict(settings_bounds)
    if bounds_path:
        override = json.loads(Path(bounds_path).read_text())
        if not isinstance(override, Mapping):
            raise TypeError(f"Bounds file '{bounds_path}' must contain a JSON object.")
        bounds_map.update({str(k): v for k, v in override.items()})

    bounds: list[Tuple[float | None, float | None]] = []
    for name in free_names:
        low, high = None, None
        spec = bounds_map.get(name)
        if spec is not None:
            if len(spec) != 2:
                raise ValueError(f"Bounds for '{name}' must contain exactly two entries.")
            low = float(spec[0]) if spec[0] is not None else None
            high = float(spec[1]) if spec[1] is not None else None
        bounds.append((low, high))
    return tuple(bounds)


def _load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "chronometers_index.meta.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def load_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, Path, Path]:
    table_path = data_dir / "chronometers_index.csv"
    cov_path = data_dir / "chronometers_index.cov.npy"

    if not table_path.exists():
        raise FileNotFoundError(f"Expected chronometer table at {table_path}")
    if not cov_path.exists():
        raise FileNotFoundError(f"Expected covariance matrix at {cov_path}")

    if pd is not None:
        df = pd.read_csv(table_path)
        required = {"z", "H"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Chronometer table missing columns: {', '.join(sorted(missing))}")
        z = df["z"].to_numpy(dtype=float)
        h_obs = df["H"].to_numpy(dtype=float)
        n_rows = len(df)
    else:  # pragma: no cover - fallback for environments without pandas
        table = np.genfromtxt(table_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = set(table.dtype.names or ())
        required = {"z", "H"}
        if not required.issubset(names):
            missing = required - names
            raise ValueError(f"Chronometer table missing columns: {', '.join(sorted(missing))}")
        z = np.asarray(table["z"], dtype=float)
        h_obs = np.asarray(table["H"], dtype=float)
        n_rows = z.size

    cov = np.load(cov_path)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix at {cov_path} must be square (got {cov.shape}).")
    if cov.shape[0] != n_rows:
        raise ValueError(f"Covariance dimension {cov.shape} does not match table length {n_rows}.")

    inv_cov = np.linalg.pinv(cov, hermitian=True)
    manifest = _load_manifest(data_dir)
    return z, h_obs, cov, inv_cov, manifest, table_path, cov_path


def _chi2(
    kernel,
    params: Mapping[str, object],
    z: np.ndarray,
    h_obs: np.ndarray,
    inv_cov: np.ndarray,
    *,
    allow_invalid: bool = False,
) -> float:
    try:
        h_model = kernel.H(z, params)  # type: ignore[attr-defined]
    except Exception:
        if allow_invalid:
            return float("inf")
        raise
    if h_model.shape != h_obs.shape:
        message = f"Model prediction shape {h_model.shape} does not match observations {h_obs.shape}."
        if allow_invalid:
            return float("inf")
        raise ValueError(message)
    if not np.all(np.isfinite(h_model)):
        if allow_invalid:
            return float("inf")
        raise ValueError("Model predictions contain non-finite values.")
    residual = h_obs - h_model
    if not np.all(np.isfinite(residual)):
        if allow_invalid:
            return float("inf")
        raise ValueError("Residuals contain non-finite values.")
    chi2_val = float(residual.T @ inv_cov @ residual)
    if not math.isfinite(chi2_val):
        if allow_invalid:
            return float("inf")
        raise ValueError("Computed χ² is non-finite.")
    return chi2_val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit H(z) cosmic chronometer data.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY), help="Model identifier to evaluate.")
    parser.add_argument("--params", help="Path to JSON with parameter values (supports {'value': x, 'fixed': true}).")
    parser.add_argument("--data-dir", required=True, help="Directory containing chronometers_index.{csv,cov.npy}.")
    parser.add_argument("--out", default="proofs/results", help="Output directory for fit artefacts (default: proofs/results).")
    parser.add_argument("--fit", action="store_true", help="Run optimiser; otherwise evaluate χ² at provided parameters.")
    parser.add_argument("--bounds", help="Optional JSON file with parameter bounds overrides.")
    parser.add_argument("--max-iter", type=int, help="Maximum optimisation iterations (defaults to config/settings.yml).")
    parser.add_argument("--tol", type=float, default=None, help="Optimiser tolerance (defaults to config/settings.yml).")
    parser.add_argument("--method", default="L-BFGS-B", help="scipy.optimize.minimize method to use.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_key = args.model.lower()
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'.")
    kernel = MODEL_REGISTRY[model_key]

    z, h_obs, cov, inv_cov, manifest, table_path, cov_path = load_data(data_dir)

    model_tag = args.model.upper()
    canonical_block = param_utils.canonical_parameters(model_tag)
    param_overrides, free_names = _parse_params(model_key, args.params)
    params = dict(canonical_block)
    params.update(param_overrides)
    cmb_calibration_path = None
    cmb_fit = read_latest_result(model=model_tag, kind="CMB")
    if cmb_fit:
        cmb_calibration_path = cmb_fit.get("_source_path")

    free_numeric = tuple(name for name in free_names if isinstance(params.get(name), (int, float, np.floating)))

    settings = read_yaml("config/settings.yml")
    fitting_cfg: Mapping[str, object] = {}
    settings_bounds: Mapping[str, Sequence[float]] = {}
    mock_flag = False
    if isinstance(settings, Mapping):
        fitting_cfg = settings.get("fitting", {}) or {}
        settings_bounds = settings.get("bounds", {}) or {}
        mock_flag = bool(settings.get("mock", False))

    tolerance = float(args.tol) if args.tol is not None else float(fitting_cfg.get("tolerance", 1.0e-6))
    max_iter = int(args.max_iter) if args.max_iter is not None else int(fitting_cfg.get("max_iter", 5000))
    bounds = _load_bounds(free_numeric, settings_bounds, args.bounds)

    optimiser_info: Dict[str, object] = {
        "success": False,
        "status": 0,
        "message": "evaluation-only",
        "method": args.method,
        "free_parameters": list(free_numeric),
    }

    if args.fit and not free_numeric:
        optimiser_info = {
            "success": False,
            "status": 0,
            "message": "no free parameters",
            "method": args.method,
            "free_parameters": [],
        }

    if args.fit and free_numeric:
        if minimize is None:  # pragma: no cover - SciPy missing fallback
            raise RuntimeError("scipy is required for --fit mode but is not available.")

        x0 = np.array([float(params[name]) for name in free_numeric], dtype=float)

        def _objective(theta: np.ndarray) -> float:
            trial = dict(params)
            for name, val in zip(free_numeric, theta):
                trial[name] = float(val)
            return _chi2(kernel, trial, z, h_obs, inv_cov, allow_invalid=True)

        opt_result = minimize(
            _objective,
            x0,
            method=args.method,
            bounds=bounds if bounds else None,
            options={"maxiter": int(max_iter), "ftol": float(tolerance)},
        )

        best_vec = np.array(opt_result.x, dtype=float) if getattr(opt_result, "x", None) is not None else x0
        if not opt_result.success:
            best_vec = x0
        for name, val in zip(free_numeric, best_vec):
            params[name] = float(val)
        chi2_val = float(opt_result.fun) if opt_result.success else _objective(best_vec)

        optimiser_info = {
            "success": bool(opt_result.success),
            "status": int(getattr(opt_result, "status", -1)),
            "message": getattr(opt_result, "message", ""),
            "method": args.method,
            "n_iter": getattr(opt_result, "nit", None),
            "n_eval": getattr(opt_result, "nfev", None),
            "free_parameters": list(free_numeric),
        }

        if opt_result.success:
            try:
                chi2_val = _chi2(kernel, params, z, h_obs, inv_cov)
            except ValueError as exc:
                chi2_val = float("inf")
                optimiser_info["success"] = False
                optimiser_info["message"] = f"invalid parameters: {exc}"
    else:
        chi2_val = _chi2(kernel, params, z, h_obs, inv_cov)

    if not math.isfinite(chi2_val):
        message = optimiser_info.get("message", "") if isinstance(optimiser_info, Mapping) else ""
        raise RuntimeError(f"Chronometer fit failed for model {model_key.upper()}: {message or 'non-finite χ²'}")

    h_model = kernel.H(z, params)  # type: ignore[attr-defined]
    residuals = h_obs - h_model
    sigma_diag = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        pulls = np.where(sigma_diag > 0, residuals / sigma_diag, 0.0)

    n_points = len(z)
    n_free = len(free_numeric)
    dof = max(n_points - n_free, 1)
    chi2_dof = float(chi2_val) / dof if dof > 0 else float("nan")
    aic = float(chi2_val) + 2.0 * n_free
    bic = float(chi2_val) + n_free * math.log(max(n_points, 1))

    model_tag = model_key.upper()
    dataset_token = _sanitize_token(manifest.get("release_tag") if isinstance(manifest, Mapping) else None, "OHD")
    mode_token = "MOCK" if mock_flag else "REAL"
    run_id = _run_id(f"CC_{model_tag}_{dataset_token}_{mode_token}")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vectors = {
        "z": z,
        "H_obs": h_obs,
        "H_model": h_model,
        "residuals": residuals,
        "sigma_H": sigma_diag,
        "pulls": pulls,
    }
    vector_paths = _save_vectors(run_dir, vectors)

    manifest_map = manifest if isinstance(manifest, Mapping) else {}
    release_tag = manifest_map.get("release_tag")
    dataset_name = f"CC_{dataset_token}"
    raw_records = manifest_map.get("records")
    primary_record = None
    if isinstance(raw_records, list) and raw_records:
        primary_record = next((rec for rec in raw_records if isinstance(rec, Mapping) and rec.get("kind") == "raw"), None)
        if primary_record is None:
            primary_record = raw_records[0] if isinstance(raw_records[0], Mapping) else None

    dataset_info = {
        "name": "Cosmic Chronometers H(z)",
        "dataset_name": dataset_name,
        "dataset_path": str(table_path.resolve()),
        "path": str(table_path.resolve()),
        "covariance": str(cov_path.resolve()),
        "release_tag": release_tag,
        "transform_version": manifest_map.get("transform_version"),
        "prepared_at": manifest_map.get("prepared_at"),
        "source": manifest_map.get("raw_directory") or (primary_record.get("path") if isinstance(primary_record, Mapping) else None),
        "raw_directory": manifest_map.get("raw_directory"),
        "derived_directory": manifest_map.get("derived_directory"),
        "citation": manifest_map.get("citation"),
        "notes": manifest_map.get("notes"),
        "covariance_recipe": manifest_map.get("covariance_recipe"),
        "frac_sys": manifest_map.get("frac_sys"),
    }
    dataset_info = {k: v for k, v in dataset_info.items() if v not in (None, "", [])}

    metrics = {
        "N": int(n_points),
        "N_params": int(n_free),
        "dof": int(dof),
        "chi2": float(chi2_val),
        "chi2_dof": chi2_dof,
        "AIC": float(aic),
        "BIC": float(bic),
        "p_value": _chi2_p_value(float(chi2_val), dof),
        "cov_health": {"cond_num": float(np.linalg.cond(cov))},
    }

    params.pop("Or0", None)
    parameter_payload = param_utils.build_parameter_payload(
        model_tag,
        fitted=params,
        free_names=free_numeric,
        canonical=canonical_block,
    )

    provenance = {
        "commit": _git_commit(),
        "settings": str(Path("config/settings.yml").resolve()),
        "constants": str(Path("config/constants.py").resolve()),
        "data_manifest": str((data_dir / "chronometers_index.meta.json").resolve())
        if (data_dir / "chronometers_index.meta.json").exists()
        else None,
        "params_input": str(Path(args.params).resolve()) if args.params else None,
        "bounds_input": str(Path(args.bounds).resolve()) if args.bounds else None,
        "cmb_calibration": cmb_calibration_path,
    }
    provenance = {k: v for k, v in provenance.items() if v is not None}

    payload = {
        "run_id": run_id,
        "timestamp": _timestamp(),
        "model": model_tag,
        "mock": mock_flag,
        "dataset": dataset_info,
        "parameters": parameter_payload,
        "metrics": metrics,
        "data_vectors": vector_paths,
        "figures": {},
        "optimisation": optimiser_info,
        "provenance": provenance,
    }

    result_path = run_dir / "fit_results.json"
    write_json_atomic(result_path, payload, indent=2)
    print(f"[INFO] Chronometer fit saved to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
