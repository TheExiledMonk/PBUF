"""Model-neutral cosmic chronometer χ² helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel
from data_interface.standardize import ensure_standard_dataset

DATA_ROOT = Path("data/cc")
DEFAULT_PATTERN = "*.npz"
STANDARDIZED_CC_PATH = Path(__file__).resolve().parents[3] / "data" / "standardized" / "cc.npz"


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray) and raw.shape == ():
        value = raw.item()
        if isinstance(value, dict):
            return dict(value)
        return {"meta": value}
    if isinstance(raw, dict):
        return dict(raw)
    return {"meta": raw}


def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT
    return str(candidate.expanduser().resolve())


def _discover_paths(target: Path) -> list[Path]:
    if target.is_file():
        paths = [target]
    elif target.is_dir():
        paths = sorted(target.glob(DEFAULT_PATTERN))
    else:
        raise FileNotFoundError(f"CC data file/directory not found: {target}")

    if not paths:
        raise FileNotFoundError(f"No CC datasets found under {target}")
    return paths


def _load_single_dataset(path: Path) -> dict:
    payload = np.load(path, allow_pickle=True)

    z = np.asarray(payload["z"], dtype=float)
    obs = np.asarray(payload["H_obs"], dtype=float)
    err = payload.get("H_err")
    cov = payload.get("cov")

    if err is not None:
        err = np.asarray(err, dtype=float)
    if cov is not None:
        cov = np.asarray(cov, dtype=float)

    if err is None and cov is None:
        raise ValueError(f"CC dataset {path} lacks uncertainty information")

    if err is not None and err.shape != obs.shape:
        raise ValueError(f"CC dataset {path} obs/err shape mismatch")

    meta = _parse_metadata(payload.get("meta"))
    meta.setdefault("file", str(path))

    return {"z": z, "obs": obs, "err": err, "cov": cov, "meta": meta}


def _block_diag(matrices: Iterable[np.ndarray]) -> np.ndarray:
    sizes = [mat.shape[0] for mat in matrices]
    total = sum(sizes)
    result = np.zeros((total, total), dtype=float)
    offset = 0
    for mat in matrices:
        n = mat.shape[0]
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"Invalid covariance shape: {mat.shape}")
        result[offset : offset + n, offset : offset + n] = mat
        offset += n
    return result


def load_cc_dataset(path: Path | str | None = None) -> Dict[str, Any]:
    """Load one or more CC .npz datasets and return a standardized payload."""
    if path is None:
        try:
            return _load_standardized_cc_dataset()
        except FileNotFoundError:
            pass

    resolved = _resolve_dataset_path(path)
    return _load_cc_dataset_cached(resolved)


@lru_cache(maxsize=None)
def _load_cc_dataset_cached(resolved_path: str) -> Dict[str, Any]:
    target = Path(resolved_path)
    files = _discover_paths(target)
    payloads = [_load_single_dataset(file) for file in files]

    z_vals = [payload["z"] for payload in payloads]
    obs_vals = [payload["obs"] for payload in payloads]
    cov_blocks: list[np.ndarray] = []

    for payload in payloads:
        cov_block = payload["cov"]
        if cov_block is None:
            err = payload["err"]
            assert err is not None
            cov_block = np.diag(err**2)
        cov_blocks.append(cov_block)

    combined_cov = _block_diag(cov_blocks)
    combined_inv_cov = np.linalg.inv(combined_cov)
    combined_err = np.sqrt(np.clip(np.diag(combined_cov), 0.0, None))
    combined_z = np.concatenate(z_vals)
    combined_obs = np.concatenate(obs_vals)

    combined_meta = {
        "files": [str(file) for file in files],
        "datasets": [payload["meta"] for payload in payloads],
        "z_min": float(np.min(combined_z)),
        "z_max": float(np.max(combined_z)),
    }

    dataset = {
        "name": "CC_compilation",
        "type": "CC",
        "z": combined_z,
        "obs": combined_obs,
        "err": combined_err,
        "cov": combined_cov,
        "meta": combined_meta,
    }

    dataset = ensure_standard_dataset(dataset, "CC")
    dataset["inv_cov"] = combined_inv_cov
    return dataset


def _load_standardized_cc_dataset() -> Dict[str, Any]:
    if not STANDARDIZED_CC_PATH.exists():
        raise FileNotFoundError(f"Standardized CC cache missing: {STANDARDIZED_CC_PATH}")

    with np.load(STANDARDIZED_CC_PATH, allow_pickle=True) as payload:
        z = np.asarray(payload["z"], dtype=float)
        obs = np.asarray(payload["obs"], dtype=float)
        cov = np.asarray(payload["cov"], dtype=float) if "cov" in payload else None
        err = payload.get("err")
        if err is not None:
            err = np.asarray(err, dtype=float)

        inv_cov = None
        if cov is not None:
            inv_cov = np.linalg.inv(cov)
            err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        if cov is None and err is None:
            raise ValueError(
                "Standardized CC dataset lacks covariance and error information"
            )

        metadata = _parse_metadata(payload.get("meta"))
        metadata.setdefault("source_file", STANDARDIZED_CC_PATH.name)

        name_entry = payload.get("name")
        if name_entry is None:
            name = "CC_compilation"
        else:
            name = str(name_entry.item()) if hasattr(name_entry, "item") else str(name_entry)

        dataset = {
            "name": name,
            "type": "CC",
            "z": z,
            "obs": obs,
            "err": err,
            "cov": cov,
            "meta": metadata,
        }

        dataset = ensure_standard_dataset(dataset, "CC")
        if inv_cov is not None:
            dataset["inv_cov"] = inv_cov
        return dataset


def run_cc_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    """Return the CC χ² for a single cosmology model."""
    dataset = dataset or load_cc_dataset()

    z = np.asarray(dataset["z"], dtype=float)
    observed = np.asarray(dataset["obs"], dtype=float)
    errors = dataset.get("err")
    cov = dataset.get("cov")
    inv_cov = dataset.get("inv_cov")

    model_values = np.asarray(model.Hubble(z), dtype=float)
    if not np.all(np.isfinite(model_values)):
        return float(np.inf), {"H_model": model_values}
    delta = observed - model_values

    if cov is not None and inv_cov is not None:
        chi2 = float(delta.T @ inv_cov @ delta)
    else:
        if errors is None:
            raise ValueError("CC dataset is missing both covariance and errors.")
        chi2 = float(np.sum((delta / errors) ** 2))

    residuals = model_values - observed
    extras = build_fit_extras(
        dataset=dataset,
        predictions=model_values,
        observed=observed,
        residuals=residuals,
    )
    extras["H_model"] = model_values
    return chi2, extras


def run_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    """Standard entrypoint for joint fits."""

    return run_cc_fit(model, dataset)
