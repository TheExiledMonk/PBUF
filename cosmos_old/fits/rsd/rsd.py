"""Model-neutral Redshift-Space Distortion χ² helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel
from data_interface.standardize import ensure_standard_dataset

DATA_ROOT = Path("data/standardized")
DEFAULT_FILE = "rsd.npz"


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        value = raw.item()
        if isinstance(value, dict):
            return dict(value)
        return {"meta": value}
    if isinstance(raw, dict):
        return dict(raw)
    return {"meta": raw}


def _select_obs_array(payload: np.lib.npyio.NpzFile) -> np.ndarray:
    candidates = ("fs8", "fsigma8", "fs8_obs", "obs")
    for key in candidates:
        if key in payload:
            return np.asarray(payload[key], dtype=float)
    raise ValueError(f"RSD dataset {payload.files} lacks an observations array.")


def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT / DEFAULT_FILE
    return str(candidate.expanduser().resolve())


@lru_cache(maxsize=None)
def _load_rsd_dataset_cached(resolved_path: str) -> Dict[str, Any]:
    target = Path(resolved_path)
    if not target.exists():
        raise FileNotFoundError(f"RSD dataset not found at {target}")

    payload = np.load(target, allow_pickle=True)
    z = np.asarray(payload["z"], dtype=float)
    obs = _select_obs_array(payload)
    cov = np.asarray(payload["cov"], dtype=float) if "cov" in payload else None
    err = np.asarray(payload["err"], dtype=float) if "err" in payload else None
    if cov is not None and err is None:
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    name_entry = payload.get("name")
    name = "RSD_compilation"
    if name_entry is not None:
        name = str(name_entry.item()) if hasattr(name_entry, "item") else str(name_entry)

    meta = _parse_metadata(payload.get("meta"))
    meta.setdefault("file", str(target))

    dataset = {
        "name": name,
        "type": "RSD",
        "z": z,
        "obs": obs,
        "err": err,
        "cov": cov,
        "meta": meta,
    }

    dataset = ensure_standard_dataset(dataset, "RSD")
    if cov is not None:
        dataset["inv_cov"] = np.linalg.inv(cov)

    return dataset


def load_rsd_dataset(path: Path | str | None = None) -> Dict[str, Any]:
    resolved = _resolve_dataset_path(path)
    return _load_rsd_dataset_cached(resolved)


def run_rsd_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    dataset = dataset or load_rsd_dataset()
    dataset = ensure_standard_dataset(dataset, "RSD")

    z = np.asarray(dataset["z"], dtype=float)
    observed = np.asarray(dataset["obs"], dtype=float)
    fs8_model = np.asarray(model.fs8(z), dtype=float)
    diff = fs8_model - observed

    cov = dataset.get("cov")
    inv_cov = dataset.get("inv_cov")
    if cov is not None and inv_cov is not None:
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        err = dataset.get("err")
        if err is None:
            raise ValueError("RSD dataset lacks both covariance and errors.")
        chi2 = float(np.sum((diff / err) ** 2))

    extras = build_fit_extras(
        dataset=dataset,
        predictions=fs8_model,
        observed=observed,
        residuals=diff,
    )
    extras["fs8_model"] = fs8_model
    return chi2, extras


def run_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    """Standard entrypoint for joint fits."""

    return run_rsd_fit(model, dataset)
