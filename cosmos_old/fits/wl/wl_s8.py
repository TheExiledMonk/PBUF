"""Weak lensing S₈ χ² helpers that stay agnostic to the cosmology engine."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel

DATA_ROOT = Path("data/standardized")
DEFAULT_FILE = "wl_s8.npz"
Dataset = Dict[str, Any]


def _ensure_vector(payload: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    if field not in payload:
        raise KeyError(f"WL S₈ dataset missing required field '{field}'")
    arr = np.asarray(payload[field], dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _parse_labels(payload: np.lib.npyio.NpzFile) -> Sequence[str] | None:
    raw = payload.get("labels")
    if raw is None:
        return None
    compressed = np.asarray(raw)
    return [str(item) for item in compressed.reshape(-1)]


def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT / DEFAULT_FILE
    return str(candidate.expanduser().resolve())


@lru_cache(maxsize=None)
def _load_wl_s8_dataset_cached(resolved_path: str) -> Dataset:
    target = Path(resolved_path)
    if not target.exists():
        raise FileNotFoundError(f"WL S₈ dataset not found at {target}")

    payload = np.load(target, allow_pickle=True)
    S8_obs = _ensure_vector(payload, "S8_obs")
    S8_err = _ensure_vector(payload, "S8_err")
    gamma = _ensure_vector(payload, "gamma")

    if not (S8_obs.shape == S8_err.shape == gamma.shape):
        raise ValueError("WL S₈ arrays must share the same shape")

    cov = payload.get("cov")
    inv_cov: np.ndarray | None = None
    cov_arr: np.ndarray | None = None
    if cov is not None:
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("WL S₈ covariance must be a square matrix")
        if cov_arr.shape[0] != len(S8_obs):
            raise ValueError("WL S₈ covariance size must match the number of measurements")
        inv_cov = np.linalg.inv(cov_arr)
        if not np.all(np.isfinite(inv_cov)):
            raise ValueError("WL S₈ covariance inverse contains non-finite values")

    labels = _parse_labels(payload)
    meta: Dict[str, Any] = {}
    if "meta" in payload:
        raw_meta = payload["meta"]
        if isinstance(raw_meta, dict):
            meta.update(raw_meta)
    meta.setdefault("file", str(target))
    if labels is not None:
        meta["labels"] = tuple(labels)

    dataset: Dataset = {
        "name": payload.get("name", "WL_S8"),
        "type": "WL_S8",
        "S8_obs": S8_obs,
        "S8_err": S8_err,
        "gamma": gamma,
        "meta": meta,
    }
    if cov_arr is not None:
        dataset["cov"] = cov_arr
        dataset["inv_cov"] = inv_cov

    return dataset


def load_wl_s8_dataset(path: Path | str | None = None) -> Dataset:
    resolved = _resolve_dataset_path(path)
    return _load_wl_s8_dataset_cached(resolved)


def run_wl_s8_fit(
    model: CosmologyModel,
    dataset: Dataset | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    dataset = dataset or load_wl_s8_dataset()
    S8_obs = np.asarray(dataset["S8_obs"], dtype=float)
    S8_err = np.asarray(dataset["S8_err"], dtype=float)
    gamma = np.asarray(dataset["gamma"], dtype=float)

    om = float(model.omega_m0())
    if om <= 0.0:
        raise ValueError("Model returned non-positive Ω_m0; cannot build S₈ prediction.")
    s8 = float(model.sigma8())

    S8_model = s8 * (om / 0.3) ** gamma
    diff = S8_model - S8_obs

    cov = dataset.get("cov")
    if cov is not None:
        inv_cov = dataset.get("inv_cov")
        if inv_cov is None:
            inv_cov = np.linalg.inv(cov)
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        if np.any(S8_err <= 0.0):
            raise ValueError("WL S₈ errors must be positive when no covariance is provided.")
        chi2 = float(np.sum((diff / S8_err) ** 2))

    extras = build_fit_extras(
        dataset=dataset,
        predictions=S8_model,
        observed=S8_obs,
        residuals=diff,
        additional={"gamma": gamma, "S8_err": S8_err, "S8_model": S8_model},
    )
    return chi2, extras


def run_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    """Standard entrypoint for joint fits."""

    return run_wl_s8_fit(model, dataset)
