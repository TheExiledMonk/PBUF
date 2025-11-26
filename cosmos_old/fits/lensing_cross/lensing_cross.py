"""Compressed lensing cross-correlation constraints for the V11 architecture."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel

DATA_ROOT = Path("data/standardized")
DEFAULT_FILE = "lensing_cross.npz"
DEFAULT_GAMMA = 0.5
Dataset = Dict[str, Any]


def _ensure_vector(payload: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    if field not in payload:
        raise KeyError(f"Lensing cross dataset missing required field '{field}'")
    array = np.asarray(payload[field], dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _ensure_length(
    payload: np.lib.npyio.NpzFile,
    field: str,
    target_length: int,
) -> np.ndarray:
    if field not in payload:
        raise KeyError(f"Lensing cross dataset missing required field '{field}'")
    array = np.asarray(payload[field], dtype=float)
    if array.ndim == 0:
        array = np.full(target_length, float(array))
    else:
        array = array.reshape(-1)
    if array.shape != (target_length,):
        raise ValueError(f"Field '{field}' must have length {target_length}")
    return array


def _ensure_weights(payload: np.lib.npyio.NpzFile, target_length: int) -> np.ndarray:
    raw = payload.get("weights")
    if raw is None:
        return np.ones(target_length, dtype=float)

    array = np.asarray(raw, dtype=float)
    if array.ndim == 0:
        array = np.full(target_length, float(array))
    else:
        array = array.reshape(-1)
    if array.shape != (target_length,):
        raise ValueError(f"Field 'weights' must have length {target_length}")
    if np.any(array <= 0.0):
        raise ValueError("Lensing cross weights must be positive")
    return array


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
def _load_lensing_cross_dataset_cached(resolved_path: str) -> Dataset:
    target = Path(resolved_path)
    if not target.exists():
        raise FileNotFoundError(f"Lensing cross dataset not found at {target}")

    payload = np.load(target, allow_pickle=True)
    A_obs = _ensure_vector(payload, "A_obs")
    n = len(A_obs)
    if n == 0:
        raise ValueError("Lensing cross dataset must contain at least one measurement.")

    A_err = _ensure_length(payload, "A_err", n)
    p_exponent = _ensure_length(payload, "p_exponent", n)
    q_exponent = _ensure_length(payload, "q_exponent", n)
    z_eff = _ensure_length(payload, "z_eff", n)
    S8_fid = _ensure_length(payload, "S8_fid", n)
    fs8_fid = _ensure_length(payload, "fs8_fid", n)
    weights = _ensure_weights(payload, n)
    gamma = payload.get("gamma")
    if gamma is None:
        gamma = np.full(n, DEFAULT_GAMMA)
    else:
        gamma = np.asarray(gamma, dtype=float)
        if gamma.ndim == 0:
            gamma = np.full(n, float(gamma))
        else:
            gamma = gamma.reshape(-1)
        if gamma.shape != (n,):
            raise ValueError(f"Field 'gamma' must have length {n}")

    labels = _parse_labels(payload)

    cov = payload.get("cov")
    inv_cov: np.ndarray | None = None
    cov_arr: np.ndarray | None = None
    if cov is not None:
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("Lensing cross covariance must be square")
        if cov_arr.shape[0] != n:
            raise ValueError("Lensing cross covariance size must match the number of measurements")
        inv_cov = np.linalg.inv(cov_arr)
        if not np.all(np.isfinite(inv_cov)):
            raise ValueError("Lensing cross covariance inverse contains non-finite values")

    meta: Dict[str, Any] = {}
    if "meta" in payload:
        raw_meta = payload["meta"]
        if isinstance(raw_meta, dict):
            meta.update(raw_meta)
    meta.setdefault("file", str(target))
    if labels is not None:
        meta["labels"] = tuple(labels)

    dataset: Dataset = {
        "name": payload.get("name", "lensing_cross"),
        "type": payload.get("type", "lensing_cross"),
        "n_datasets": n,
        "A_obs": A_obs,
        "A_err": A_err,
        "p_exponent": p_exponent,
        "q_exponent": q_exponent,
        "z_eff": z_eff,
        "S8_fid": S8_fid,
        "fs8_fid": fs8_fid,
        "gamma": gamma,
        "weights": weights,
        "meta": meta,
    }
    if labels is not None:
        dataset["labels"] = tuple(labels)
    if cov_arr is not None:
        dataset["cov"] = cov_arr
        dataset["inv_cov"] = inv_cov

    return dataset


def load_lensing_cross_dataset(path: Path | str | None = None) -> Dataset:
    resolved = _resolve_dataset_path(path)
    return _load_lensing_cross_dataset_cached(resolved)


def run_lensing_cross_fit(model: CosmologyModel, dataset: Dataset | None = None) -> tuple[float, Dict[str, np.ndarray]]:
    dataset = dataset or load_lensing_cross_dataset()
    A_obs = np.asarray(dataset["A_obs"], dtype=float)
    A_err = np.asarray(dataset["A_err"], dtype=float)
    p_exponent = np.asarray(dataset["p_exponent"], dtype=float)
    q_exponent = np.asarray(dataset["q_exponent"], dtype=float)
    z_eff = np.asarray(dataset["z_eff"], dtype=float)
    S8_fid = np.asarray(dataset["S8_fid"], dtype=float)
    fs8_fid = np.asarray(dataset["fs8_fid"], dtype=float)
    gamma = np.asarray(dataset["gamma"], dtype=float)
    weights = np.asarray(dataset["weights"], dtype=float)

    n = len(A_obs)
    if not (
        A_err.shape == (n,)
        and p_exponent.shape == (n,)
        and q_exponent.shape == (n,)
        and z_eff.shape == (n,)
        and S8_fid.shape == (n,)
        and fs8_fid.shape == (n,)
        and gamma.shape == (n,)
        and weights.shape == (n,)
    ):
        raise ValueError("Lensing cross dataset payload contains mismatched array shapes.")

    if np.any(weights <= 0.0):
        raise ValueError("Lensing cross weights must be positive.")
    S8_model = np.asarray([model.S8(g) for g in gamma], dtype=float)
    fs8_model = np.asarray(model.fs8(z_eff), dtype=float).reshape(n)

    A_model = (S8_model / S8_fid) ** p_exponent * (fs8_model / fs8_fid) ** q_exponent

    diff = A_model - A_obs
    scaled_err = A_err / weights
    cov = dataset.get("cov")
    if cov is not None:
        inv_cov = dataset.get("inv_cov")
        if inv_cov is None:
            inv_cov = np.linalg.inv(cov)
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        if np.any(A_err <= 0.0):
            raise ValueError("Lensing cross errors must be positive when no covariance is provided.")
        chi2 = float(np.sum((diff / scaled_err) ** 2))

    additional = {
        "A_model": A_model,
        "z_eff": z_eff,
        "scaled_err": scaled_err,
        "gamma": gamma,
        "fs8_model": fs8_model,
        "S8_model": S8_model,
    }
    if "labels" in dataset:
        additional["labels"] = np.asarray(dataset["labels"], dtype=object)

    extras = build_fit_extras(
        dataset=dataset,
        predictions=A_model,
        observed=A_obs,
        residuals=diff,
        weights=weights,
        additional=additional,
    )

    return chi2, extras


def run_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    """Standard entrypoint for joint fits."""

    return run_lensing_cross_fit(model, dataset)
