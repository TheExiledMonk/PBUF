"""Galaxy power spectrum fit for cosmos2 models (compressed observables)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras


def _normalize_label(raw_label: Any) -> str:
    label = str(raw_label).strip().lower()
    for needle, repl in (("σ", "sigma"), ("Σ", "sigma")):
        label = label.replace(needle, repl)
    cleaned = []
    for ch in label:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    label = "".join(cleaned).strip("_")
    while "__" in label:
        label = label.replace("__", "_")
    return label


def _is_fs8_label(key: str) -> bool:
    return "fs8" in key or "f_sigma" in key or "fsigma" in key


def _is_dh_label(key: str) -> bool:
    return "dh" in key and "fid" in key


def _is_dm_label(key: str) -> bool:
    return "dm" in key and "fid" in key and "dh" not in key


def _is_h_label(key: str) -> bool:
    return "h" in key and "fid" in key and "dh" not in key and "dm" not in key


def _build_model_matrix(model: Any, z: np.ndarray, labels: list[str], fiducials: Dict[str, np.ndarray]) -> np.ndarray:
    preds = []
    for label in labels:
        key = _normalize_label(label)
        if _is_fs8_label(key):
            preds.append(np.asarray(model.fs8(z), dtype=float))
        elif _is_dh_label(key):
            dh_fid = fiducials.get("DH")
            if dh_fid is None:
                raise ValueError("Galaxy PK dataset requires DH fiducials for DH_obs.")
            preds.append(np.asarray(model.DH(z), dtype=float) / dh_fid)
        elif _is_dm_label(key):
            dm_fid = fiducials.get("DM")
            if dm_fid is None:
                raise ValueError("Galaxy PK dataset requires DM fiducials for DM_obs.")
            preds.append(np.asarray(model.DM(z), dtype=float) / dm_fid)
        elif _is_h_label(key):
            h_fid = fiducials.get("H")
            if h_fid is None:
                raise ValueError("Galaxy PK dataset requires H fiducials for H_obs.")
            preds.append(np.asarray(model.Hubble(z), dtype=float) / h_fid)
        else:
            raise ValueError(f"Unsupported Galaxy PK observable label '{label}'.")
    return np.column_stack(preds)


def run_galaxy_pk_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("galaxy_pk")
    z = np.asarray(dataset["z"], dtype=float)
    obs_matrix = np.asarray(dataset["obs"], dtype=float)
    if obs_matrix.ndim == 1:
        obs_matrix = obs_matrix.reshape((-1, 1))
    if obs_matrix.shape[0] != len(z):
        raise ValueError("Galaxy PK dataset row count does not match provided redshifts.")

    labels = dataset.get("labels")
    if labels is None:
        labels = [f"obs_{idx}" for idx in range(obs_matrix.shape[1])]
    else:
        labels = [str(label).strip() for label in np.asarray(labels).reshape(-1)]
    if len(labels) != obs_matrix.shape[1]:
        raise ValueError("Galaxy PK labels length mismatches number of observable columns.")

    fiducials = dataset.get("fiducials") or {}
    if isinstance(fiducials, np.ndarray):
        fiducials = fiducials.item()
    if fiducials is None:
        fiducials = {}
    model_matrix = _build_model_matrix(model, z, labels, fiducials)
    model_vector = model_matrix.ravel(order="C")
    observed_vector = obs_matrix.ravel(order="C")
    diff = observed_vector - model_vector

    inv_cov = dataset.get("inv_cov")
    cov = dataset.get("cov")
    if inv_cov is None and cov is not None:
        inv_cov = np.linalg.inv(cov)
    if inv_cov is not None:
        inv_cov = np.asarray(inv_cov, dtype=float)
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        err = dataset.get("err")
        if err is None:
            raise ValueError("Galaxy PK dataset lacks both covariance and error arrays.")
        err = np.asarray(err, dtype=float)
        if err.shape != observed_vector.shape:
            if err.ndim == 1 and err.shape[0] == obs_matrix.shape[0]:
                err = np.repeat(err, obs_matrix.shape[1])
            else:
                raise ValueError("Galaxy PK error array shape mismatch.")
        chi2 = float(np.sum((diff / err) ** 2))

    extras = build_fit_extras(
        dataset=dataset,
        predictions=model_vector,
        observed=observed_vector,
        residuals=diff,
        additional={"labels": labels},
    )
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_galaxy_pk_fit(model, dataset)


__all__ = ["run_fit", "run_galaxy_pk_fit"]
