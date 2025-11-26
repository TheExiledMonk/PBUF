"""Model-neutral compressed galaxy power spectrum χ² helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel
from data_interface.standardize import ensure_standard_dataset

DATA_ROOT = Path("data/standardized")
DEFAULT_FILE = "galaxy_pks.npz"

_LABEL_TRANSLATIONS = str.maketrans({"σ": "sigma", "Σ": "sigma"})
_SKIP_KEYS = {
    "z",
    "obs",
    "cov",
    "err",
    "meta",
    "type",
    "name",
    "labels",
    "fiducials",
    "obs_labels",
    "observable_labels",
}


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


def _normalize_label(raw_label: Any) -> str:
    label = str(raw_label).translate(_LABEL_TRANSLATIONS).strip().lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return label.strip("_")


def _prepare_obs_matrix(raw_obs: Iterable[float]) -> np.ndarray:
    obs = np.asarray(raw_obs, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape((-1, 1))
    if obs.ndim != 2:
        raise ValueError(f"Galaxy PK obs array must be 1D or 2D, got shape {obs.shape}")
    return obs


def _flatten_errors(raw_err: Iterable[float], shape: tuple[int, int]) -> np.ndarray:
    err = np.asarray(raw_err, dtype=float)
    if err.shape == shape:
        return err.ravel(order="C")
    if err.ndim == 1 and err.shape[0] == shape[0] * shape[1]:
        return err
    raise ValueError("Galaxy PK err array shape does not match the observations.")


def _extract_labels(payload: np.lib.npyio.NpzFile, obs_cols: int) -> list[str]:
    candidates = ("labels", "obs_labels", "observable_labels", "names")
    for key in candidates:
        if key in payload:
            raw = np.asarray(payload[key])
            raw = raw.reshape(-1)
            labels = [str(entry).strip() for entry in raw]
            if len(labels) != obs_cols:
                raise ValueError(
                    f"Galaxy PK dataset labels length {len(labels)} does not match obs columns {obs_cols}."
                )
            return labels
    return [f"obs_{idx}" for idx in range(obs_cols)]


def _match_length(value: Any, length: int) -> np.ndarray:
    if value is None:
        raise ValueError("Requested fiducial value is missing.")
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 1:
        return np.full(shape=length, fill_value=float(arr[0]), dtype=float)
    if arr.size == length:
        return arr
    raise ValueError(f"Fiducial array length {arr.size} mismatches expected {length}.")


def _is_fs8_label(key: str) -> bool:
    return "fs8" in key or "fsigma" in key or "f_sigma" in key


def _is_dh_label(key: str) -> bool:
    return "dh" in key and "fid" in key


def _is_dm_label(key: str) -> bool:
    return "dm" in key and "fid" in key and "dh" not in key


def _is_h_label(key: str) -> bool:
    return "h" in key and "fid" in key and "dh" not in key and "dm" not in key


def _collect_fiducials(payload: np.lib.npyio.NpzFile, n_points: int) -> Dict[str, np.ndarray]:
    fiducials: Dict[str, np.ndarray] = {}

    def _scan_entry(key: str, value: Any) -> None:
        if key.lower() in _SKIP_KEYS:
            return
        normalized = _normalize_label(key)
        if _is_dh_label(normalized) and "DH" not in fiducials:
            fiducials["DH"] = _match_length(value, n_points)
        elif _is_dm_label(normalized) and "DM" not in fiducials:
            fiducials["DM"] = _match_length(value, n_points)
        elif _is_h_label(normalized) and "H" not in fiducials:
            fiducials["H"] = _match_length(value, n_points)

    meta_candidate = payload.get("fiducials")
    if meta_candidate is not None:
        raw = meta_candidate
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, dict):
            for entry, value in raw.items():
                _scan_entry(entry, value)

    for entry in payload.files:
        if entry == "fiducials":
            continue
        _scan_entry(entry, payload[entry])

    return fiducials
def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT / DEFAULT_FILE
    return str(candidate.expanduser().resolve())


@lru_cache(maxsize=None)
def _load_galaxy_pk_dataset_cached(resolved_path: str) -> Dict[str, Any]:
    target = Path(resolved_path)
    if not target.exists():
        raise FileNotFoundError(f"Galaxy PK dataset not found at {target}")

    payload = np.load(target, allow_pickle=True)
    z = np.asarray(payload["z"], dtype=float)
    obs = _prepare_obs_matrix(payload["obs"])
    if obs.shape[0] != z.size:
        raise ValueError("Galaxy PK dataset redshifts do not match observation rows.")
    labels = _extract_labels(payload, obs.shape[1])

    cov = np.asarray(payload["cov"], dtype=float) if "cov" in payload else None
    if cov is not None:
        expected = obs.shape[0] * obs.shape[1]
        if cov.shape != (expected, expected):
            raise ValueError(f"Galaxy PK covariance shape {cov.shape} mismatches {expected}.")

    meta = _parse_metadata(payload.get("meta"))
    meta.setdefault("file", str(target))

    dataset = {
        "name": str(payload.get("name", "GalaxyPkObs")),
        "type": str(payload.get("type", "GALAXY_PK")),
        "z": z,
        "obs": obs,
        "cov": cov,
        "err": None,
        "meta": meta,
    }
    dataset = ensure_standard_dataset(dataset, "GALAXY_PK")
    dataset["labels"] = labels
    dataset["fiducials"] = _collect_fiducials(payload, len(z))
    if cov is not None and "inv_cov" not in dataset:
        dataset["inv_cov"] = np.linalg.inv(cov)

    return dataset


def load_galaxy_pk_dataset(path: Path | str | None = None) -> Dict[str, Any]:
    resolved = _resolve_dataset_path(path)
    return _load_galaxy_pk_dataset_cached(resolved)


def _build_model_matrix(
    model: CosmologyModel,
    z: np.ndarray,
    labels: list[str],
    fiducials: Dict[str, np.ndarray],
) -> np.ndarray:
    predictions = []
    for label in labels:
        predictions.append(_predict_observable(model, z, label, fiducials))
    return np.column_stack(predictions)


def _predict_observable(
    model: CosmologyModel,
    z: np.ndarray,
    label: str,
    fiducials: Dict[str, np.ndarray],
) -> np.ndarray:
    key = _normalize_label(label)
    if _is_fs8_label(key):
        return np.asarray(model.fs8(z), dtype=float)
    if _is_dh_label(key):
        dh_fid = fiducials.get("DH")
        if dh_fid is None:
            raise ValueError("Galaxy PK dataset requires DH fiducials for DH_obs.")
        return np.asarray(model.DH(z), dtype=float) / dh_fid
    if _is_dm_label(key):
        dm_fid = fiducials.get("DM")
        if dm_fid is None:
            raise ValueError("Galaxy PK dataset requires DM fiducials for DM_obs.")
        return np.asarray(model.DM(z), dtype=float) / dm_fid
    if _is_h_label(key):
        h_fid = fiducials.get("H")
        if h_fid is None:
            raise ValueError("Galaxy PK dataset requires H fiducials for H_obs.")
        return np.asarray(model.Hubble(z), dtype=float) / h_fid
    raise ValueError(f"Unsupported Galaxy PK observable label '{label}'.")


def run_galaxy_pk_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    dataset = dataset or load_galaxy_pk_dataset()
    dataset = ensure_standard_dataset(dataset, "GALAXY_PK")

    z = np.asarray(dataset["z"], dtype=float)
    obs_matrix = _prepare_obs_matrix(dataset["obs"])
    if obs_matrix.shape[0] != len(z):
        raise ValueError("Galaxy PK dataset row count does not match provided redshifts.")

    labels = dataset.get("labels")
    if labels is None:
        labels = [f"obs_{idx}" for idx in range(obs_matrix.shape[1])]
    if len(labels) != obs_matrix.shape[1]:
        raise ValueError("Galaxy PK labels length mismatches number of observable columns.")

    fiducials = dataset.get("fiducials") or {}
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
        err = _flatten_errors(err, obs_matrix.shape)
        chi2 = float(np.sum((diff / err) ** 2))

    extras = build_fit_extras(
        dataset=dataset,
        predictions=model_vector,
        observed=observed_vector,
        additional={"labels": labels, "galaxy_pk_model_vector": model_vector},
    )
    return chi2, extras


def run_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    """Standard entrypoint for joint fits."""

    return run_galaxy_pk_fit(model, dataset)
