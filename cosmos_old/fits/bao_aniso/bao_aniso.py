"""Model-neutral BAO anisotropic χ² helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel
from cosmos.models.common.distance_utils import C_LIGHT
from data_interface.bao_loader import load_bao_data
from data_interface.standardize import ensure_standard_dataset

DATA_ROOT = Path("data/bao_aniso")
DEFAULT_FILE = "desi_bao_aniso.npz"


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
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            value = raw.item()
            if isinstance(value, dict):
                return dict(value)
            return {"meta": value}
        if raw.size == 1:
            candidate = raw.reshape(-1)[0]
            if isinstance(candidate, dict):
                return dict(candidate)
            return {"meta": candidate}
        return {"meta": raw.tolist()}
    return {"meta": raw}


def _detect_observables(payload: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    if "DM_over_rd" in payload and "DH_over_rd" in payload:
        return ("DM_over_rd", "DH_over_rd")
    if "DA_over_rd" in payload and "H_times_rd" in payload:
        return ("DA_over_rd", "H_times_rd")
    raise ValueError(
        "BAO anisotropic dataset must expose either (DM_over_rd & DH_over_rd) "
        "or (DA_over_rd & H_times_rd) arrays."
    )


def _canonicalize_label(label: str) -> str:
    normalized = "".join(ch for ch in label.lower() if ch.isalnum())
    if "dh" in normalized and "htimes" not in normalized:
        return "DH_over_rd"
    if "dm" in normalized:
        return "DM_over_rd"
    if "da" in normalized and "htimes" not in normalized:
        return "DA_over_rd"
    if "htimes" in normalized:
        return "H_times_rd"
    raise ValueError(f"Unknown BAO anisotropic label: {label}")


def _build_from_standardized(dataset: Dict[str, Any]) -> Dict[str, Any]:
    dataset = ensure_standard_dataset(dataset, "BAO_ANISO")

    z = np.asarray(dataset["z"], dtype=float)
    obs = np.asarray(dataset["obs"], dtype=float)
    cov = dataset.get("cov")
    if cov is None:
        raise ValueError("BAO anisotropic dataset requires a covariance matrix")
    cov = np.asarray(cov, dtype=float)

    inv_cov = np.linalg.inv(cov)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    labels_raw = dataset.get("labels")
    if labels_raw is None:
        labels_raw = dataset.get("meta", {}).get("labels")
    if labels_raw is None:
        raise ValueError("BAO anisotropic dataset requires observable labels")
    labels = np.asarray(labels_raw, dtype=object)
    if labels.shape[0] != obs.shape[0]:
        raise ValueError(
            "BAO anisotropic labels length mismatches flattened observables"
        )

    n_bins = z.size
    if n_bins == 0:
        raise ValueError("BAO anisotropic dataset contains no redshifts")
    if obs.size % n_bins != 0:
        raise ValueError("BAO anisotropic observations cannot be reshaped per redshift")
    observables_per_bin = obs.size // n_bins
    canonical_order = [
        _canonicalize_label(str(labels[idx])) for idx in range(observables_per_bin)
    ]
    observables = tuple(canonical_order)

    values = {name: np.empty(n_bins, dtype=float) for name in set(observables)}
    for idx, value in enumerate(obs):
        bin_idx = idx // observables_per_bin
        canonical = _canonicalize_label(str(labels[idx]))
        values[canonical][bin_idx] = float(value)

    metadata = _parse_metadata(dataset.get("meta"))
    name_entry = dataset.get("name")
    if name_entry is None:
        name = "BAO_ANISO"
    else:
        name = str(name_entry.item()) if hasattr(name_entry, "item") else str(name_entry)

    return {
        "name": name,
        "type": "BAO_ANISO",
        "z": z,
        "obs": obs,
        "err": err,
        "cov": cov,
        "inv_cov": inv_cov,
        "meta": metadata,
        "labels": labels,
        "observables": observables,
        "values": values,
        "observables_per_bin": observables_per_bin,
    }


def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT / DEFAULT_FILE
    return str(candidate.expanduser().resolve())


@lru_cache(maxsize=None)
def _load_bao_aniso_npz(resolved_path: str) -> Dict[str, Any]:
    return _load_from_npz(Path(resolved_path))


@lru_cache(maxsize=None)
def _load_bao_aniso_standardized() -> Dict[str, Any]:
    try:
        standardized = load_bao_data()
    except FileNotFoundError:
        resolved = _resolve_dataset_path(None)
        return _load_bao_aniso_npz(resolved)
    return _build_from_standardized(standardized)


def load_bao_aniso_dataset(path: Path | str | None = None) -> Dict[str, Any]:
    """Read a cached anisotropic BAO dataset and expose a standard dictionary."""

    if path is None:
        return _load_bao_aniso_standardized()
    resolved = _resolve_dataset_path(path)
    return _load_bao_aniso_npz(resolved)


def _load_from_npz(target: Path) -> Dict[str, Any]:
    """Legacy loader for local `data/bao_aniso` cached snapshots."""

    if not target.exists():
        raise FileNotFoundError(f"BAO anisotropic dataset not found at {target}")

    with np.load(target, allow_pickle=True) as payload:
        observables = _detect_observables(payload)
        z_raw = payload.get("z_eff")
        if z_raw is None:
            z_raw = payload.get("z")
        if z_raw is None:
            raise ValueError("BAO anisotropic dataset lacks 'z_eff' or 'z' arrays")

        z = np.asarray(z_raw, dtype=float)
        if z.ndim != 1:
            raise ValueError("BAO anisotropic redshifts must be one-dimensional")

        values: Dict[str, np.ndarray] = {}
        for name in observables:
            arr = np.asarray(payload[name], dtype=float)
            if arr.shape != z.shape:
                raise ValueError(
                    f"BAO observable '{name}' shape {arr.shape} mismatches z shape {z.shape}"
                )
            values[name] = arr

        if payload.get("cov") is None:
            raise ValueError("BAO anisotropic dataset requires a covariance matrix")

        cov = np.asarray(payload["cov"], dtype=float)
        full_size = z.size * len(observables)
        if cov.shape != (full_size, full_size):
            raise ValueError(
                f"Expected covariance shape {(full_size, full_size)}, got {cov.shape}"
            )

        inv_cov = np.linalg.inv(cov)
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        labels = np.array(
            [name for _ in range(z.size) for name in observables], dtype=object
        )

        stacked = np.empty(full_size, dtype=float)
        for idx in range(z.size):
            for jdx, name in enumerate(observables):
                stacked[idx * len(observables) + jdx] = values[name][idx]

        metadata = _parse_metadata(payload.get("meta"))
        name_entry = payload.get("name")
        if name_entry is None:
            name = "BAO_ANISO"
        else:
            name = str(name_entry.item()) if hasattr(name_entry, "item") else str(name_entry)

        return {
            "name": name,
            "type": "BAO_ANISO",
            "z": z,
            "obs": stacked,
            "err": err,
            "cov": cov,
            "inv_cov": inv_cov,
            "meta": metadata,
            "labels": labels,
            "observables": observables,
            "values": values,
            "observables_per_bin": len(observables),
        }


def _make_model_vector(model: CosmologyModel, observables: tuple[str, ...], z: np.ndarray) -> np.ndarray:
    rd = float(model.sound_horizon())
    if rd <= 0.0:
        raise ValueError("Model returned a non-positive sound horizon")

    def _dm(z_val: float) -> float:
        return float(model.DM(z_val)) / rd

    def _dh(z_val: float) -> float:
        return float(model.DH(z_val)) / rd

    def _da(z_val: float) -> float:
        return float(model.DA(z_val)) / rd

    def _h_times(z_val: float) -> float:
        return float(model.Hubble(z_val)) * rd / C_LIGHT

    dispatcher: Dict[str, Callable[[float], float]] = {
        "DM_over_rd": _dm,
        "DH_over_rd": _dh,
        "DA_over_rd": _da,
        "H_times_rd": _h_times,
    }

    total = z.size * len(observables)
    vector = np.empty(total, dtype=float)
    for idx, z_val in enumerate(z):
        for jdx, name in enumerate(observables):
            func = dispatcher.get(name)
            if func is None:
                raise ValueError(f"Unsupported BAO observable '{name}'")
            vector[idx * len(observables) + jdx] = func(float(z_val))
    return vector


def run_bao_aniso_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, np.ndarray]]:
    dataset = dataset or load_bao_aniso_dataset()

    z = np.asarray(dataset["z"], dtype=float)
    observed = np.asarray(dataset["obs"], dtype=float)
    observables = tuple(dataset["observables"])
    model_vector = _make_model_vector(model, observables, z)

    diff = model_vector - observed
    inv_cov = dataset.get("inv_cov")
    if inv_cov is not None:
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        err = dataset.get("err")
        if err is None:
            raise ValueError("BAO anisotropic dataset lacks covariance and errors")
        chi2 = float(np.sum((diff / err) ** 2))

    extras = build_fit_extras(
        dataset=dataset,
        predictions=model_vector,
        observed=observed,
        residuals=diff,
        additional={"observables": observables, "bao_aniso_model": model_vector},
    )
    return chi2, extras


def run_fit(model: CosmologyModel, dataset: Dict[str, Any] | None = None) -> tuple[float, Dict[str, np.ndarray]]:
    """Standard entrypoint for joint fits."""

    return run_bao_aniso_fit(model, dataset)
