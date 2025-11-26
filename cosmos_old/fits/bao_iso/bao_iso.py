"""Model-neutral BAO isotropic χ² helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel
from data_interface.bao_loader import load_bao_iso_data
from data_interface.standardize import ensure_standard_dataset

DATA_ROOT = Path("data/bao_iso")
DEFAULT_FILE = "desi_bao_iso.npz"


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        value = raw.item()
        if isinstance(value, dict):
            return dict(value)
        return {"meta": value}
    return dict(raw) if isinstance(raw, dict) else {"meta": raw}


def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT / DEFAULT_FILE
    return str(candidate.expanduser().resolve())


def _load_from_npz(target: Path) -> Dict[str, Any]:
    if not target.exists():
        raise FileNotFoundError(f"BAO isotropic dataset not found at {target}")

    payload = np.load(target, allow_pickle=True)
    z = np.asarray(payload["z"], dtype=float)
    obs = np.asarray(payload["Dv_over_rd"], dtype=float)
    cov = np.asarray(payload["cov"], dtype=float) if "cov" in payload else None
    sigma = np.asarray(payload["sigma"], dtype=float) if "sigma" in payload else None

    if sigma is not None:
        err = sigma
    elif cov is not None:
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        raise ValueError("BAO isotropic dataset lacks uncertainty information")

    name_entry = payload.get("name")
    if name_entry is None:
        name = "BAO_ISO"
    else:
        name = str(name_entry.item()) if hasattr(name_entry, "item") else str(name_entry)

    metadata = _parse_metadata(payload.get("meta"))

    return {
        "name": name,
        "type": "BAO_ISO",
        "z": z,
        "obs": obs,
        "err": err,
        "cov": cov,
        "meta": metadata,
    }


def _finalize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    dataset = ensure_standard_dataset(dataset, "BAO_ISO")

    cov = dataset.get("cov")
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        dataset["cov"] = cov
        dataset["inv_cov"] = np.linalg.inv(cov)
        dataset["err"] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        err = dataset.get("err")
        if err is None:
            raise ValueError("BAO isotropic dataset lacks uncertainty information")
        dataset["err"] = np.asarray(err, dtype=float)

    return dataset


@lru_cache(maxsize=None)
def _load_bao_iso_npz(resolved_path: str) -> Dict[str, Any]:
    return _finalize_dataset(_load_from_npz(Path(resolved_path)))


@lru_cache(maxsize=None)
def _load_bao_iso_standardized() -> Dict[str, Any]:
    try:
        dataset = load_bao_iso_data()
    except FileNotFoundError:
        resolved = _resolve_dataset_path(None)
        return _load_bao_iso_npz(resolved)
    return _finalize_dataset(dataset)


def load_bao_iso_dataset(path: Path | str | None = None) -> Dict[str, Any]:
    if path is None:
        return _load_bao_iso_standardized()
    resolved = _resolve_dataset_path(path)
    return _load_bao_iso_npz(resolved)


def run_bao_iso_fit(model: CosmologyModel, dataset: Dict[str, Any] | None = None) -> tuple[float, Dict[str, np.ndarray]]:
    dataset = dataset or load_bao_iso_dataset()
    dataset = ensure_standard_dataset(dataset, "BAO_ISO")

    z = np.asarray(dataset["z"], dtype=float)
    observed = np.asarray(dataset["obs"], dtype=float)

    dv_model = np.asarray(model.DV(z), dtype=float)
    rd = float(model.sound_horizon())
    if rd <= 0.0:
        raise ValueError("Model returned a non-positive sound horizon")

    dv_over_rd_model = dv_model / rd
    diff = dv_over_rd_model - observed

    inv_cov = dataset.get("inv_cov")
    if inv_cov is not None:
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        err = dataset.get("err")
        if err is None:
            raise ValueError("BAO isotropic dataset lacks covariance and errors")
        chi2 = float(np.sum((diff / err) ** 2))

    extras = build_fit_extras(
        dataset=dataset,
        predictions=dv_over_rd_model,
        observed=observed,
        residuals=diff,
        additional={"rd": rd, "DV_over_rd_model": dv_over_rd_model},
    )
    return chi2, extras


def run_fit(model: CosmologyModel, dataset: Dict[str, Any] | None = None) -> tuple[float, Dict[str, np.ndarray]]:
    """Standard entrypoint for joint fits."""

    return run_bao_iso_fit(model, dataset)
