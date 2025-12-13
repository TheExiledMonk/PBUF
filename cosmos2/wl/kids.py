"""Helpers for KiDS-1000 weak-lensing standardization."""

from __future__ import annotations

import numpy as np

KIDS_DATA_ORDER = "xi_plus_then_xi_minus"


def _ensure_dict(obj: object) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, np.ndarray) and obj.shape == ():
        value = obj.item()
        if isinstance(value, dict):
            return value
    return {"value": obj}


def tomo_pairs(n_bins: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_bins) for j in range(i, n_bins)]


def flatten_xi_block(xi_block: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    pieces = []
    for i, j in pairs:
        pieces.append(np.asarray(xi_block[i, j, :], dtype=float))
    return np.concatenate(pieces, axis=0) if pieces else np.array([], dtype=float)


def _theta_to_radians(theta: np.ndarray, units: str | None) -> np.ndarray:
    units_clean = (units or "").strip().lower()
    if units_clean in {"arcmin", "arcminute", "arcminutes"}:
        return np.deg2rad(np.asarray(theta, dtype=float) / 60.0)
    if units_clean in {"deg", "degree", "degrees"}:
        return np.deg2rad(np.asarray(theta, dtype=float))
    return np.asarray(theta, dtype=float)


def standardize_kids1000(raw: dict) -> dict:
    """
    Convert the raw KiDS-1000 NPZ payload into the WL standard schema.

    Expected raw keys: xi_plus, xi_minus, theta, theta_units, nz, z_grid, covariance.
    """
    xi_plus = np.asarray(raw["xi_plus"], dtype=float)
    xi_minus = np.asarray(raw["xi_minus"], dtype=float)
    theta = np.asarray(raw["theta"], dtype=float)
    theta_units = str(raw.get("theta_units", "arcmin"))
    z_grid = np.asarray(raw.get("z_grid"), dtype=float)
    nz = np.asarray(raw.get("nz"), dtype=float)
    covariance = np.asarray(raw.get("covariance"), dtype=float)

    if xi_plus.shape != xi_minus.shape:
        raise ValueError("xi_plus and xi_minus must share the same shape.")
    n_bins, _, n_theta = xi_plus.shape
    pairs = tomo_pairs(n_bins)

    theta_rad = _theta_to_radians(theta, theta_units)
    xi_plus_flat = flatten_xi_block(xi_plus, pairs)
    xi_minus_flat = flatten_xi_block(xi_minus, pairs)
    data_vector = np.concatenate([xi_plus_flat, xi_minus_flat], axis=0)

    nz_rows = []
    for row in np.asarray(nz):
        nz_rows.append(np.asarray(row, dtype=float))

    meta = _ensure_dict(raw.get("meta"))
    meta = dict(meta)
    meta.setdefault("survey", "KiDS-1000")
    meta.setdefault("theta_units", theta_units)
    meta.setdefault("data_order", KIDS_DATA_ORDER)
    meta.setdefault("n_tomo_bins", n_bins)
    meta["kids_ordering_notes"] = "Flatten xi_plus(tomo_pairs,theta) then xi_minus with tomo_pairs ordered by (i<=j)."

    payload = {
        "name": "weak_lensing_kids1000",
        "type": "WL",
        "data_vector": data_vector,
        "theta_bins": theta_rad,
        "theta_bins_orig": theta,
        "theta_units": theta_units,
        "tomo_pairs": np.asarray(pairs, dtype=int),
        "covariance": covariance,
        "n_of_z": np.asarray(nz_rows, dtype=float),
        "z_grid": z_grid,
        "shear_m": np.zeros(n_bins, dtype=float),
        "meta": meta,
    }
    return payload


__all__ = [
    "standardize_kids1000",
    "tomo_pairs",
    "flatten_xi_block",
    "KIDS_DATA_ORDER",
]
