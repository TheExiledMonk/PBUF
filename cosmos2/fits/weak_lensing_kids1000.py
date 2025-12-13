"""Full KiDS-1000 weak lensing ξ± fit (model-agnostic backend)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras
from cosmos2.wl.backend import WeakLensingBackend
from cosmos2.wl.kids import tomo_pairs
from cosmos2.wl.scale_cuts import (
    apply_scale_cuts,
    build_custom_scale_cuts,
    build_scale_cut_mask,
    kids_default_scale_cuts,
)
from cosmos2.wl.theory import compute_shear_predictions


def _prepare_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(dataset)
    meta = payload.get("meta")
    if isinstance(meta, np.ndarray) and meta.shape == ():
        payload["meta"] = meta.item()
    elif meta is None:
        payload["meta"] = {}
    return payload


def run_wl_kids1000_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = _prepare_dataset(dataset or get_dataset("weak_lensing_kids1000"))

    data_vector = np.asarray(dataset["data_vector"], dtype=float).reshape(-1)
    covariance = np.asarray(dataset["covariance"], dtype=float)
    inv_cov = np.asarray(dataset.get("inv_cov") or np.linalg.inv(covariance), dtype=float)
    theta_bins = np.asarray(dataset["theta_bins"], dtype=float)
    n_of_z = np.asarray(dataset["n_of_z"], dtype=float)
    z_grid = np.asarray(dataset["z_grid"], dtype=float)
    shear_m = np.asarray(dataset.get("shear_m", np.zeros(n_of_z.shape[0])), dtype=float)
    meta_raw = dataset.get("meta", {}) or {}
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    wl_cfg_raw = meta.get("wl", {}) if isinstance(meta, dict) else {}
    wl_cfg = wl_cfg_raw if isinstance(wl_cfg_raw, dict) else {}
    photo_z_shifts = wl_cfg.get("photo_z_shifts")
    ia_params = wl_cfg.get("ia_params")
    include_ia = bool(wl_cfg.get("include_ia", False))
    use_fftlog = bool(wl_cfg.get("use_fftlog", True))
    ell_min = int(wl_cfg.get("ell_min", 2))
    ell_max = int(wl_cfg.get("ell_max", 3000))
    fftlog_ell_samples = wl_cfg.get("fftlog_ell_samples")
    apply_cuts = bool(wl_cfg.get("apply_scale_cuts", False))
    scale_cut_profile = wl_cfg.get("scale_cut_profile", "kids_default")
    scale_cut_table = wl_cfg.get("scale_cut_table")
    use_nonlinear = bool(wl_cfg.get("nonlinear", False))

    backend = WeakLensingBackend(model)
    xi_plus, xi_minus, model_vector = compute_shear_predictions(
        backend,
        data_vector,
        n_of_z,
        z_grid,
        theta_bins,
        shear_m=shear_m,
        photo_z_shifts=photo_z_shifts,
        ia_params=ia_params,
        include_ia=include_ia,
        use_fftlog=use_fftlog,
        fftlog_ell_samples=fftlog_ell_samples,
        ell_min=ell_min,
        ell_max=ell_max,
        nonlinear=use_nonlinear,
    )

    if apply_cuts:
        pairs_arr = np.asarray(dataset.get("tomo_pairs") or tomo_pairs(n_of_z.shape[0]), dtype=int)
        pairs = [tuple(map(int, pair)) for pair in pairs_arr]
        if scale_cut_table and isinstance(scale_cut_table, dict):
            default_cuts = kids_default_scale_cuts(
                n_of_z.shape[0],
                xi_plus_min_arcmin=float(wl_cfg.get("xi_plus_min_arcmin", 0.5)),
                xi_minus_min_arcmin=float(wl_cfg.get("xi_minus_min_arcmin", 4.2)),
                xi_plus_max_arcmin=float(wl_cfg.get("xi_plus_max_arcmin", 300.0)),
                xi_minus_max_arcmin=float(wl_cfg.get("xi_minus_max_arcmin", 300.0)),
                use_official_kids_minus=bool(wl_cfg.get("use_official_kids_minus", True)),
            )
            # Expect values in arcmin; convert to radians here.
            table_rad = {}
            for key, val in scale_cut_table.items():
                if isinstance(key, str) and "-" in key:
                    parts = key.split("-")
                    try:
                        key_tuple = (int(parts[0]), int(parts[1]))
                    except Exception:
                        continue
                else:
                    key_tuple = tuple(key) if isinstance(key, (list, tuple)) else None
                if key_tuple is None or len(key_tuple) != 2:
                    continue
                arr = tuple(val) if isinstance(val, (list, tuple)) else None
                if arr is None:
                    continue
                # Convert arcmin -> rad if numeric
                converted = []
                for entry in arr:
                    if entry is None:
                        converted.append(None)
                        continue
                    try:
                        converted.append(float(entry) * np.pi / (180.0 * 60.0))
                    except Exception:
                        converted.append(None)
                table_rad[key_tuple] = tuple(converted)
            cuts = build_custom_scale_cuts(n_of_z.shape[0], table_rad, default=default_cuts)
        elif scale_cut_profile in {"kids_default", "kids_official"}:
            cuts = kids_default_scale_cuts(
                n_of_z.shape[0],
                xi_plus_min_arcmin=float(wl_cfg.get("xi_plus_min_arcmin", 0.5)),
                xi_minus_min_arcmin=float(wl_cfg.get("xi_minus_min_arcmin", 4.2)),
                xi_plus_max_arcmin=float(wl_cfg.get("xi_plus_max_arcmin", 300.0)),
                xi_minus_max_arcmin=float(wl_cfg.get("xi_minus_max_arcmin", 300.0)),
                use_official_kids_minus=bool(wl_cfg.get("use_official_kids_minus", True)),
            )
        elif isinstance(scale_cut_profile, dict):
            cuts = {tuple(map(int, k if isinstance(k, tuple) else k)): tuple(v) for k, v in scale_cut_profile.items()}
        else:
            cuts = kids_default_scale_cuts(n_of_z.shape[0])
        mask = build_scale_cut_mask(theta_bins, pairs, cuts)
        data_vector, covariance = apply_scale_cuts(data_vector, covariance, mask)
        model_vector = model_vector[mask.combined]
        inv_cov = np.linalg.inv(covariance)
        theta_bins = theta_bins  # unchanged; kept for reporting

    residuals = model_vector - data_vector
    chi2 = float(residuals.T @ inv_cov @ residuals)
    wl_flags = {
        "power_spectrum": "eh+halofit" if use_nonlinear else "eh+linear",
        "xi_method": "fftlog" if use_fftlog else "bessel",
        "include_ia": bool(include_ia),
        "photo_z_shift": bool(photo_z_shifts is not None),
        "shear_m_applied": True,
        "scale_cuts_applied": bool(apply_cuts),
        "scale_cut_profile": scale_cut_profile if apply_cuts else None,
        "xi_plus_min_arcmin": wl_cfg.get("xi_plus_min_arcmin", 0.5) if apply_cuts else None,
        "xi_minus_min_arcmin": wl_cfg.get("xi_minus_min_arcmin", 4.2) if apply_cuts else None,
        "scale_cut_table": scale_cut_table if apply_cuts else None,
        "ell_range": (ell_min, ell_max),
    }
    extras = build_fit_extras(
        dataset=dataset,
        predictions=model_vector,
        observed=data_vector,
        residuals=residuals,
        additional={
            "xi_plus": xi_plus,
            "xi_minus": xi_minus,
            "data_order": dataset.get("meta", {}).get("data_order"),
            "wl_flags": wl_flags,
        },
    )
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_wl_kids1000_fit(model, dataset)


__all__ = ["run_fit", "run_wl_kids1000_fit"]
