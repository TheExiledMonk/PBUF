"""
High-detail joint joint-fit capture for LCDM and PBUF models.

This module extends the existing joint optimisation workflow to emit a
fully-instrumented record of a run, including repro metadata, dataset-level
predictives, diagnostics, and optimizer traces.  It reuses the established
`fit_joint` optimiser (including its acceptance logic) so that the new reports
remain 100% compatible with the legacy pipeline.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from cosmos.fits._dataset_loader import STANDARDIZED_DIR, load_cmb_dataset
from cosmos.fits.joint.likelihoods import compute_joint_chi2
from cosmos.fits.joint.optimizer import fit_joint
from cosmos.helper.constants import C_LIGHT
from cosmos.helper.distances import (
    comoving_distance,
    luminosity_distance,
    sound_horizon,
    transverse_comoving_distance,
)
from cosmos.helper.growth import fsigma8, growth_factor, growth_rate
from cosmos.lcdm.model import LCDM
from cosmos.optim.parameter_defaults import (
    LCDM_PARAMETER_DEFAULTS,
    PBUF_PARAMETER_DEFAULTS,
    SIGMA8_PLANCK,
)
from cosmos.pbuf.model import PBUF
from cosmos.physchecks.phase6a import phase6a_passes

from cosmos.fits.cmb.observables import cmb_observables, redshift_star
from cosmos.fits.sn.pantheon.loader import load_pantheon_data
from cosmos.fits.sn.pantheon.observables import compute_pantheon_mu_model
from cosmos.fits.sn.sh0es.loader import load_sh0es_data
from cosmos.fits.sn.sh0es.observables import (
    compute_sh0es_mu_model,
    extract_model_h0,
)
from cosmos.fits.bao.iso.data_loader import load_bao_iso_data
from cosmos.fits.bao.iso.observables import compute_bao_dv_over_rd
from cosmos.fits.bao.aniso.data_loader import load_bao_aniso_data
from cosmos.fits.bao.aniso.observables import compute_bao_anisotropic_observables
from cosmos.fits.cc.data_loader import load_cc_data
from cosmos.fits.cc.observables import compute_cc_hubble_model
from cosmos.fits.rsd.data_loader import load_rsd_data
from cosmos.fits.rsd.observables import compute_rsd_observable

from data_interface.standardize import ensure_standard_dataset


DEFAULT_DATASET_ORDER = ("cmb", "pantheon", "sh0es", "iso", "aniso", "cc", "rsd")

DATASET_FILE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "cmb": ("cmb.npz", "planck2018_distance_priors.npz", "cmb_planck2018.npz"),
    "pantheon": (
        "sn_pantheon_shoes.npz",
        "sn_pantheonplus.npz",
        "sn_pantheon_full.npz",
        "sn_pantheon.npz",
    ),
    "sh0es": ("sn_pantheon_shoes.npz", "sn_sh0es.npz", "sh0es.npz"),
    "iso": ("bao_iso.npz", "bao_iso_dr16.npz"),
    "aniso": ("bao_aniso.npz", "bao_aniso_dr16.npz"),
    "cc": ("cc.npz", "cc_compilation.npz"),
    "rsd": ("rsd.npz", "rsd_compilation.npz"),
}


@dataclass
class DatasetContext:
    """Container for dataset payload + provenance information."""

    key: str
    data: Dict[str, Any]
    manifest: Dict[str, Any]


def _hash_file(path: Path) -> str:
    import hashlib

    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _resolve_dataset_manifest(key: str) -> Dict[str, Any]:
    candidates = DATASET_FILE_CANDIDATES.get(key, ())
    found_path: Optional[Path] = None
    for candidate in candidates:
        path = STANDARDIZED_DIR / candidate
        if path.exists():
            found_path = path
            break
    manifest: Dict[str, Any] = {"key": key}
    if found_path is None:
        manifest["status"] = "missing"
        manifest["candidates"] = [str(STANDARDIZED_DIR / c) for c in candidates]
    else:
        manifest["status"] = "present"
        manifest["path"] = str(found_path)
        manifest["sha256"] = _hash_file(found_path)
    return manifest


def _load_dataset(key: str) -> DatasetContext:
    loaders = {
        "cmb": load_cmb_dataset,
        "pantheon": load_pantheon_data,
        "sh0es": load_sh0es_data,
        "iso": load_bao_iso_data,
        "aniso": load_bao_aniso_data,
        "cc": load_cc_data,
        "rsd": load_rsd_data,
    }
    dataset_type_map = {
        "cmb": "CMB",
        "pantheon": "SN",
        "sh0es": "SN",
        "iso": "BAO_ISO",
        "aniso": "BAO_ANISO",
        "cc": "CC",
        "rsd": "RSD",
    }
    loader = loaders[key]
    raw = loader()
    manifest = _resolve_dataset_manifest(key)
    if key == "sh0es" and "H0_obs" in raw:
        data = dict(raw)
        data.setdefault("name", "SH0ES_prior")
        data.setdefault("type", "SN")
        manifest["n_data"] = int(data.get("n", 1))
        manifest["name"] = data.get("name", "SH0ES_prior")
        manifest["meta"] = {"mode": "H0_prior"}
        return DatasetContext(key=key, data=data, manifest=manifest)
    data = ensure_standard_dataset(raw, dataset_type_map[key])
    n_points = data.get("n_data")
    if n_points is None:
        obs = data.get("obs")
        if obs is not None:
            n_points = int(np.asarray(obs).shape[0])
        else:
            z_values = data.get("z")
            n_points = int(np.asarray(z_values).shape[0]) if z_values is not None else 0
    manifest["n_data"] = int(n_points)
    manifest["name"] = data.get("name", key)
    manifest["meta"] = data.get("meta", {})
    return DatasetContext(key=key, data=data, manifest=manifest)


def _gather_environment_metadata() -> Dict[str, Any]:
    python_version = platform.python_version()
    numpy_version = np.__version__
    try:
        import scipy
    except ImportError:  # pragma: no cover - scipy is required but guard anyway
        scipy_version = "missing"
    else:
        scipy_version = scipy.__version__

    git_commit = None
    git_dirty = None
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        git_status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        git_dirty = bool(git_status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"
        git_dirty = None

    env_versions = {
        "python": python_version,
        "numpy": numpy_version,
        "scipy": scipy_version,
        "platform": platform.platform(),
    }
    env_hash = _hash_text(json.dumps(env_versions, sort_keys=True))

    return {
        "python_version": python_version,
        "numpy_version": numpy_version,
        "scipy_version": scipy_version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "env_hash": env_hash,
    }


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _instantiate_model(model_type: str, params: Mapping[str, Any]):
    params = dict(params)
    H0 = float(params["H0"])
    h = H0 / 100.0
    omega_b = float(params.get("Obh2", 0.02237)) / (h ** 2)
    Ok0 = float(params.get("Ok0", 0.0))
    Or0 = float(params.get("Or0", 9.2e-5))
    if model_type == "lcdm":
        return LCDM(
            omega_m=float(params["Om0"]),
            omega_lambda=float(params.get("Ol0", 1.0 - params["Om0"] - Or0 - Ok0)),
            h=h,
            omega_k=Ok0,
            omega_r=Or0,
            omega_b=omega_b,
        )
    if model_type == "pbuf":
        return PBUF(
            omega_m=float(params["Om0"]),
            h=h,
            alpha=float(params.get("alpha", PBUF_PARAMETER_DEFAULTS["alpha"])),
            Rmax=float(params.get("Rmax", PBUF_PARAMETER_DEFAULTS["Rmax"])),
            k_sat=float(params.get("k_sat", PBUF_PARAMETER_DEFAULTS["k_sat"])),
            eps0=float(params.get("eps0", PBUF_PARAMETER_DEFAULTS.get("eps0", 0.7))),
            n_alpha=float(params.get("n_alpha", 0.0)),
            n_eps=float(params.get("n_eps", 0.0)),
            n_R=float(params.get("n_R", 0.0)),
            omega_k=Ok0,
            omega_r=Or0,
            omega_b=omega_b,
        )
    raise ValueError(f"Unknown model type '{model_type}'")


def _safe_float(value: Any) -> Optional[float]:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(fval) or math.isinf(fval):
        return None
    return fval


def _serialize_array(arr: np.ndarray) -> List[float]:
    if arr is None:
        return []
    np_arr = np.asarray(arr)
    if np_arr.dtype == object:
        result: List[Optional[float]] = []
        for item in np_arr.ravel():
            try:
                val = float(item)
            except (TypeError, ValueError):
                result.append(None)
                continue
            if math.isfinite(val):
                result.append(val)
            else:
                result.append(None)
        return result
    return [float(x) if np.isfinite(x) else None for x in np_arr.ravel()]


def _deep_make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _deep_make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_deep_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _chi2_breakdown(model_instance, dataset_keys: Sequence[str]) -> Tuple[float, Dict[str, float]]:
    chi2_total, breakdown = compute_joint_chi2(model_instance, datasets=list(dataset_keys), verbose=False)
    clean_breakdown = {str(k): float(v) for k, v in breakdown.items()}
    return float(chi2_total), clean_breakdown


def _compute_degrees_of_freedom(breakdown: Dict[str, float], datasets: Dict[str, DatasetContext], n_params: int) -> Dict[str, Any]:
    n_data_total = 0
    per_dataset = {}
    for key, chi2_val in breakdown.items():
        ctx = datasets.get(key)
        n_points = ctx.manifest.get("n_data", 0) if ctx else 0
        per_dataset[key] = {
            "chi2": float(chi2_val),
            "n_data": n_points,
            "reduced_chi2": float(chi2_val) / max(n_points - n_params, 1) if n_points else None,
        }
        n_data_total += n_points
    dof = max(n_data_total - n_params, 1)
    return {
        "per_dataset": per_dataset,
        "n_data_total": n_data_total,
        "dof": dof,
    }


def _pantheon_predictives(model, params: Mapping[str, Any], dataset: DatasetContext) -> Dict[str, Any]:
    data = dataset.data
    z = np.asarray(data["z"], dtype=float)
    mu_obs = np.asarray(data.get("obs_abs", data.get("obs")), dtype=float)
    mu_model = compute_pantheon_mu_model(model, z, M=0.0)
    residuals = mu_obs - mu_model
    cov = np.asarray(data.get("cov"))
    err = np.asarray(data.get("err")) if data.get("err") is not None else None
    if cov is not None and cov.size:
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = None
    else:
        cov_inv = None

    delta_M = 0.0
    cov_weighted = None
    norm_resid = None
    if cov_inv is not None:
        ones = np.ones_like(residuals)
        denom = float(ones.T @ cov_inv @ ones)
        if denom > 0:
            delta_M = float((ones.T @ cov_inv @ residuals) / denom)
        cov_weighted = cov_inv @ (residuals - delta_M)
    elif err is not None and err.size:
        weights = 1.0 / np.clip(err, 1e-12, None)
        denom = np.sum(weights)
        if denom > 0:
            delta_M = float(np.sum(weights * residuals) / denom)
        norm_resid = (residuals - delta_M) / np.clip(err, 1e-12, None)

    adjusted = residuals - delta_M
    if norm_resid is None and err is not None:
        norm_resid = adjusted / np.clip(err, 1e-12, None)
    if cov_weighted is None and cov_inv is not None:
        cov_weighted = cov_inv @ adjusted

    # Binned RMS vs z (10 uniform bins)
    bin_edges = np.linspace(z.min(), z.max(), 11) if z.size else np.array([0, 1])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_rms = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (z >= lo) & (z < hi)
        if not np.any(mask):
            bin_rms.append(None)
            continue
        rms = float(np.sqrt(np.mean(adjusted[mask] ** 2)))
        bin_rms.append(rms)

    outlier_indices = np.argsort(-np.abs(adjusted))[:10]
    outliers = [
        {
            "index": int(idx),
            "z": float(z[idx]),
            "delta_mu": float(adjusted[idx]),
            "normalized": float(norm_resid[idx]) if norm_resid is not None else None,
        }
        for idx in outlier_indices
    ]

    return {
        "z": _serialize_array(z),
        "mu_obs": _serialize_array(mu_obs),
        "mu_model": _serialize_array(mu_model),
        "delta_mu": _serialize_array(adjusted),
        "normalized_residuals": _serialize_array(norm_resid) if norm_resid is not None else [],
        "cov_weighted_residuals": _serialize_array(cov_weighted) if cov_weighted is not None else [],
        "delta_M": delta_M,
        "mask": data.get("mask", None),
        "binned_rms": {
            "z_centers": [float(x) for x in bin_centers],
            "rms": bin_rms,
        },
        "outliers": outliers,
    }


def _shoes_predictives(model, params: Mapping[str, Any], dataset: DatasetContext) -> Dict[str, Any]:
    if dataset.data.get("H0_obs") is not None:
        H0_model = extract_model_h0(model, params)
        H0_obs = float(dataset.data["H0_obs"])
        H0_err = float(dataset.data["H0_err"])
        chi2 = ((H0_model - H0_obs) / H0_err) ** 2
        return {
            "mode": "prior",
            "H0_model": float(H0_model),
            "H0_obs": H0_obs,
            "H0_err": H0_err,
            "chi2": float(chi2),
            "posterior_mean": float(H0_model),
            "posterior_sigma": H0_err,
        }

    data = dataset.data
    z = np.asarray(data["z"], dtype=float)
    mu_obs = np.asarray(data["obs"], dtype=float)
    mu_model = compute_sh0es_mu_model(model, z, M=0.0)
    residuals = mu_obs - mu_model
    err = np.asarray(data.get("err"))
    cov = np.asarray(data.get("cov")) if data.get("cov") is not None else None
    if cov is not None and cov.size:
        cov_inv = np.linalg.inv(cov)
        chi2 = float(residuals.T @ cov_inv @ residuals)
    else:
        err_safe = np.clip(err, 1e-12, None)
        chi2 = float(np.sum((residuals / err_safe) ** 2))
    return {
        "mode": "local_sn",
        "z": _serialize_array(z),
        "mu_obs": _serialize_array(mu_obs),
        "mu_model": _serialize_array(mu_model),
        "delta_mu": _serialize_array(residuals),
        "chi2": chi2,
    }


def _bao_iso_predictives(model, dataset: DatasetContext) -> Dict[str, Any]:
    data = dataset.data
    z = np.asarray(data["z"], dtype=float)
    obs = np.asarray(data["obs"], dtype=float)
    pred = compute_bao_dv_over_rd(model, z)
    err = np.asarray(data.get("err"))
    cov = np.asarray(data.get("cov")) if data.get("cov") is not None else None
    cov_weighted = None
    if cov is not None and cov.size:
        cov_inv = np.linalg.inv(cov)
        residual = obs - pred
        chi2 = float(residual.T @ cov_inv @ residual)
        cov_weighted = cov_inv @ residual
        point_chi2 = []
    else:
        err_safe = np.clip(err, 1e-12, None)
        residuals = (obs - pred) / err_safe
        chi2 = float(np.sum(residuals ** 2))
        point_chi2 = residuals ** 2

    outlier_idx = np.argsort(-np.abs(obs - pred))[:5]
    outliers = [
        {
            "index": int(idx),
            "z": float(z[idx]),
            "obs": float(obs[idx]),
            "model": float(pred[idx]),
            "residual": float(obs[idx] - pred[idx]),
        }
        for idx in outlier_idx
    ]

    return {
        "z": _serialize_array(z),
        "obs_DV_over_rd": _serialize_array(obs),
        "model_DV_over_rd": _serialize_array(pred),
        "residuals": _serialize_array(obs - pred),
        "point_chi2": _serialize_array(point_chi2),
        "chi2": chi2,
        "ap_scaling": dataset.data.get("meta", {}).get("ap_scaling"),
        "cov_weighted_residuals": _serialize_array(cov_weighted),
        "outliers": outliers,
    }


def _bao_aniso_predictives(model, dataset: DatasetContext) -> Dict[str, Any]:
    data = dataset.data
    raw_obs = np.asarray(data["obs"], dtype=float)
    z_raw = np.asarray(data["z"], dtype=float)
    if raw_obs.ndim == 1 and raw_obs.size % 2 == 0:
        obs_matrix = raw_obs.reshape(-1, 2)
    else:
        obs_matrix = raw_obs
    if z_raw.ndim == 1 and z_raw.size == obs_matrix.shape[0] * 2:
        z = z_raw.reshape(-1, 2)[:, 0]
    else:
        z = z_raw
    obs_dm = np.asarray(obs_matrix[:, 0], dtype=float)
    obs_dh = np.asarray(obs_matrix[:, 1], dtype=float)
    model_vals = compute_bao_anisotropic_observables(model, z)
    dm_model = model_vals["DM_over_rd"]
    dh_model = model_vals["DH_over_rd"]
    residual_dm = obs_dm - dm_model
    residual_dh = obs_dh - dh_model
    err = np.asarray(data.get("err"))
    if err is not None and err.ndim == 2:
        err_dm = err[:, 0]
        err_dh = err[:, 1]
    else:
        err_dm = err_dh = None

    point_chi2 = []
    if err_dm is not None and err_dh is not None:
        point_chi2 = (residual_dm / np.clip(err_dm, 1e-12, None)) ** 2 + (
            residual_dh / np.clip(err_dh, 1e-12, None)
        ) ** 2
    else:
        point_chi2 = [None] * len(z)

    return {
        "z": _serialize_array(z),
        "obs_DM_over_rd": _serialize_array(obs_dm),
        "obs_DH_over_rd": _serialize_array(obs_dh),
        "model_DM_over_rd": _serialize_array(dm_model),
        "model_DH_over_rd": _serialize_array(dh_model),
        "residual_DM": _serialize_array(residual_dm),
        "residual_DH": _serialize_array(residual_dh),
        "point_chi2": _serialize_array(point_chi2),
        "chi2": None,  # supplied by breakdown
    }


def _cmb_predictives(model, dataset: DatasetContext) -> Dict[str, Any]:
    obs = cmb_observables(model)
    data = dataset.data
    mean = np.asarray(data["obs"], dtype=float)
    cov = np.asarray(data["cov"], dtype=float)
    data_vec = np.array([obs["R"], obs["la"], obs["theta_star"]])
    diff = data_vec - mean
    cov_inv = np.linalg.inv(0.5 * (cov + cov.T))
    chi2 = float(diff.T @ cov_inv @ diff)
    return {
        "observables": {
            "R": float(obs["R"]),
            "lA": float(obs["la"]),
            "theta_star": float(obs["theta_star"]),
            "z_star": float(obs["z_star"]),
            "Omega_b_h2": float(model.omega_b * (model.h ** 2)),
        },
        "residual_vector": _serialize_array(diff),
        "chi2": chi2,
        "prior_type": data.get("meta", {}).get("prior", "Planck2018"),
    }


def _cc_predictives(model, dataset: DatasetContext) -> Dict[str, Any]:
    data = dataset.data
    z = np.asarray(data["z"], dtype=float)
    obs = np.asarray(data["obs"], dtype=float)
    model_h = compute_cc_hubble_model(model, z)
    err = np.asarray(data.get("err"))
    residuals = obs - model_h
    if err is not None:
        err_arr = np.asarray(err, dtype=float).reshape(-1)
        point_chi2 = []
        for res, sigma in zip(residuals, err_arr):
            if np.isfinite(sigma) and sigma > 0:
                point_chi2.append((res / sigma) ** 2)
            else:
                point_chi2.append(None)
    else:
        point_chi2 = [None] * residuals.size
    return {
        "z": _serialize_array(z),
        "H_obs": _serialize_array(obs),
        "H_model": _serialize_array(model_h),
        "residuals": _serialize_array(residuals),
        "point_chi2": _serialize_array(point_chi2),
    }


def _rsd_predictives(model, dataset: DatasetContext, sigma8_0: float) -> Dict[str, Any]:
    data = dataset.data
    z = np.asarray(data["z"], dtype=float)
    obs = np.asarray(data["obs"], dtype=float)
    model_fs8 = compute_rsd_observable(model, z, sigma8_0=sigma8_0)
    err = np.asarray(data.get("err"))
    residuals = obs - model_fs8
    if err is not None:
        err_arr = np.asarray(err, dtype=float).reshape(-1)
        point_chi2 = []
        for res, sigma in zip(residuals, err_arr):
            if np.isfinite(sigma) and sigma > 0:
                point_chi2.append((res / sigma) ** 2)
            else:
                point_chi2.append(None)
    else:
        point_chi2 = [None] * residuals.size
    return {
        "z": _serialize_array(z),
        "fsigma8_obs": _serialize_array(obs),
        "fsigma8_model": _serialize_array(model_fs8),
        "residuals": _serialize_array(residuals),
        "point_chi2": _serialize_array(point_chi2),
        "sigma8_convention": "sigma8(z)=sigma8_0 * D(z)",
    }


def _geometry_traces(model, grid: np.ndarray) -> Dict[str, Any]:
    Hz = np.array([model.H(z) for z in grid])
    H0 = model.H(0.0)
    Ez = Hz / H0
    DM = np.array([transverse_comoving_distance(z, model) for z in grid])
    DH = C_LIGHT / 1000.0 / Hz  # in Mpc
    DL = luminosity_distance(grid, model)
    return {
        "z_grid": _serialize_array(grid),
        "H_z": _serialize_array(Hz),
        "E_z": _serialize_array(Ez),
        "D_M": _serialize_array(DM),
        "D_H": _serialize_array(DH),
        "D_L": _serialize_array(DL),
    }


def _growth_traces(model, grid: np.ndarray, sigma8_0: float) -> Dict[str, Any]:
    D_a = growth_factor(grid, model, sigma8_0=sigma8_0)
    f_a = growth_rate(grid, model)
    fs8 = fsigma8(grid, model, sigma8_0=sigma8_0)
    sigma8_z = sigma8_0 * D_a
    return {
        "z_grid": _serialize_array(grid),
        "growth_factor": _serialize_array(D_a),
        "growth_rate": _serialize_array(f_a),
        "fsigma8": _serialize_array(fs8),
        "sigma8_z": _serialize_array(sigma8_z),
        "sigma8_0": float(sigma8_0),
    }


def _elastic_diagnostics(model: PBUF, params: Mapping[str, Any], grid: np.ndarray) -> Dict[str, Any]:
    omega_sigma = np.array([model.omega_sigma(1.0 / (1.0 + z)) for z in grid])
    rho_el = np.array([model.elastic_energy_density(z) for z in grid])
    H_vals = np.array([model.H(z) for z in grid])
    ratio = rho_el / np.clip(H_vals ** 2, 1e-12, None)
    curvature = np.gradient(np.gradient(H_vals, grid), grid)
    first_grad = np.gradient(H_vals, grid)
    ratio_knee = np.abs(curvature) / (np.abs(first_grad) + 1e-12)
    return {
        "omega_sigma": _serialize_array(omega_sigma),
        "rho_el_over_H2": _serialize_array(ratio),
        "rho_el": _serialize_array(rho_el),
        "knee_ratio": _serialize_array(ratio_knee),
        "grid": _serialize_array(grid),
        "summary": {
            "omega_sigma_min": float(np.min(omega_sigma)),
            "omega_sigma_max": float(np.max(omega_sigma)),
            "rho_el_over_H2_max": float(np.max(ratio)),
            "knee_ratio_max": float(np.max(ratio_knee)),
        },
    }


def _phase6a_summary(model: PBUF, params: Mapping[str, Any]) -> Dict[str, Any]:
    helpers = {
        "H_of_z": model.H,
        "rho_elastic_of_z": model.elastic_energy_density,
    }
    passes = phase6a_passes("pbuf", params, helpers, debug=False)
    grid = np.linspace(0, 4, 41)
    omega_sigma = np.array([model.omega_sigma(1.0 / (1.0 + z)) for z in grid])
    rho_el = np.array([model.elastic_energy_density(z) for z in grid])
    H_vals = np.array([model.H(z) for z in grid])
    knee = np.gradient(np.gradient(H_vals, grid), grid)
    first_grad = np.gradient(H_vals, grid)
    ratio = np.abs(knee) / (np.abs(first_grad) + 1e-12)
    return {
        "status": bool(passes),
        "omega_sigma_min": float(np.min(omega_sigma)),
        "rho_el_over_H2_max": float(np.max(rho_el / np.clip(H_vals ** 2, 1e-12, None))),
        "knee_ratio_max": float(np.max(ratio)),
        "grid": _serialize_array(grid),
    }


def _profile_likelihood(
    model_type: str,
    base_params: Mapping[str, Any],
    datasets: Sequence[str],
    primary: str,
    secondary: Optional[str] = None,
    span: float = 0.05,
    num: int = 11,
) -> Dict[str, Any]:
    params = dict(base_params)
    base_value_primary = float(params[primary])
    primary_grid = np.linspace(
        base_value_primary * (1 - span),
        base_value_primary * (1 + span),
        num,
    )
    results = []
    if secondary:
        base_value_secondary = float(params[secondary])
        secondary_grid = np.linspace(
            base_value_secondary * (1 - span),
            base_value_secondary * (1 + span),
            num,
        )
        for pv in primary_grid:
            row = []
            for sv in secondary_grid:
                params[primary] = pv
                params[secondary] = sv
                model = _instantiate_model(model_type, params)
                chi2, _ = _chi2_breakdown(model, datasets)
                row.append(float(chi2))
            results.append(row)
        return {
            "primary": primary,
            "secondary": secondary,
            "primary_grid": [float(x) for x in primary_grid],
            "secondary_grid": [float(x) for x in secondary_grid],
            "chi2_grid": [[float(val) for val in row] for row in results],
        }
    for pv in primary_grid:
        params[primary] = pv
        model = _instantiate_model(model_type, params)
        chi2, _ = _chi2_breakdown(model, datasets)
        results.append(float(chi2))
    return {
        "primary": primary,
        "primary_grid": [float(x) for x in primary_grid],
        "chi2": results,
    }


def _sensitivity_scan(
    model_type: str,
    base_params: Mapping[str, Any],
    datasets: Sequence[str],
    rel_step: float,
    chi2_base: float,
) -> Dict[str, Any]:
    results = {}
    for key, value in base_params.items():
        if key == "sigma8_0":
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if abs(value) < 1e-12:
            continue
        step = max(abs(value) * rel_step, rel_step)
        step = max(step, 1e-6)
        params_plus = dict(base_params)
        params_minus = dict(base_params)
        params_plus[key] = value + step
        positive_params = {"H0", "Om0", "Or0", "Ol0", "alpha", "Rmax", "k_sat", "eps0"}
        if key in positive_params:
            candidate_minus = value - step
            if candidate_minus <= 0:
                candidate_minus = value * (1.0 - rel_step * 0.5)
                candidate_minus = max(candidate_minus, 1e-8)
            params_minus[key] = candidate_minus
        else:
            params_minus[key] = value - step
        chi2_plus, _ = _chi2_breakdown(_instantiate_model(model_type, params_plus), datasets)
        chi2_minus, _ = _chi2_breakdown(_instantiate_model(model_type, params_minus), datasets)
        results[key] = {
            "step": step,
            "chi2_plus": chi2_plus,
            "chi2_minus": chi2_minus,
            "delta_plus": chi2_plus - chi2_base,
            "delta_minus": chi2_minus - chi2_base,
        }
    return results


def _approximate_hessian(
    model_type: str,
    base_params: Mapping[str, Any],
    datasets: Sequence[str],
    relative_step: float = 1e-3,
) -> Dict[str, Any]:
    params = dict(base_params)
    keys = [k for k in params.keys() if k not in {"sigma8_0"}]
    base_vector = np.array([float(params[k]) for k in keys], dtype=float)
    n = len(keys)
    if n == 0:
        return {"parameters": keys, "hessian": [], "covariance": []}

    def evaluate(vec: np.ndarray) -> float:
        for idx, key in enumerate(keys):
            params[key] = vec[idx]
        model = _instantiate_model(model_type, params)
        chi2, _ = _chi2_breakdown(model, datasets)
        return chi2

    hessian = np.zeros((n, n), dtype=float)
    f0 = evaluate(base_vector)
    step_sizes = np.maximum(np.abs(base_vector) * relative_step, 1e-6)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step_sizes[i]
        fp = evaluate(base_vector + ei)
        fm = evaluate(base_vector - ei)
        hessian[i, i] = (fp - 2 * f0 + fm) / (step_sizes[i] ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step_sizes[j]
            fpp = evaluate(base_vector + ei + ej)
            fpm = evaluate(base_vector + ei - ej)
            fmp = evaluate(base_vector - ei + ej)
            fmm = evaluate(base_vector - ei - ej)
            mixed = (fpp - fpm - fmp + fmm) / (4 * step_sizes[i] * step_sizes[j])
            hessian[i, j] = mixed
            hessian[j, i] = mixed

    # Covariance approximation: invert Hessian (guard singular)
    try:
        cov = np.linalg.inv(0.5 * (hessian + hessian.T))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(0.5 * (hessian + hessian.T))

    return {
        "parameters": keys,
        "hessian": [[float(val) for val in row] for row in hessian],
        "covariance": [[float(val) for val in row] for row in cov],
    }


def _top_n_trace(history: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    evaluated = [entry for entry in history if entry.get("status") == "evaluated"]
    evaluated.sort(key=lambda item: item.get("chi2", float("inf")))
    top = evaluated[:n]
    return [
        {
            "iteration": item.get("iteration"),
            "chi2": item.get("chi2"),
            "params": item.get("params"),
            "breakdown": item.get("breakdown"),
        }
        for item in top
    ]


def run_joint_capture(
    datasets: Optional[Sequence[str]] = None,
    output_dir: Path = Path("data/results"),
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the joint pipeline and persist the comprehensive artefact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)
    stage_timings: Dict[str, Dict[str, float]] = {}
    cpu_start = time.process_time()
    wall_pipeline_start = time.perf_counter()
    dataset_keys = list(datasets) if datasets is not None else list(DEFAULT_DATASET_ORDER)
    if not dataset_keys:
        dataset_keys = list(DEFAULT_DATASET_ORDER)
    datasets_context = {key: _load_dataset(key) for key in dataset_keys}
    env_meta = _gather_environment_metadata()

    model_payloads: Dict[str, Any] = {}
    history_payloads: Dict[str, Any] = {}

    for model_type in ("lcdm", "pbuf"):
        stage_label = f"{model_type}_fit"
        wall_start = time.perf_counter()
        cpu_stage_start = time.process_time()
        fit_result = fit_joint(
            model_type=model_type,
            datasets=list(dataset_keys),
            verbose=verbose,
            collect_history=True,
        )
        wall_end = time.perf_counter()
        cpu_stage_end = time.process_time()
        stage_timings[stage_label] = {
            "wall_seconds": wall_end - wall_start,
            "cpu_seconds": cpu_stage_end - cpu_stage_start,
        }
        best_params = fit_result["params"]
        model = _instantiate_model(model_type, best_params)
        chi2_total, breakdown = _chi2_breakdown(model, dataset_keys)
        dof_info = _compute_degrees_of_freedom(breakdown, datasets_context, len(best_params))
        sigma8_0 = best_params.get("sigma8_0", SIGMA8_PLANCK)
        predictives = {}
        if "pantheon" in datasets_context:
            predictives["pantheon"] = _pantheon_predictives(model, best_params, datasets_context["pantheon"])
        if "sh0es" in datasets_context:
            predictives["sh0es"] = _shoes_predictives(model, best_params, datasets_context["sh0es"])
        if "iso" in datasets_context:
            predictives["bao_iso"] = _bao_iso_predictives(model, datasets_context["iso"])
        if "aniso" in datasets_context:
            predictives["bao_aniso"] = _bao_aniso_predictives(model, datasets_context["aniso"])
        if "cmb" in datasets_context:
            predictives["cmb"] = _cmb_predictives(model, datasets_context["cmb"])
        if "cc" in datasets_context:
            predictives["cc"] = _cc_predictives(model, datasets_context["cc"])
        if "rsd" in datasets_context:
            predictives["rsd"] = _rsd_predictives(model, datasets_context["rsd"], sigma8_0=float(sigma8_0))
        z_sources: List[np.ndarray] = []
        for key in ("cc", "rsd", "iso", "aniso", "pantheon"):
            if key in datasets_context:
                values = datasets_context[key].data.get("z")
                if values is not None:
                    z_sources.append(np.asarray(values, dtype=float).ravel())
        z_union = np.concatenate(z_sources) if z_sources else np.array([0.0])
        z_grid = np.unique(np.concatenate([np.linspace(0, 3, 60), z_union]))
        geometry = _geometry_traces(model, z_grid)
        growth = _growth_traces(model, z_grid, sigma8_0=float(sigma8_0))

        r_d = sound_horizon(model)
        z_star = redshift_star(model)
        D_A_star = transverse_comoving_distance(z_star, model) / (1 + z_star)
        derived = {
            "r_d_Mpc": float(r_d),
            "D_A_zstar": float(D_A_star),
            "z_star": float(z_star),
            "E_grid": geometry["E_z"],
            "Omega_k": float(best_params.get("Ok0", 0.0)),
        }

        physics_flags = {"phase6a": None}
        elastic_diag = None

        if model_type == "pbuf":
            elastic_diag = _elastic_diagnostics(model, best_params, z_grid)
            physics_flags["phase6a"] = _phase6a_summary(model, best_params)
        else:
            physics_flags["phase6a"] = {"status": True, "note": "LCDM auto-pass"}

        sensitivity = _sensitivity_scan(model_type, best_params, dataset_keys, rel_step=0.01, chi2_base=chi2_total)

        model_payloads[model_type] = {
            "best_params": {k: float(v) for k, v in best_params.items()},
            "chi2_total": chi2_total,
            "breakdown": breakdown,
            "degrees_of_freedom": dof_info,
            "predictives": predictives,
            "geometry": geometry,
            "growth": growth,
            "derived": derived,
            "physics_flags": physics_flags,
            "elastic_diagnostics": elastic_diag,
            "sensitivity": sensitivity,
            "optimizer": {
                "status": fit_result.get("status"),
                "nfev": fit_result.get("nfev"),
                "nit": fit_result.get("nit"),
                "message": fit_result.get("optimizer_message"),
                "dataset_weights": fit_result.get("dataset_weights"),
            },
        }

        history_payloads[model_type] = {
            "counters": fit_result.get("evaluation_counters", {}),
            "top_evaluations": _top_n_trace(fit_result.get("evaluation_history", [])),
            "full_history": fit_result.get("evaluation_history", []),
        }

        model_payloads[model_type]["profiles"] = [
            _profile_likelihood(model_type, best_params, dataset_keys, "H0"),
            _profile_likelihood(model_type, best_params, dataset_keys, "Om0"),
        ]
        if model_type == "pbuf":
            model_payloads[model_type]["profiles"].append(
                _profile_likelihood(model_type, best_params, dataset_keys, "alpha", "k_sat")
            )
        else:
            model_payloads[model_type]["profiles"].append(
                _profile_likelihood(model_type, best_params, dataset_keys, "H0", "Om0")
            )

        model_payloads[model_type]["hessian"] = _approximate_hessian(model_type, best_params, dataset_keys)

    wall_total = time.perf_counter()
    cpu_total = time.process_time()

    comparison = _compute_model_comparison(model_payloads["lcdm"], model_payloads["pbuf"])
    fairness = _build_fairness_notes(datasets_context)
    performance = _compile_performance_metrics(model_payloads, comparison)

    lcdm_nfev = model_payloads["lcdm"]["optimizer"]["nfev"]
    pbuf_nfev = model_payloads["pbuf"]["optimizer"]["nfev"]
    compute_budget_parity = (
        lcdm_nfev is not None
        and pbuf_nfev is not None
        and int(lcdm_nfev) == int(pbuf_nfev)
    )

    run_meta = {
        "started_at": start_time.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": wall_total - wall_pipeline_start,
        "cpu_time_seconds": cpu_total - cpu_start,
        "stage_timings": stage_timings,
        "workers": os.cpu_count(),
        "environment": env_meta,
        "datasets": {k: ctx.manifest for k, ctx in datasets_context.items()},
        "compute_budget_parity": compute_budget_parity,
        "early_stop": {
            "lcdm": model_payloads["lcdm"]["optimizer"]["status"] != "success",
            "pbuf": model_payloads["pbuf"]["optimizer"]["status"] != "success",
        },
        "random_seed_head": int(np.random.get_state()[1][0]),
    }

    payload = {
        "run_meta": run_meta,
        "best_fit": model_payloads,
        "predictives": {k: v["predictives"] for k, v in model_payloads.items()},
        "diagnostics": {
            "profiles": {k: model_payloads[k]["profiles"] for k in model_payloads},
            "hessian": {k: model_payloads[k]["hessian"] for k in model_payloads},
            "physics_flags": {k: model_payloads[k]["physics_flags"] for k in model_payloads},
        },
        "optimizer_trace": history_payloads,
        "fairness_notes": fairness,
        "performance": performance,
        "comparison": comparison,
    }

    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = output_dir / f"joint_capture_{timestamp}.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload["artifact_path"] = str(output_path)
    payload["datasets_used"] = dataset_keys
    serializable_payload = _deep_make_serializable(payload)
    with output_path.open("w") as handle:
        json.dump(serializable_payload, handle, indent=2)

    if verbose:
        print(f"[joint-capture] saved {output_path}")

    return payload


def _compute_model_comparison(lcdm_payload: Dict[str, Any], pbuf_payload: Dict[str, Any]) -> Dict[str, Any]:
    chi2_lcdm = lcdm_payload["chi2_total"]
    chi2_pbuf = pbuf_payload["chi2_total"]
    k_lcdm = len(lcdm_payload["best_params"])
    k_pbuf = len(pbuf_payload["best_params"])
    n_data = lcdm_payload["degrees_of_freedom"]["n_data_total"]
    aic_lcdm = chi2_lcdm + 2 * k_lcdm
    aic_pbuf = chi2_pbuf + 2 * k_pbuf
    bic_lcdm = chi2_lcdm + k_lcdm * math.log(max(n_data, 1))
    bic_pbuf = chi2_pbuf + k_pbuf * math.log(max(n_data, 1))
    delta_aic = aic_pbuf - aic_lcdm
    delta_bic = bic_pbuf - bic_lcdm
    lr_stat = chi2_lcdm - chi2_pbuf
    p_value = stats.chi2.sf(max(lr_stat, 0.0), df=max(k_pbuf - k_lcdm, 1))
    return {
        "AIC": {"lcdm": aic_lcdm, "pbuf": aic_pbuf, "delta": delta_aic},
        "BIC": {"lcdm": bic_lcdm, "pbuf": bic_pbuf, "delta": delta_bic},
        "chi2": {"lcdm": chi2_lcdm, "pbuf": chi2_pbuf, "delta": chi2_pbuf - chi2_lcdm},
        "likelihood_ratio": {
            "statistic": lr_stat,
            "p_value": float(p_value),
            "df": max(k_pbuf - k_lcdm, 1),
        },
    }


def _build_fairness_notes(datasets_context: Dict[str, DatasetContext]) -> Dict[str, Any]:
    notes = {}
    for key, ctx in datasets_context.items():
        meta = ctx.manifest.get("meta", {})
        notes[key] = {
            "calibrated_to_lcdm": bool(meta.get("lcdm_calibrated", key in {"iso", "aniso", "rsd"})),
            "ap_corrections": meta.get("ap_scaling"),
            "reference": meta.get("reference"),
        }
    notes["priors"] = {
        "lcdm": "flat bounds (no explicit priors)",
        "pbuf": "flat bounds; Phase-6a enforced post-fit",
    }
    return notes


def _compile_performance_metrics(models: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    perf = {}
    for model_type, payload in models.items():
        chi2_total = payload["chi2_total"]
        dof = payload["degrees_of_freedom"]["dof"]
        reduced = chi2_total / max(dof, 1)
        ppp = stats.chi2.sf(chi2_total, df=max(dof, 1))
        perf[model_type] = {
            "reduced_chi2": reduced,
            "posterior_predictive_p": float(ppp),
            "chi2_total": chi2_total,
        }
    perf["comparison"] = comparison
    return perf


__all__ = ["run_joint_capture"]
