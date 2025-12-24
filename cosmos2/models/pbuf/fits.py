"""PBUF-specific fit evaluators ported from the legacy cosmos stack."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras
from cosmos2.fits.joint import build_joint_chi2_evaluator, resolve_joint_fits
from cosmos2.fits.weak_lensing_kids1000 import run_fit as run_wl_kids1000_fit
from cosmos2.fits.wl import run_fit as run_wl_s8_fit
from cosmos2.models.pbuf.utils import C_LIGHT


def run_cmb_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("cmb")
    observed = np.asarray(dataset["obs"] if "obs" in dataset else dataset.get("observed"), dtype=float)
    inv_cov = np.asarray(dataset.get("inv_cov"), dtype=float)
    meta = dataset.get("meta")
    if isinstance(meta, np.ndarray):
        try:
            meta = meta.item()
        except Exception:
            meta = None
    z_star = dataset.get("z_star")
    if z_star is None and isinstance(meta, dict):
        z_star = meta.get("z_star") or meta.get("z")
    if z_star is None and "z" in dataset:
        try:
            z_star = float(np.asarray(dataset["z"]).flatten()[0])
        except Exception:
            pass
    z_star = z_star if z_star is not None else 1090.0
    cmb_out = model.cmb(dataset)
    predicted = np.array([cmb_out.R, cmb_out.l_A, cmb_out.theta_star], dtype=float)
    residual = predicted - observed
    chi2 = float(residual.T @ inv_cov @ residual)
    extras = build_fit_extras(
        dataset=dataset,
        predictions=predicted,
        observed=observed,
        residuals=residual,
        additional={"cmb_output": cmb_out, "z_star": z_star},
    )
    return chi2, extras


def run_bao_iso_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("bao_iso")
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
        inv_cov = np.asarray(inv_cov, dtype=float)
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


def _canonicalize_bao_label(label: str) -> str:
    normalized = "".join(ch for ch in label.lower() if ch.isalnum())
    if "htimes" in normalized:
        return "H_times_rd"
    if "dh" in normalized:
        return "DH_over_rd"
    if "dm" in normalized:
        return "DM_over_rd"
    if "da" in normalized:
        return "DA_over_rd"
    return normalized


def _resolve_bao_observables(dataset: Dict[str, Any], z: np.ndarray) -> tuple[str, ...]:
    if "observables" in dataset and dataset["observables"] is not None:
        raw = dataset["observables"]
        arr = np.asarray(raw, dtype=object).reshape(-1)
        return tuple(_canonicalize_bao_label(str(entry)) for entry in arr)

    labels = dataset.get("labels")
    if labels is None:
        raise ValueError("BAO anisotropic dataset requires observables or labels.")
    labels_arr = np.asarray(labels, dtype=object).reshape(-1)
    if z.size > 0 and labels_arr.size % z.size == 0:
        per_bin = max(1, labels_arr.size // z.size)
        labels_arr = labels_arr[:per_bin]
    return tuple(_canonicalize_bao_label(str(entry)) for entry in labels_arr)


def _make_bao_model_vector(model: Any, observables: tuple[str, ...], z: np.ndarray, rd: float) -> np.ndarray:
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


def run_bao_aniso_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("bao_aniso")

    z = np.asarray(dataset["z"], dtype=float)
    observed = np.asarray(dataset["obs"], dtype=float)
    observables = _resolve_bao_observables(dataset, z)
    rd = float(model.sound_horizon())
    if rd <= 0.0:
        raise ValueError("Model returned a non-positive sound horizon")

    model_vector = _make_bao_model_vector(model, observables, z, rd)
    diff = model_vector - observed

    inv_cov = dataset.get("inv_cov")
    if inv_cov is not None:
        inv_cov = np.asarray(inv_cov, dtype=float)
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
        additional={"observables": observables, "bao_aniso_model": model_vector, "rd": rd},
    )
    return chi2, extras


def run_sn_pantheon_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("sn")
    z = np.asarray(dataset["z"], dtype=float)
    mu_obs = np.asarray(dataset["obs"], dtype=float)
    cov = dataset.get("cov")
    err = dataset.get("err")
    if cov is not None:
        cov_full = np.asarray(cov, dtype=float)
    elif err is not None:
        cov_full = np.diag(np.asarray(err, dtype=float) ** 2)
    else:
        raise ValueError("SN dataset lacks covariance/errors")
    inv_cov = np.linalg.inv(cov_full)

    mu_model = np.asarray(model.distance_modulus(z), dtype=float)
    residuals = mu_model - mu_obs
    chi2 = float(residuals.T @ inv_cov @ residuals)

    extras = build_fit_extras(dataset=dataset, predictions=mu_model, observed=mu_obs, residuals=residuals)
    return chi2, extras


def run_sh0es_prior(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("sh0es")
    H0_obs = float(dataset.get("H0") or dataset["obs"][0])
    sigma = float(dataset.get("sigma") or dataset.get("err", [0.0])[0])
    H0_model = float(model.Hubble(0.0))
    chi2 = ((H0_model - H0_obs) / sigma) ** 2
    extras = build_fit_extras(dataset=dataset, predictions=H0_model, observed=H0_obs, residuals=H0_model - H0_obs)
    return chi2, extras


def run_cc_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("cc")
    z = np.asarray(dataset["z"], dtype=float)
    obs = np.asarray(dataset["obs"], dtype=float)
    cov = dataset.get("cov")
    err = dataset.get("err")
    if cov is not None:
        cov_full = np.asarray(cov, dtype=float)
    elif err is not None:
        cov_full = np.diag(np.asarray(err, dtype=float) ** 2)
    else:
        raise ValueError("CC dataset lacks covariance/errors")
    inv_cov = np.linalg.inv(cov_full)

    preds = np.asarray(model.Hubble(z), dtype=float)
    residuals = preds - obs
    chi2 = float(residuals.T @ inv_cov @ residuals)
    extras = build_fit_extras(dataset=dataset, predictions=preds, observed=obs, residuals=residuals)
    return chi2, extras


def run_rsd_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("rsd")
    z = np.asarray(dataset["z"], dtype=float)
    observed = np.asarray(dataset["obs"], dtype=float)
    cov = dataset.get("cov")
    err = dataset.get("err")
    if cov is not None:
        cov_full = np.asarray(cov, dtype=float)
    elif err is not None:
        cov_full = np.diag(np.asarray(err, dtype=float) ** 2)
    else:
        raise ValueError("RSD dataset lacks covariance/errors")
    inv_cov = np.linalg.inv(cov_full)

    fs8_model = np.asarray(model.fs8(z), dtype=float)
    residuals = fs8_model - observed
    chi2 = float(residuals.T @ inv_cov @ residuals)
    extras = build_fit_extras(dataset=dataset, predictions=fs8_model, observed=observed, residuals=residuals)
    extras["fs8_model"] = fs8_model
    return chi2, extras


def _resolve_gamma(dataset: Dict[str, Any], length: int) -> np.ndarray:
    gamma = dataset.get("gamma")
    if gamma is None:
        return np.full(length, 0.5, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    if gamma_arr.size == 1 and length > 1:
        return np.full(length, float(gamma_arr.reshape(-1)[0]), dtype=float)
    gamma_arr = gamma_arr.reshape(-1)
    if gamma_arr.size != length:
        raise ValueError(f"WL S8 gamma length {gamma_arr.size} mismatches observations length {length}")
    return gamma_arr


def run_wl_s8_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("wl_s8")
    S8_obs = np.asarray(dataset["S8_obs"] if "S8_obs" in dataset else dataset.get("obs"), dtype=float).reshape(-1)
    S8_err = np.asarray(dataset["S8_err"] if "S8_err" in dataset else dataset.get("err"), dtype=float).reshape(-1)
    if S8_obs.size != S8_err.size:
        raise ValueError("WL S8 observation/error length mismatch.")
    gamma = _resolve_gamma(dataset, S8_obs.size)

    cov = dataset.get("cov")
    if cov is not None:
        cov_full = np.asarray(cov, dtype=float)
        inv_cov = np.linalg.inv(cov_full)
    else:
        if np.any(S8_err <= 0.0):
            raise ValueError("WL S8 errors must be positive when no covariance is provided.")
        inv_cov = np.diag(1.0 / (S8_err * S8_err))

    om = float(model.omega_m0())
    if om <= 0.0:
        raise ValueError("Model returned non-positive Ω_m0; cannot build S₈ prediction.")
    s8 = float(model.sigma8())

    S8_model = s8 * (om / 0.3) ** gamma
    diff = S8_model - S8_obs
    chi2 = float(diff.T @ inv_cov @ diff)

    extras = build_fit_extras(
        dataset=dataset,
        predictions=S8_model,
        observed=S8_obs,
        residuals=diff,
        additional={"gamma": gamma, "S8_err": S8_err, "S8_model": S8_model},
    )
    return chi2, extras


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


def _build_galaxy_pk_model_matrix(model: Any, z: np.ndarray, labels: list[str], fiducials: Dict[str, np.ndarray]) -> np.ndarray:
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
    model_matrix = _build_galaxy_pk_model_matrix(model, z, labels, fiducials)
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


def run_lensing_cross_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("lensing_cross")
    A_obs = np.asarray(dataset["A_obs"] if "A_obs" in dataset else dataset.get("obs"), dtype=float)
    A_err = np.asarray(dataset["A_err"] if "A_err" in dataset else dataset.get("err"), dtype=float)
    p_exponent = np.asarray(dataset.get("p_exponent"), dtype=float)
    q_exponent = np.asarray(dataset.get("q_exponent"), dtype=float)
    z_eff = np.asarray(dataset.get("z_eff"), dtype=float)
    S8_fid = np.asarray(dataset.get("S8_fid"), dtype=float)
    fs8_fid = np.asarray(dataset.get("fs8_fid"), dtype=float)
    gamma = np.asarray(dataset.get("gamma", 0.5), dtype=float)
    weights = np.asarray(dataset.get("weights", 1.0), dtype=float)

    n = len(A_obs)
    for arr_name, arr in (
        ("A_err", A_err),
        ("p_exponent", p_exponent),
        ("q_exponent", q_exponent),
        ("z_eff", z_eff),
        ("S8_fid", S8_fid),
        ("fs8_fid", fs8_fid),
        ("gamma", gamma),
        ("weights", weights),
    ):
        if arr is None:
            raise ValueError(f"Lensing cross dataset missing field {arr_name}")
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (n,):
            raise ValueError(f"Lensing cross field {arr_name} has shape {arr.shape}, expected {(n,)}")
    if np.any(weights <= 0.0):
        raise ValueError("Lensing cross weights must be positive.")

    S8_model = np.asarray([model.S8(g) for g in gamma], dtype=float)
    fs8_model = np.asarray(model.fs8(z_eff), dtype=float).reshape(n)
    A_model = (S8_model / S8_fid) ** p_exponent * (fs8_model / fs8_fid) ** q_exponent

    diff = A_model - A_obs
    scaled_err = A_err / weights
    cov = dataset.get("cov")
    if cov is not None:
        inv_cov = dataset.get("inv_cov") or np.linalg.inv(np.asarray(cov, dtype=float))
        inv_cov = np.asarray(inv_cov, dtype=float)
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        if np.any(scaled_err <= 0.0):
            raise ValueError("Lensing cross errors must be positive when no covariance is provided.")
        chi2 = float(np.sum((diff / scaled_err) ** 2))

    additional = {
        "A_model": A_model,
        "z_eff": z_eff,
        "gamma": gamma,
        "fs8_model": fs8_model,
        "S8_model": S8_model,
        "weights": weights,
        "scaled_err": scaled_err,
    }
    if "labels" in dataset:
        additional["labels"] = dataset["labels"]

    extras = build_fit_extras(
        dataset=dataset,
        predictions=A_model,
        observed=A_obs,
        residuals=diff,
        additional=additional,
    )
    return chi2, extras


PBUF_FIT_REGISTRY: Dict[str, Callable[[Any], Tuple[float, Dict[str, Any]]]] = {
    "cmb": run_cmb_fit,
    "sn": run_sn_pantheon_fit,
    "sn_pantheon": run_sn_pantheon_fit,
    "sn_pantheonplus": run_sn_pantheon_fit,
    "sn_pantheon_shoes": run_sn_pantheon_fit,
    "sh0es": run_sh0es_prior,
    "bao_iso": run_bao_iso_fit,
    "bao_aniso": run_bao_aniso_fit,
    "bao_iso_full": lambda model: run_bao_iso_fit(model, get_dataset("bao_iso_full")),
    "bao_aniso_full": lambda model: run_bao_aniso_fit(model, get_dataset("bao_aniso_full")),
    "cc": run_cc_fit,
    "rsd": run_rsd_fit,
    "wl_s8": run_wl_s8_fit,
    "wl_kids1000": run_wl_kids1000_fit,
    "galaxy_pk": run_galaxy_pk_fit,
    "lensing_cross": run_lensing_cross_fit,
    "lensing_x": run_lensing_cross_fit,
}


def get_pbuf_fit(name: str) -> Callable[[Any], Tuple[float, Dict[str, Any]]]:
    normalized = name.strip().lower()
    if normalized not in PBUF_FIT_REGISTRY:
        raise ValueError(f"Unknown PBUF fit '{name}'.")
    return PBUF_FIT_REGISTRY[normalized]


def resolve_pbuf_joint_fits(joint_config_path: str | Any) -> Tuple[list[str], Dict[str, float]]:
    return resolve_joint_fits(joint_config_path, registry=PBUF_FIT_REGISTRY)


def build_pbuf_joint_chi2(
    model_factory: Callable[[Dict[str, float]], Any],
    joint_config_path: str | Any,
    *,
    skip_valid: bool = False,
    registry: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Callable[[Dict[str, float]], float]:
    # Use provided registry or default
    evaluator_registry = registry if registry is not None else PBUF_FIT_REGISTRY
    return build_joint_chi2_evaluator(
        model_factory,
        joint_config_path,
        skip_valid=skip_valid,
        registry=evaluator_registry,
    )


__all__ = [
    "PBUF_FIT_REGISTRY",
    "build_pbuf_joint_chi2",
    "get_pbuf_fit",
    "resolve_pbuf_joint_fits",
    "run_bao_aniso_fit",
    "run_bao_iso_fit",
    "run_cc_fit",
    "run_cmb_fit",
    "run_galaxy_pk_fit",
    "run_lensing_cross_fit",
    "run_rsd_fit",
    "run_sh0es_prior",
    "run_sn_pantheon_fit",
    "run_wl_s8_fit",
]
