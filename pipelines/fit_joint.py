#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint SN + BAO + CMB fitting pipeline for ΛCDM and PBUF models.

Combines supernova distance moduli (Pantheon+),
BAO isotropic and anisotropic distance ratios,
and Planck 2018 compressed CMB distance priors.

This yields a single calibration of H0, Ωm0, and related background params.
"""

from __future__ import annotations
import argparse, datetime as dt, math, subprocess, sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist

from config.constants import NEFF, TCMB
from core import cmb_priors, gr_models, pbuf_models
from core.bao_background import bao_distance_ratios, bao_anisotropic_ratios
from dataio.loaders import (
    load_dataset,
    load_bao_priors,
    load_bao_ani_priors,
    load_cmb_priors,
)
from utils import logging as log
from utils.io import read_yaml, write_json_atomic
from utils.plotting import plot_residuals, plot_pull_distribution

# ----------------------------------------------------------------------
# Parameter configuration
# ----------------------------------------------------------------------
PARAM_ORDER_LCDM: Sequence[str] = ("H0", "Om0", "Obh2")
PARAM_ORDER_PBUF: Sequence[str] = ("H0", "Om0", "Obh2", "alpha", "Rmax", "eps0", "n_eps", "k_sat")

DEFAULT_PARAMS_LCDM = {"H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649, "recomb_method": "PLANCK18"}
DEFAULT_PARAMS_PBUF = {
    "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237,
    "alpha": 5.0e-4, "Rmax": 1.0e9, "eps0": 0.7, "n_eps": 0.0, "k_sat": 1.0,
    "ns": 0.9649, "recomb_method": "PLANCK18",
}
MODEL_MAP = {"lcdm": gr_models, "pbuf": pbuf_models}

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _timestamp():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")

def _run_id(tag: str):
    return f"{tag}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"

def _vector_to_params(names: Sequence[str], values: Iterable[float]) -> Dict[str, float]:
    return {n: float(v) for n, v in zip(names, values)}

def _chi2_from_residuals(residuals: np.ndarray, sigma=None, cov=None) -> float:
    if cov is not None and np.size(cov) > 0:
        inv = np.linalg.pinv(cov)
        return float(residuals.T @ inv @ residuals)
    if sigma is not None and np.size(sigma) > 0:
        scaled = residuals / sigma
        return float(np.dot(scaled, scaled))
    return float(np.dot(residuals, residuals))

def _ensure_constants(params):
    params.setdefault("Tcmb", TCMB)
    params.setdefault("Neff", NEFF)
    params.setdefault("ns", 0.9649)
    return params

# ----------------------------------------------------------------------
# Individual χ² components
# ----------------------------------------------------------------------

def _sn_components(params, dataset, model_module):
    """Compute χ² for supernova distance moduli."""
    z = np.asarray(dataset["z"], dtype=float)
    mu_obs = np.asarray(dataset["y"], dtype=float)
    mu_model = model_module.mu(z, params)
    residuals = mu_obs - mu_model
    cov = np.asarray(dataset.get("cov")) if dataset.get("cov") is not None else None
    sigma = np.asarray(dataset.get("sigma")) if dataset.get("sigma") is not None else None
    chi2 = _chi2_from_residuals(residuals, sigma, cov)
    return chi2, mu_model, residuals


def _cmb_components(params, priors, model_module):
    """Compute χ² for CMB compressed distance priors."""
    preds = cmb_priors.distance_priors(params, model=model_module)
    chi2 = cmb_priors.chi2_cmb(preds, priors)
    return chi2, preds


def _bao_chi2(preds: Mapping[str, float], priors: Mapping[str, object]) -> float:
    labels = priors["labels"]
    y = np.array([priors["means"][k] for k in labels], dtype=float)
    y_model = np.array([preds.get(k, np.nan) for k in labels], dtype=float)
    cov = np.asarray(priors["cov"], dtype=float)
    diff = y_model - y
    inv_cov = np.linalg.pinv(cov)
    return float(diff.T @ inv_cov @ diff)


# --- NEW FUNCTION (dedicated for anisotropic BAO) ---
def _bao_aniso_chi2(preds: Mapping[str, float], priors: Mapping[str, object]) -> float:
    """
    Compute anisotropic BAO χ² using DM_rs_z and Hrs_z labels.
    These correspond to transverse and radial distances in units of r_s.
    """
    labels = priors["labels"]
    y = np.array([priors["means"][k] for k in labels], dtype=float)
    y_model = np.array([preds.get(k, np.nan) for k in labels], dtype=float)
    cov = np.asarray(priors["cov"], dtype=float)
    diff = y_model - y
    inv_cov = np.linalg.pinv(cov)
    return float(diff.T @ inv_cov @ diff)


def _bao_components(params: Mapping[str, float], priors: Mapping[str, object], model_module):
    preds = bao_distance_ratios(params, model=model_module, priors=priors)
    chi2_val = _bao_chi2(preds, priors)
    return float(chi2_val), {k: float(v) for k, v in preds.items()}


def _bao_aniso_components(params: Mapping[str, float], priors: Mapping[str, object], model_module):
    preds = bao_anisotropic_ratios(params, model=model_module, priors=priors)
    chi2_val = _bao_aniso_chi2(preds, priors)
    if not np.isfinite(chi2_val):
        import math
        log.warn("[debug] BAO_ANI χ² is NaN! preds=%s", preds)
        chi2_val = 0.0
    else:
        log.info("[debug] BAO_ANI χ²=%.6f  DM_rs_0.38=%.3f  Hrs_0.38=%.5f",
                 chi2_val, preds.get("DM_rs_0.38"), preds.get("Hrs_0.38"))
    return float(chi2_val), {k: float(v) for k, v in preds.items()}


# ----------------------------------------------------------------------
# Metrics and statistics
# ----------------------------------------------------------------------
def _metrics_block(chi2_value, dof, n_params, n_points):
    chi2_dof = chi2_value / dof if dof > 0 else math.nan
    aic = chi2_value + 2.0 * n_params
    bic = chi2_value + n_params * math.log(max(n_points, 1))
    p_value = float(chi2_dist.sf(chi2_value, dof)) if dof > 0 else None
    return dict(chi2=chi2_value, dof=dof, chi2_dof=chi2_dof, AIC=aic, BIC=bic, p_value=p_value)

# ----------------------------------------------------------------------
# Small verifiers (so you know we use the actual data)
# ----------------------------------------------------------------------
def _verify_labels(name: str, priors: Mapping[str, object], preds: Mapping[str, float]) -> None:
    labels = list(priors.get("labels", []))
    missing = [k for k in labels if k not in preds or not np.isfinite(preds[k])]
    log.info("[verify] %s: %d labels", name, len(labels))
    if missing:
        log.warn("[verify] %s: missing/non-finite %d/%d keys: %s", name, len(missing), len(labels), ", ".join(missing))

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Joint SN+BAO+CMB calibration pipeline.")
    parser.add_argument("--model", choices=["lcdm", "pbuf"], default="pbuf")
    parser.add_argument("--datasets", default="sn,bao,bao_ani,cmb", help="Comma-separated list of components to include")
    parser.add_argument("--sn-dataset", default="pantheon_plus")
    parser.add_argument("--bao-priors", default="bao_mixed")
    parser.add_argument("--bao-ani-priors", default="bao_ani")
    parser.add_argument("--cmb-priors", default="planck2018")
    parser.add_argument("--out", default="proofs/results")
    args = parser.parse_args()

    components = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
    log.info("Preparing joint fit with components: %s", ", ".join(sorted(components)))

    # ------------------------------------------------------------------
    # Load configuration and datasets
    # ------------------------------------------------------------------
    settings = read_yaml("config/settings.yml")
    datasets_cfg = read_yaml("config/datasets.yml")
    model_module = MODEL_MAP[args.model]

    sn_dataset = load_dataset(args.sn_dataset, datasets_cfg) if "sn" in components else None
    bao_prior = load_bao_priors(args.bao_priors) if "bao" in components else None
    bao_ani_prior = load_bao_ani_priors(args.bao_ani_priors) if "bao_ani" in components else None
    if "bao_ani" in components and bao_ani_prior is None:
        log.warn("[warn] BAO anisotropic requested but not loaded — BAO_ANI χ² will be 0!")
    cmb_prior = load_cmb_priors(args.cmb_priors) if "cmb" in components else None
    if "cmb" in components and cmb_prior is None:
        log.warn("[warn] CMB requested but not loaded — CMB χ² will be 0!")


    # Basic input verifiers
    if sn_dataset:
        log.info("[verify] SN points: N=%d (z: %s)", len(sn_dataset["z"]), sn_dataset.get("meta", {}).get("release_tag"))
    if bao_prior:
        log.info("[verify] BAO iso labels: %s", bao_prior.get("labels"))
    if bao_ani_prior:
        log.info("[verify] BAO aniso labels: %s", bao_ani_prior.get("labels"))
    if cmb_prior:
        log.info("[verify] CMB labels: %s", cmb_prior.get("labels"))

    param_order = PARAM_ORDER_PBUF if args.model == "pbuf" else PARAM_ORDER_LCDM
    defaults = DEFAULT_PARAMS_PBUF if args.model == "pbuf" else DEFAULT_PARAMS_LCDM
    x0 = np.array([defaults[n] for n in param_order], dtype=float)

    bounds_cfg = settings.get("bounds", {})
    bounds = [(float(bounds_cfg.get(n, (None, None))[0] or -np.inf),
               float(bounds_cfg.get(n, (None, None))[1] or np.inf)) for n in param_order]

    # ------------------------------------------------------------------
    # Objective: total χ²
    # ------------------------------------------------------------------
    def objective(theta):
        params = _ensure_constants(_vector_to_params(param_order, theta))
        total = 0.0
        if sn_dataset is not None:
            chi2_sn, _, _ = _sn_components(params, sn_dataset, model_module)
            total += chi2_sn
        if bao_prior is not None:
            chi2_bao, _ = _bao_components(params, bao_prior, model_module)
            total += chi2_bao
        if bao_ani_prior is not None:
            chi2_bao_ani, _ = _bao_aniso_components(params, bao_ani_prior, model_module)
            total += chi2_bao_ani
        if cmb_prior is not None:
            chi2_cmb, _ = _cmb_components(params, cmb_prior, model_module)
            total += chi2_cmb
        return float(total if np.isfinite(total) else 1e9)

    # ------------------------------------------------------------------
    # Minimize joint χ²
    # ------------------------------------------------------------------
    log.info("Running minimization for model %s ...", args.model.upper())
    optimisation = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    if not optimisation.success:
        log.warn("Optimization terminated: %s", optimisation.message)

    best_params = _ensure_constants(_vector_to_params(param_order, optimisation.x))

    # Evaluate each component separately + verify
    sn_chi2 = bao_chi2 = bao_ani_chi2 = cmb_chi2 = 0.0
    bao_preds = bao_ani_preds = cmb_preds = {}

    if sn_dataset:
        sn_chi2, sn_model_mu, sn_residuals = _sn_components(best_params, sn_dataset, model_module)
    if bao_prior:
        bao_chi2, bao_preds = _bao_components(best_params, bao_prior, model_module)
        _verify_labels("BAO isotropic", bao_prior, bao_preds)
    if bao_ani_prior:
        bao_ani_chi2, bao_ani_preds = _bao_aniso_components(best_params, bao_ani_prior, model_module)
        _verify_labels("BAO anisotropic", bao_ani_prior, bao_ani_preds)
        log.info("[debug] bao_ani χ²=%.6f  keys(preds)=%d  labels=%d",
        bao_ani_chi2, len(bao_ani_preds), len(bao_ani_prior.get('labels', [])))
        log.info("[debug] bao_ani_prior keys: %s", list(bao_ani_prior.keys()))


    if cmb_prior:
            cmb_chi2, cmb_preds = _cmb_components(best_params, cmb_prior, model_module)
            from core import cmb_priors as cmb_utils
            try:
                cmb_resid = cmb_utils.residuals_sigma(cmb_preds, cmb_prior)
            except Exception:
                cmb_resid = {}
    else:
        cmb_resid = {}

    total_chi2 = sn_chi2 + bao_chi2 + bao_ani_chi2 + cmb_chi2

    log.info("χ² totals: SN=%.3f  BAO=%.3f  BAO_ANI=%.3f  CMB=%.3f  TOTAL=%.3f",
             sn_chi2, bao_chi2, bao_ani_chi2, cmb_chi2, total_chi2)

    # ------------------------------------------------------------------
    # Proper DoF for metrics (points - params)
    # ------------------------------------------------------------------
    n_sn = len(sn_dataset["z"]) if sn_dataset else 0
    n_bao = len(bao_prior["labels"]) if bao_prior else 0
    n_bao_ani = len(bao_ani_prior["labels"]) if bao_ani_prior else 0
    n_cmb = len(cmb_prior["labels"]) if cmb_prior else 0
    n_params = len(param_order)

    n_total = n_sn + n_bao + n_bao_ani + n_cmb

    def _m(chi2, npts):  # safe per-block metrics
        dof = max(npts - n_params, 1)
        return _metrics_block(chi2, dof, n_params, npts)

    metrics_payload = {
        "total": _metrics_block(total_chi2, max(n_total - n_params, 1), n_params, n_total),
        "sn": _m(sn_chi2, n_sn),
        "bao": _m(bao_chi2, n_bao),
        "bao_ani": _m(bao_ani_chi2, n_bao_ani),
        "cmb": _m(cmb_chi2, n_cmb),
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    run_id = _run_id(f"JOINT_SN_BAO_CMB_{args.model.upper()}")
    run_dir = Path(args.out).resolve() / f"JOINT_{args.model.upper()}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "fit_results.json"

    result = {
        "run_id": run_id,
        "timestamp": _timestamp(),
        "model": args.model.upper(),
        "mock": settings.get("mock", False),
        "parameters": {k: float(v) for k, v in best_params.items()},
        "metrics": metrics_payload,
        "dataset": {
            "name": "Joint SN+BAO+CMB",
            "sn": sn_dataset.get("meta", {}) if sn_dataset else None,
            "bao": bao_prior.get("meta", {}) if bao_prior else None,
            "bao_ani": bao_ani_prior.get("meta", {}) if bao_ani_prior else None,
            "cmb": cmb_prior.get("meta", {}) if cmb_prior else None,
        },
        "predictions": {"bao": bao_preds, "bao_ani": bao_ani_preds, "cmb": cmb_preds},
        "provenance": {
            "commit": _git_commit(),
            "_source_path": str(json_path.resolve()),
            "code_version": "v0.1.0",
        },
    }

    # Minimal SN figure outputs if we had SN
    if sn_dataset:
        z = np.asarray(sn_dataset["z"], dtype=float)
        mu_obs = np.asarray(sn_dataset["y"], dtype=float)
        mu_model = model_module.mu(z, best_params)
        residuals = mu_obs - mu_model
        figures = {}
        figures["sn_residuals_vs_z"] = plot_residuals(z, residuals, f"{args.model.upper()}_JOINT_SN", run_dir)
        sigma = np.asarray(sn_dataset.get("sigma")) if sn_dataset.get("sigma") is not None else None
        if sigma is not None and sigma.size == residuals.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                pulls = np.divide(residuals, sigma, out=np.zeros_like(residuals), where=sigma != 0)
            figures["sn_pull_distribution"] = plot_pull_distribution(pulls, f"{args.model.upper()}_JOINT_SN", run_dir)
        result["figures"] = figures

    write_json_atomic(json_path, result)
    log.info("✅ Joint fit JSON saved: %s", json_path)


if __name__ == "__main__":
    main()
