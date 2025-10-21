# pipelines/fit_bao_ani.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAO anisotropic fitting pipeline (D_M / r_s and H(z)*r_s).

Tests expansion geometry and rigidity across redshift shells.
"""

from __future__ import annotations
import argparse, datetime as dt, math, subprocess
from pathlib import Path
from typing import Dict, Mapping, Sequence
import numpy as np
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import cmb_priors, gr_models, pbuf_models
from dataio.loaders import load_bao_ani_priors
from utils import logging as log
from utils.io import write_json_atomic, read_latest_result
from utils import parameters as param_utils

# ---------------------------------------------------------------------
#  Model registry
# ---------------------------------------------------------------------
MODEL_MAP = {"lcdm": gr_models, "pbuf": pbuf_models}

# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def _timestamp(): 
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")

def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "n/a"

def _metrics(chi2_val, dof, n_params, n_points):
    chi2_dof = chi2_val / dof if dof > 0 else math.nan
    aic = chi2_val + 2 * n_params
    bic = chi2_val + n_params * math.log(n_points)
    p_value = float(chi2_dist.sf(chi2_val, dof)) if dof > 0 else None
    return {
        "chi2": chi2_val, "dof": dof, "chi2_dof": chi2_dof,
        "AIC": aic, "BIC": bic, "p_value": p_value
    }

def _bao_ani_predictions(params: Mapping[str, float], model):
    """Compute D_M/r_s and H(z)*r_s predictions for anisotropic BAO."""
    from core.bao_background import bao_anisotropic_ratios
    return bao_anisotropic_ratios(params, model=model)

def _evaluate_bao_ani(params, priors, model):
    """
    Compute anisotropic BAO χ² using DM_rs_z and Hrs_z labels.
    """
    preds = _bao_ani_predictions(params, model)
    labels = priors["labels"]
    y = np.array([priors["means"][k] for k in labels], dtype=float)
    y_model = np.array([preds.get(k, np.nan) for k in labels], dtype=float)
    cov = np.asarray(priors["cov"], dtype=float)
    diff = y_model - y
    inv_cov = np.linalg.pinv(cov)
    chi2_val = float(diff.T @ inv_cov @ diff)
    return preds, chi2_val


def _to_jsonable(obj):
    """Recursively convert NumPy objects to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BAO anisotropic fitting pipeline.")
    parser.add_argument("--model", choices=["lcdm", "pbuf"], default="pbuf")
    parser.add_argument("--priors", default="bao_ani")
    parser.add_argument("--recomb", choices=["HS96", "PLANCK18", "EH98"], default="PLANCK18")
    parser.add_argument("--out", default="proofs/results")
    args = parser.parse_args()

    log.info("Loading BAO anisotropic priors '%s'", args.priors)
    priors = load_bao_ani_priors(args.priors)
    model_module = MODEL_MAP[args.model]
    model_name = args.model.upper()

    # ------------------------------------------------------------
    #  Start from the canonical global calibration
    # ------------------------------------------------------------
    canonical_block = param_utils.canonical_parameters(model_name)
    params: Dict[str, float | str] = dict(canonical_block)
    # Allow experiments with alternative recombination approximations.
    params["recomb_method"] = args.recomb
    cmb_fit_meta = read_latest_result(model=model_name, kind="CMB")
    cmb_source_path = cmb_fit_meta.get("_source_path") if cmb_fit_meta else None

    # ------------------------------------------------------------
    #  Evaluate anisotropic BAO
    # ------------------------------------------------------------
    pred, chi2_val = _evaluate_bao_ani(params, priors, model_module)

    labels = priors["labels"]
    n_params = 1 if args.model == "pbuf" else 0
    dof = max(len(labels) - n_params, 1)
    metrics = _metrics(chi2_val, dof, n_params, len(labels))

    out_dir = Path(args.out).resolve() / f"BAO_ANI_{args.model.upper()}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "timestamp": _timestamp(),
        "model": args.model,
        "priors": _to_jsonable(priors),
        "parameters": param_utils.build_parameter_payload(
            model_name,
            fitted=params,
            canonical=canonical_block,
        ),
        "predictions": _to_jsonable(pred),
        "metrics": metrics,
        "provenance": {
            "commit": _git_commit(),
            "recomb": args.recomb,
            "cmb_calibration": cmb_source_path,
        },
    }

    write_json_atomic(out_dir / "fit_results.json", _to_jsonable(result))
    log.info("Best-fit χ² = %.3f for %s anisotropic BAO", chi2_val, args.model.upper())


if __name__ == "__main__":
    main()
