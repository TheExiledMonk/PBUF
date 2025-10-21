# pipelines/fit_bao.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAO-only fitting pipeline using isotropic distance ratios.

This pipeline tests the mid-scale geometry (D_V / r_s) against Planck-calibrated sound horizon.
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
from utils import logging as log
from utils.io import write_json_atomic, read_latest_result
from dataio.loaders import load_bao_priors, load_bao_real_data
from utils import parameters as param_utils

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


# ---------------------------------------------------------------------
#  Core evaluation
# ---------------------------------------------------------------------
def _bao_predictions(params: Mapping[str, float], model, priors: Mapping[str, object] | None = None):
    """
    Compute theoretical BAO predictions matching the priors' labels.
    Handles isotropic (Dv/rs) and anisotropic (DM/rs, Hrs) cases automatically.
    """
    from core.bao_background import bao_all_ratios
    return bao_all_ratios(params, model=model, priors=priors)


def _evaluate_bao(params, priors, model):
    preds = _bao_predictions(params, model, priors)
    chi2_val = cmb_priors.chi2_cmb(preds, priors)
    return preds, chi2_val



# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BAO isotropic distance-prior calibration.")
    parser.add_argument("--model", choices=["lcdm", "pbuf"], default="pbuf")
    parser.add_argument("--priors", default="bao_mixed")
    parser.add_argument("--recomb", choices=["HS96", "PLANCK18", "EH98"], default="PLANCK18")
    parser.add_argument("--out", default="proofs/results")
    args = parser.parse_args()

    log.info("Loading BAO priors '%s'", args.priors)

    # --- NEW: dynamic loader ---
    if args.priors == "bao_real":
        priors = load_bao_real_data()
    else:
        priors = load_bao_priors(args.priors)

    model_module = MODEL_MAP[args.model]
    model_name = args.model.upper()

    # ------------------------------------------------------------
    #  Canonical parameters + optional recombination override
    # ------------------------------------------------------------
    canonical_block = param_utils.canonical_parameters(model_name)
    params: Dict[str, float | str] = dict(canonical_block)
    params["recomb_method"] = args.recomb
    cmb_fit = read_latest_result(model=model_name, kind="CMB")
    cmb_source_path = cmb_fit.get("_source_path") if cmb_fit else None

    # ------------------------------------------------------------
    #  Evaluate fit
    # ------------------------------------------------------------
    pred, chi2_val = _evaluate_bao(params, priors, model_module)

    labels = priors["labels"]
    n_params = 1 if args.model == "pbuf" else 0
    dof = max(len(labels) - n_params, 1)
    metrics = _metrics(chi2_val, dof, n_params, len(labels))

    out_dir = Path(args.out).resolve() / f"BAO_{args.model.upper()}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON-safe conversion ---
    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    result = {
        "timestamp": _timestamp(),
        "model": args.model,
        "priors": {k: _to_serializable(v) for k, v in priors.items()},
        "parameters": param_utils.build_parameter_payload(
            model_name,
            fitted=params,
            canonical=canonical_block,
        ),
        "predictions": {k: _to_serializable(v) for k, v in pred.items()},
        "metrics": metrics,
        "provenance": {
            "commit": _git_commit(),
            "recomb": args.recomb,
            "cmb_calibration": cmb_source_path,
        },
    }

    write_json_atomic(out_dir / "fit_results.json", result)
    log.info("Best-fit χ² = %.3f for %s BAO priors", chi2_val, args.model.upper())


if __name__ == "__main__":
    main()
