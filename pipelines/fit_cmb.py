#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMB-only fitting pipeline using compressed distance priors.

This script calibrates the PBUF saturation stiffness ``k_sat`` (or
validates ΛCDM predictions) by matching Planck-style acoustic scale
constraints without running a perturbation solver.
"""

from __future__ import annotations
import argparse, datetime as dt, math, subprocess, sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config.constants import NEFF, TCMB
from core import cmb_priors, gr_models, pbuf_models
from dataio.loaders import load_cmb_priors
from utils import logging as log
from utils.io import read_yaml, write_json_atomic
from utils import parameters as param_utils


# ---------------------------------------------------------------------
#  Defaults
# ---------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Dict[str, float]] = {
    "lcdm": {"H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649, "recomb_method": "PLANCK18"},
    "pbuf": {
        "H0": 67.4,
        "Om0": 0.315,
        "Obh2": 0.02237,
        "alpha": 5.0e-4,
        "Rmax": 1.0e9,
        "eps0": 0.7,
        "n_eps": 0.0,
        "k_sat": 1.0,
        "ns": 0.9649,
        "recomb_method": "PLANCK18",
    },
}

MODEL_MAP = {"lcdm": gr_models, "pbuf": pbuf_models}


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _run_id(tag: str) -> str:
    return f"{tag}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "n/a"


def _parse_grid(spec: str | None, bounds: Tuple[float, float]) -> np.ndarray:
    """Parse a grid spec 'N,low,high' or 'auto' into a numeric grid."""
    lower, upper = bounds
    if spec is None or spec.strip().lower() == "auto":
        return np.linspace(lower, upper, 64)
    toks = [t.strip() for t in spec.split(",") if t.strip()]
    if len(toks) == 1:
        n = max(int(float(toks[0])), 3)
        return np.linspace(lower, upper, n)
    if len(toks) >= 3:
        n = max(int(float(toks[0])), 3)
        lo, hi = float(toks[1]), float(toks[2])
        if lo >= hi:
            raise ValueError("Lower k_sat bound must be smaller than upper bound")
        return np.linspace(lo, hi, n)
    raise ValueError(f"Bad grid spec '{spec}'")


def _seed_params(model_name: str) -> Dict[str, float]:
    params = dict(DEFAULT_PARAMS[model_name])
    params.setdefault("Tcmb", TCMB)
    params.setdefault("Neff", NEFF)
    params.setdefault("recomb_method", "PLANCK18")
    params.setdefault("Ok0", 0.0)
    params.setdefault("ns", 0.9649)
    if model_name == "lcdm":
        params.pop("k_sat", None)
    return params


def _fit_lcdm_parameters(
    base_params: Dict[str, float],
    priors: Mapping[str, object],
    model_module,
    bounds_cfg: Mapping[str, Sequence[float]],
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, object]]:
    """Optimise {H0, Om0, Obh2} for ΛCDM distance priors."""

    names = ("H0", "Om0", "Obh2")
    x0 = np.array([float(base_params[name]) for name in names], dtype=float)

    def _with_vector(vec: np.ndarray) -> Dict[str, float]:
        trial = dict(base_params)
        for name, value in zip(names, vec):
            trial[name] = float(value)
        return trial

    def objective(vec: np.ndarray) -> float:
        trial = _with_vector(vec)
        _, chi2_val = _evaluate_distance_priors(trial, priors, model_module)
        return chi2_val

    bounds = []
    for name in names:
        lower, upper = bounds_cfg.get(name, (None, None))
        lower = float(lower) if lower is not None else -np.inf
        upper = float(upper) if upper is not None else np.inf
        bounds.append((lower, upper))

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-9, "gtol": 1e-8, "maxiter": 1000},
    )

    best_vec = result.x if result.success else x0
    best_params = _with_vector(best_vec)
    predictions, chi2_val = _evaluate_distance_priors(best_params, priors, model_module)

    optimisation = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "n_iter": getattr(result, "nit", None),
        "n_function": getattr(result, "nfev", None),
    }
    return best_params, predictions, chi2_val, optimisation


# ---------------------------------------------------------------------
#  Core evaluation
# ---------------------------------------------------------------------
def _evaluate_distance_priors(
    params: Mapping[str, float],
    priors: Mapping[str, object],
    model_module,
) -> Tuple[Dict[str, float], float]:
    """Compute distance priors and χ² for given parameters."""
    preds = cmb_priors.distance_priors(params, model=model_module)
    chi2_val = cmb_priors.chi2_cmb(preds, priors)
    return preds, chi2_val


def _scan_k_sat(
    base_params: Dict[str, float],
    grid: np.ndarray,
    priors: Mapping[str, object],
    model_module,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Grid scan for PBUF k_sat calibration."""
    chi2_vals: list[float] = []
    preds: list[dict[str, float]] = []
    for val in grid:
        p = dict(base_params)
        p["k_sat"] = float(val)
        pred, c2 = _evaluate_distance_priors(p, priors, model_module)
        chi2_vals.append(c2)
        preds.append(pred)
    return grid, np.array(chi2_vals), preds


def _plot_chi2(grid: np.ndarray, chi2: np.ndarray, best_idx: int, out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(grid, chi2, label=r"$\chi^2(k_{\mathrm{sat}})$", color="#1f77b4")
    ax.scatter(grid[best_idx], chi2[best_idx], color="#d62728", zorder=5, label="Best fit")
    ax.set_xlabel(r"$k_{\mathrm{sat}}$")
    ax.set_ylabel(r"$\chi^2$")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path.resolve())


def _metrics(chi2_val: float, dof: int, n_params: int, n_points: int) -> Dict[str, float | int | None]:
    """Compute fit metrics."""
    chi2_dof = chi2_val / dof if dof > 0 else math.nan
    aic = chi2_val + 2.0 * n_params
    bic = chi2_val + n_params * math.log(n_points)
    try:
        p_value = float(chi2_dist.sf(chi2_val, dof)) if dof > 0 else None
    except Exception:
        p_value = None
    return {"chi2": chi2_val, "dof": dof, "chi2_dof": chi2_dof, "AIC": aic, "BIC": bic, "p_value": p_value}


def _residuals_sigma(pred: Mapping[str, float], priors: Mapping[str, object]) -> Dict[str, float]:
    """Return per-parameter residuals in σ units."""
    means, labels = priors["means"], priors["labels"]
    cov = np.asarray(priors["cov"], dtype=float)
    sigmas = np.sqrt(np.diag(cov))
    return {
        label: float((pred[label] - means[label]) / sigmas[i]) if sigmas[i] > 0 else math.nan
        for i, label in enumerate(labels)
    }


def _dataset_meta(priors: Mapping[str, object], name: str) -> Dict[str, object]:
    meta = dict(priors.get("meta", {}))
    meta.setdefault("name", name)
    if "resolved_path" in meta:
        meta["resolved_path"] = str(Path(str(meta["resolved_path"])).resolve())
    return meta


# ---------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CMB distance-prior calibration pipeline.")
    parser.add_argument("--model", choices=["lcdm", "pbuf"], default="pbuf")
    parser.add_argument("--priors", default="planck2018")
    parser.add_argument("--out", default="proofs/results")
    parser.add_argument("--grid-ksat", dest="grid_ksat", default=None)
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument(
    "--recomb",
    choices=["HS96", "EH98", "PLANCK18"],
    default="PLANCK18",
    help="Recombination redshift approximation to use (default: PLANCK18).",
)
    args = parser.parse_args()

    log.info("Loading configuration and priors '%s'", args.priors)
    settings = read_yaml("config/settings.yml")
    bounds = settings.get("bounds", {})
    k_bounds = bounds.get("k_sat", [0.5, 8.0])
    grid = _parse_grid(args.grid_ksat, (float(k_bounds[0]), float(k_bounds[1])))

    priors = load_cmb_priors(args.priors)
    priors_meta = _dataset_meta(priors, args.priors)

    model_module = MODEL_MAP[args.model]
    params = _seed_params(args.model)
    params["recomb_method"] = args.recomb
    
    # Optional: if you also want a toggle to turn off micro-calibration in cmb_priors later:
    # parser.add_argument("--calibrate-zstar", action="store_true")
    # params["calibrate_zstar"] = args.calibrate_zstar


    # --- Evaluate fits ---
    optimisation_meta: Dict[str, object] | None = None
    if args.model == "pbuf":
        log.info("Evaluating %d grid points for k_sat", len(grid))
        grid, chi2_vals, preds = _scan_k_sat(params, grid, priors, model_module)
        best_idx = int(np.argmin(chi2_vals))
        params["k_sat"] = float(grid[best_idx])
        pred = preds[best_idx]
        chi2_val = float(chi2_vals[best_idx])
    else:
        params, pred, chi2_val, optimisation_meta = _fit_lcdm_parameters(
            params, priors, model_module, bounds
        )
        grid = np.array([])
        chi2_vals = np.array([chi2_val])
        best_idx = 0

    # --- Metrics ---
    labels: Sequence[str] = priors["labels"]
    n_params = 1 if args.model == "pbuf" else 3
    dof = max(len(labels) - n_params, 1)
    metrics = _metrics(chi2_val, dof, n_params, len(labels))
    residuals = _residuals_sigma(pred, priors)

    dataset_tag = priors_meta.get("release_tag", args.priors).upper()
    mode_tag = "MOCK" if settings.get("mock") else "REAL"
    run_tag = f"CMB_{dataset_tag}_{mode_tag}"
    run_id = _run_id(run_tag)
    model_name = args.model.upper()

    # --- Output ---
    run_root = Path(args.out).resolve()
    run_dir = run_root / f"CMB_{model_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    figures: Dict[str, str] = {}
    if args.model == "pbuf":
        fig_path = run_dir / "chi2_vs_k_sat.png"
        figures["chi2_vs_k_sat"] = _plot_chi2(grid, chi2_vals, best_idx, fig_path)
        grid_payload = {"k_sat": [float(x) for x in grid], "chi2": [float(x) for x in chi2_vals]}
    else:
        grid_payload = {}

    canonical_block = param_utils.canonical_parameters(model_name)
    for name in list(canonical_block.keys()):
        if name in params:
            canonical_block[name] = params[name]
    parameter_payload = param_utils.build_parameter_payload(
        model_name,
        fitted=params,
        canonical=canonical_block,
    )

    result = {
        "run_id": run_id,
        "timestamp": _timestamp(),
        "model": model_name,
        "mock": settings.get("mock", False),
        "parameters": parameter_payload,
        "metrics": metrics,
        "dataset": priors_meta,
        "priors": {"labels": list(labels), "means": priors["means"], "cov": np.asarray(priors["cov"]).tolist()},
        "predictions": pred,
        "residuals_sigma": residuals,
        "grid_scan": grid_payload,
        "figures": figures,
        "provenance": {
            "commit": _git_commit(),
            "constants": str(Path("config/constants.py").resolve()),
            "settings": str(Path("config/settings.yml").resolve()),
            "priors_file": priors_meta.get("resolved_path"),
        },
    }

    if optimisation_meta is not None:
        result["optimisation"] = optimisation_meta

    json_path = run_dir / "fit_results.json"
    write_json_atomic(json_path, result)

    if args.model == "pbuf":
        log.info("Best-fit χ² = %.3f at k_sat=%.4f", chi2_val, params.get("k_sat", math.nan))
    else:
        log.info(
            "Best-fit ΛCDM: χ² = %.3f (H0=%.3f, Ωm=%.4f, Ωbh²=%.5f)",
            chi2_val,
            params.get("H0"),
            params.get("Om0"),
            params.get("Obh2"),
        )

    if args.generate_report:
        from pipelines.generate_report import render_report
        report_path = render_report(result, run_dir)
        log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
