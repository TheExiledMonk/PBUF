#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAO background-distance predictions for PBUF and ΛCDM models.

Provides isotropic (D_V / r_s) and anisotropic (D_M / r_s, H(z)*r_s)
ratios for BAO datasets, using the same geometric core as CMB priors.
"""

from __future__ import annotations
from typing import Dict, Mapping, List
import numpy as np

from config.constants import C_LIGHT, KM_TO_M
from core import cmb_priors, gr_models, pbuf_models

KM_PER_S = C_LIGHT / KM_TO_M


# ---------------------------------------------------------------------
#  Utility geometry
# ---------------------------------------------------------------------
def _select_model(model):
    """Return the chosen model or default to PBUF."""
    return model if model is not None else pbuf_models


def _D_H(z: float, params: Mapping[str, float], model=None) -> float:
    """Hubble distance D_H(z) = c / H(z) [Mpc]."""
    work = cmb_priors.prepare_background_params(params)
    model = _select_model(model)
    return KM_PER_S / model.H(z, work)


def _D_M(z: float, params: Mapping[str, float], model=None) -> float:
    """Transverse comoving distance D_M(z) in Mpc."""
    return float((1 + z) * cmb_priors.D_A(z, params, model=model))


def _D_V(z: float, params: Mapping[str, float], model=None) -> float:
    """Volume-averaged distance D_V(z) = [ (1+z)^2 D_A^2(z) c z / H(z) ]^(1/3)."""
    work = cmb_priors.prepare_background_params(params)
    model = _select_model(model)
    da = cmb_priors.D_A(z, work, model=model)
    hz = model.H(z, work)
    dv = ((1 + z) ** 2 * da**2 * KM_PER_S * z / hz) ** (1.0 / 3.0)
    return float(dv)


# ---------------------------------------------------------------------
#  Isotropic BAO: D_V / r_s
# ---------------------------------------------------------------------
def bao_distance_ratios(
    params: Mapping[str, float],
    model=None,
    priors: Mapping[str, object] | None = None,
) -> Dict[str, float]:
    """
    Compute BAO isotropic predictions D_V(z)/r_s(z_d).
    If priors are provided, extract redshifts from label names.
    """
    work = cmb_priors.prepare_background_params(params)
    model = _select_model(model)

    # --- Dynamically extract z-values ---
    if priors is not None:
        z_points = []
        for lbl in priors.get("labels", []):
            if lbl.startswith("Dv_rs_"):
                try:
                    z_points.append(float(lbl.split("_")[-1]))
                except ValueError:
                    continue
        if not z_points:
            z_points = [0.106, 0.15, 0.38, 0.51, 0.61]
    else:
        z_points = [0.106, 0.15, 0.38, 0.51, 0.61]

    z_points = np.asarray(sorted(z_points), dtype=float)

    # --- Compute baryon-drag redshift and sound horizon ---
    z_d = cmb_priors.z_drag(work["Obh2"], work["Omh2"])
    r_s = cmb_priors.sound_horizon_a(z_d, work, model=model)

    # --- Evaluate D_V/r_s for each z ---
    results = {}
    for z in z_points:
        key = f"Dv_rs_{z:.3f}".rstrip("0").rstrip(".")
        results[key] = _D_V(z, work, model) / r_s

    results.update({"r_s": r_s, "z_d": z_d})
    return results


# ---------------------------------------------------------------------
#  Anisotropic BAO: D_M / r_s and H(z)*r_s
# ---------------------------------------------------------------------
def bao_anisotropic_ratios(
    params: Mapping[str, float], model=None, priors: Mapping[str, object] | None = None
) -> Dict[str, float]:
    """
    Compute anisotropic BAO ratios for each redshift:
        D_M(z)/r_s   and   H(z)*r_s
    """
    work = cmb_priors.prepare_background_params(params)
    model = _select_model(model)

    # Try to infer redshifts dynamically
    z_points = []
    if priors is not None:
        for lbl in priors.get("labels", []):
            if lbl.startswith(("DM_rs_", "Hrs_")):
                try:
                    z_points.append(float(lbl.split("_")[-1]))
                except ValueError:
                    continue
    if not z_points:
        z_points = [0.38, 0.51, 0.61]

    z_points = np.asarray(sorted(set(z_points)), dtype=float)

    # --- Baryon drag epoch and sound horizon ---
    z_d = float(cmb_priors.z_drag(work["Obh2"], work["Omh2"]))
    r_s = float(cmb_priors.sound_horizon_a(z_d, work, model=model))

    results: Dict[str, float] = {}
    for z in z_points:
        d_m = float(_D_M(z, work, model))
        h = float(model.H(z, work))
        results[f"DM_rs_{z:.3f}".rstrip("0").rstrip(".")] = d_m / r_s
        results[f"Hrs_{z:.3f}".rstrip("0").rstrip(".")] = h * r_s / KM_PER_S  # dimensionless combo

    results["r_s"] = r_s
    results["z_d"] = z_d
    return results


# ---------------------------------------------------------------------
#  Unified BAO predictor
# ---------------------------------------------------------------------
def bao_all_ratios(
    params: Mapping[str, float],
    model=None,
    priors: Mapping[str, object] | None = None,
) -> Dict[str, float]:
    """
    Unified BAO prediction that combines isotropic and anisotropic terms.
    Chooses which to compute automatically based on label names.
    """
    work_iso = bao_distance_ratios(params, model=model, priors=priors)
    work_ani = bao_anisotropic_ratios(params, model=model, priors=priors)

    results = {}

    if priors is None:
        # default: return both sets
        results.update(work_iso)
        results.update(work_ani)
        return results

    for lbl in priors.get("labels", []):
        if lbl.startswith("Dv_rs_"):
            results[lbl] = work_iso.get(lbl)
        elif lbl.startswith("DM_rs_"):
            results[lbl] = work_ani.get(lbl)
        elif lbl.startswith("Hrs_"):
            results[lbl] = work_ani.get(lbl)

    # Always include base quantities
    results["r_s"] = work_iso.get("r_s", work_ani.get("r_s"))
    results["z_d"] = work_iso.get("z_d", work_ani.get("z_d"))
    return results
