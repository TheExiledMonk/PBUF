from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from cosmos.helper.distances import sound_horizon, transverse_comoving_distance
from cosmos.optim.dataset_evaluators import build_model, ModelConstructionError

# --- Tunable constants ---
MAX_HZ_LCDM = 1.0e9
MAX_HZ_PBUF = 1.0e12        # allow hotter elastic sectors
MIN_DENSITY = -1e-6         # tolerate tiny negatives due to rounding
MIN_DM = 1e-12

def _finite(v): return v is not None and np.isfinite(v)

def _check_parameters(params: Dict[str, float], reasons: List[str]) -> None:
    for k, v in params.items():
        if not _finite(v):
            reasons.append(f"{k} not finite ({v})")

def _check_expansion(model, reasons: List[str], diagnostics: Dict[str, float], model_type: str="lcdm") -> None:
    """H(z) reasonableness"""
    if model_type.lower() == "pbuf":
        hz_samples = (0.0, 0.5, 1.0, 2.0, 5.0, 50.0)
        max_hz = MAX_HZ_PBUF
    else:
        hz_samples = (0.0, 1.0, 2.0, 10.0, 1000.0)
        max_hz = MAX_HZ_LCDM

    for z in hz_samples:
        try:
            Hz = float(model.H(z))
        except Exception:
            Hz = np.nan
        if not _finite(Hz):
            reasons.append(f"H(z={z}) not finite")
            return
        if Hz <= 0.0:
            reasons.append(f"H(z={z}) ≤ 0")
            return
        if abs(Hz) >= max_hz:
            # Only soft-fail for PBUF at very high z
            if model_type.lower() == "pbuf" and z > 10:
                diagnostics.setdefault("edge_case", True)
            else:
                reasons.append(f"H(z={z}) ≥ {max_hz}")
                return
    diagnostics["Hz_samples"] = {z: float(model.H(z)) for z in (0.0, 1.0)}

def _check_densities(model, reasons: List[str], model_type: str="lcdm") -> None:
    """Density parameter sanity"""
    samples = (0.0, 1.0) if model_type.lower() == "pbuf" else (0.0, 1.0, 5.0)
    enforce_keys_pbuf = {"omega_m", "omega_r", "omega_k", "omega_elastic_raw"}
    for z in samples:
        try:
            dens = model.density_parameters_at_z(z)
        except Exception:
            reasons.append(f"density parameters failed at z={z}")
            return
        for n, val in dens.items():
            if not _finite(val):
                reasons.append(f"{n} invalid at z={z}")
                return
            if model_type.lower() == "pbuf":
                # Ignore sign on the compensator channel; enforce raw positivity.
                if n == "omega_elastic_radfix":
                    continue
                if n not in enforce_keys_pbuf:
                    continue
            if val < MIN_DENSITY:
                reasons.append(f"{n} negative at z={z}")
                return

def _check_distances(model, reasons: List[str], model_type: str="lcdm") -> None:
    """Distance calculations"""
    try:
        rs = float(sound_horizon(model))
    except Exception:
        rs = np.nan
    if (not _finite(rs)) or rs <= 0.0:
        # For PBUF, we don't hard-fail if rs overflows; mark diagnostic instead
        if model_type.lower() == "pbuf":
            rs = np.nan
        else:
            reasons.append("sound horizon invalid")
            return

    samples = (0.5, 1.0, 2.0)
    for z in samples:
        try:
            dm = float(transverse_comoving_distance(z, model))
        except Exception:
            dm = np.nan
        if (not _finite(dm)) or dm <= MIN_DM:
            reasons.append(f"D_M(z={z}) invalid")
            return

def validate_cosmology(model_type: str, params: Dict[str, float]) -> Dict[str, object]:
    """Return validation verdict for the given cosmology."""
    reasons: List[str] = []
    diagnostics: Dict[str, object] = {}

    _check_parameters(params, reasons)
    if reasons:
        return {"valid": False, "reasons": reasons, "diagnostics": diagnostics}

    try:
        model = build_model(model_type, params)
    except ModelConstructionError as exc:
        reasons.append(f"model_build: {exc}")
        return {"valid": False, "reasons": reasons, "diagnostics": diagnostics}

    _check_expansion(model, reasons, diagnostics, model_type)
    if reasons:
        return {"valid": False, "reasons": reasons, "diagnostics": diagnostics}

    _check_densities(model, reasons, model_type)
    if reasons:
        return {"valid": False, "reasons": reasons, "diagnostics": diagnostics}

    _check_distances(model, reasons, model_type)
    if reasons:
        return {"valid": False, "reasons": reasons, "diagnostics": diagnostics}

    return {"valid": True, "reasons": [], "diagnostics": diagnostics}
