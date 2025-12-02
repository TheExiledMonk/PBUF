"""Parameter normalization and closure handling for the PBUF model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Tuple

from .distances import omega_total_at_a
from .elastic import omega_sigma_of_a, omega_sigma_raw_of_a
from .params import PBUFParams
from .thermal_table import ThermalTable

BARYON_FRACTION = 0.135


def _derive_density_from_alpha(alpha: float) -> tuple[float, float]:
    baryons = 2.0 * alpha
    if BARYON_FRACTION <= 0.0:
        raise ValueError("The baryon fraction must be positive to derive Omega_m0.")
    return baryons, baryons / BARYON_FRACTION


def resolve_alpha(params: PBUFParams, table: ThermalTable, metadata: Mapping[str, Any] | None = None) -> float:
    """
    Resolve the curvature parameter α from Quantum metadata or the thermal table, falling back to input params.
    """

    sources: tuple[Mapping[str, Any], ...] = (
        metadata or {},
        getattr(table, "metadata", {}),
    )
    for source in sources:
        for key in ("alpha_qm", "alpha"):
            if key in source:
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    continue

    try:
        return float(params.alpha)
    except Exception:
        return 0.0


def apply_omega_normalization(
    params: PBUFParams, table: ThermalTable, alpha: float
) -> Tuple[PBUFParams, Dict[str, Any]]:
    """
    Enforce the requested Ω normalization mode, returning updated params and diagnostics.
    """

    mode = params.omega_normalization
    if mode == "free":
        return params, {"mode": "free", "sigma_rescale": 1.0}

    if mode == "flat_today":
        sigma_target = 1.0 - params.Omega_m0 - params.Omega_r0 - alpha
        if sigma_target <= 0.0:
            raise ValueError("Cannot enforce flat_today normalization because Ω_sigma target ≤ 0.")

        omega_raw = omega_sigma_raw_of_a(1.0, params, table)
        if omega_raw <= 0.0:
            raise ValueError("Cannot normalize Ω_sigma because the raw Ω_sigma(a=1) ≤ 0.")

        rescale = sigma_target / omega_raw
        resolved = replace(params, sigma_rescale=rescale)
        omega_total = omega_total_at_a(1.0, resolved, table, alpha=alpha)
        omega_sigma = omega_sigma_of_a(1.0, resolved, table)
        metadata = {
            "mode": "flat_today",
            "sigma_rescale": rescale,
            "omega_total_a1": omega_total,
            "omega_sigma_a1": omega_sigma,
            "omega_sigma_target": sigma_target,
            "omega_sigma_raw_a1": omega_raw,
        }
        return resolved, metadata

    raise RuntimeError(f"Unsupported omega_normalization mode '{mode}'.")


def normalize_parameters(
    raw_params: PBUFParams, table: ThermalTable, metadata: Mapping[str, Any] | None = None
) -> Tuple[PBUFParams, Dict[str, Any], float]:
    """
    Resolve α from metadata and apply Ω normalization, returning finalized params and diagnostics.
    """

    alpha = resolve_alpha(raw_params, table, metadata)
    derived_ob, derived_om0 = _derive_density_from_alpha(alpha)
    derived_params = replace(
        raw_params,
        alpha=alpha,
        Omega_b0=derived_ob,
        Omega_m0=derived_om0,
    )
    normalized, norm_meta = apply_omega_normalization(derived_params, table, alpha)
    finalized = replace(
        normalized,
        alpha=alpha,
        Omega_b0=derived_ob,
        Omega_m0=derived_om0,
    )
    return finalized, norm_meta, alpha


__all__ = ["resolve_alpha", "apply_omega_normalization", "normalize_parameters"]
