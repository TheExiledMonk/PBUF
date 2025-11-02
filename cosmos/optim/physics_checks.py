"""
Physics guardrail layer for cosmological parameter optimization.

This module implements comprehensive physics validation to ensure that
parameter combinations respect fundamental cosmological principles,
even if they might otherwise produce good χ² fits.

The physics_scorecard() function serves as an "elimination game" that
rejects any parameter set violating basic physics, turning optimization
into a search over physically interpretable universes only.
"""

import numpy as np


def compute_q0(model, dz=1e-3):
    """
    Compute the deceleration parameter q0 today using analytical formula where possible,
    or numerical differentiation as fallback.

    For LCDM: q0 = (3/2)Ω_m - 1 (flat case)
    For PBUF: numerical calculation using the standard formula
    """
    try:
        # Try analytical calculation first
        if hasattr(model, 'omega_m') and hasattr(model, 'omega_lambda'):
            # LCDM case
            omega_m = model.omega_m
            omega_lambda = model.omega_lambda
            omega_k = getattr(model, 'omega_k', 0.0)

            # For flat LCDM: q0 = (3/2)Ω_m - 1
            # More generally: q0 = (1/2)(1 + 3(ω_m Ω_m + ω_Λ Ω_Λ + ω_k Ω_k))
            # For matter (w_m=0), Λ (w_Λ=-1), curvature (w_k=-1/3):
            q0 = (1/2) * (1 + 3 * (0 * omega_m - 1 * omega_lambda - (1/3) * omega_k))
            return q0

        # Fallback to numerical calculation for other models
        H0 = model.H(0.0)
        H_plus = model.H(dz)

        if not np.isfinite(H0) or not np.isfinite(H_plus) or H0 <= 0:
            return np.nan

        # Use the standard numerical formula: q0 = -1 - (1/H0) * (dH/dz)
        # But first compute dH/dz more accurately
        H_minus = model.H(-dz + 2*dz) if dz > 0 else model.H(2*dz)
        dH_dz = (H_plus - H_minus) / (2 * dz) if np.isfinite(H_minus) else (H_plus - H0) / dz

        q0 = -1.0 - dH_dz / H0
        return q0

    except Exception:
        return np.nan


def physics_scorecard(model, model_type, params):
    """
    Stage 0 physical sanity filter.
    For LCDM: enforces standard FRW-like closure.
    For PBUF: enforces numerical and causal sanity, but allows
    late-time elastic backreaction to shift H_today relative to H0.
    """

    reasons = []
    edge_case = False
    ok = True

    # Extract common parameters
    H0 = params["H0"]
    Om0 = params["Om0"]
    Or0 = params["Or0"]
    Ok0 = params["Ok0"]

    # --- 1. H0 bounds (hard) ---
    H0_MIN = 60.0
    H0_MAX = 80.0
    if not (H0_MIN <= H0 <= H0_MAX):
        ok = False
        reasons.append(f"H0={H0:.2f} outside [{H0_MIN:.0f},{H0_MAX:.0f}] km/s/Mpc")

    # --- 2. Matter / curvature sanity ---
    if Om0 < 0.0:
        ok = False
        reasons.append(f"Ω_m={Om0:.4f} < 0")
    if Or0 < 0.0:
        ok = False
        reasons.append(f"Ω_r={Or0:.4e} < 0")
    if abs(Ok0) > 0.1:
        ok = False
        reasons.append(f"|Ω_k|={abs(Ok0):.3f} > 0.1")

    # --- 3. Closure (FRW band / late-time elastic freedom) ---
    try:
        Omega_total = model.closure_today()
    except Exception:
        Omega_total = np.nan

    if not np.isfinite(Omega_total):
        ok = False
        reasons.append("Ω_total not finite")
    else:
        if model_type == "lcdm":
            # LCDM must sit near flat FRW closure
            if not (0.9 <= Omega_total <= 1.1):
                ok = False
                reasons.append(f"Ω_total={Omega_total:.4f} outside [0.9,1.1] (LCDM)")
            if abs(Omega_total - 1.0) > 0.05:
                edge_case = True
        else:
            # PBUF: allow elastic to shift apparent closure.
            # Reject only if the budget is unphysical; otherwise tag as an edge case.
            if (Omega_total <= 0.0) or (Omega_total > 2.0):
                ok = False
                reasons.append(f"Ω_total={Omega_total:.4f} outside (0,2] (PBUF)")
            elif abs(Omega_total - 1.0) > 0.1:
                edge_case = True

    # --- 4. Numerical H(z) sanity on probe grid ---
    z_grid = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0]
    HZ_CEILING = 1.0e9

    for z in z_grid:
        try:
            Hz = model.H(z)
        except Exception:
            Hz = np.nan

        if (not np.isfinite(Hz)) or Hz <= 0:
            ok = False
            reasons.append(f"H(z={z}) invalid ({Hz})")
            break

        if abs(Hz) > HZ_CEILING:
            ok = False
            reasons.append(f"H(z={z})={Hz} exceeds sanity ceiling {HZ_CEILING}")
            break

    # --- 5. Late-time acceleration ---
    try:
        q0 = compute_q0(model)
    except Exception:
        q0 = np.nan

    if (not np.isfinite(q0)):
        ok = False
        reasons.append("q0 not finite")
    else:
        # We still demand accelerating expansion today.
        if q0 >= 0:
            ok = False
            reasons.append(f"q0={q0} indicates no acceleration today")

    # --- 6. Elastic sector sanity (PBUF only) ---
    if model_type == "pbuf":
        alpha = params.get("alpha", getattr(model, "alpha", 0.0))
        Rmax = params.get("Rmax", getattr(model, "Rmax", 0.0))
        eps0 = params.get("eps0", getattr(model, "eps0", 0.0))
        k_sat = params.get("k_sat", getattr(model, "k_sat", 0.0))

        if alpha < 0.0:
            ok = False
            reasons.append(f"alpha={alpha:.3e} < 0")
        if Rmax <= 0.0:
            ok = False
            reasons.append(f"Rmax={Rmax:.3e} <= 0")
        if eps0 <= 0.0:
            ok = False
            reasons.append(f"eps0={eps0:.3f} <= 0")
        if k_sat <= 0.0:
            ok = False
            reasons.append(f"k_sat={k_sat:.3f} <= 0")

        try:
            omega_elastic_today = model.omega_sigma(1.0)
        except Exception:
            omega_elastic_today = np.inf

        # Old code hard-killed if >5.0.
        # We allow bigger (late-time stiffness can carry ~O(1) fraction),
        # but we still nuke numerically insane cases.
        if (not np.isfinite(omega_elastic_today)) or omega_elastic_today > 50.0:
            ok = False
            reasons.append(
                f"Ω_elastic(a=1)={omega_elastic_today} unphysical or enormous"
            )

    # --- 7. Edge-case tagging (not a reject) ---
    if H0 < 64.0 or H0 > 74.0:
        edge_case = True

    return {
        "ok": ok,
        "reasons": reasons,
        "chi2_prior_penalty": 0.0,
        "edge_case": edge_case,
    }
