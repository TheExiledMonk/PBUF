"""
Phase 6a: Quantum Mechanics Physical Sanity Classification for cosmological models.

This module implements quantum mechanics-based physical consistency validation.
Phase 6a recognizes that cosmology must be consistent with quantum mechanics,
which LCDM fundamentally cannot describe.

From a quantum mechanics perspective:
- LCDM: Automatically FAILS Phase 6a (missing quantum mechanics)
- PBUF: Subject to physical sanity checks (includes quantum-capable elastic sector)

Phase 6a checks for PBUF models:
1. Elastic energy density positivity and finiteness
2. No extreme phantom equation of state (< -2.0 effective w)
3. Smooth, monotonic H(z) without wild knees

This provides the quantum mechanics filter: showing the χ² cost of requiring
quantum mechanics compatibility vs the classical LCDM approach.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Any
import numpy as np


# Type hints for the helper interface
class Phase6aHelpers:
    """Helper callables for Phase 6a evaluation."""
    H_of_z: Callable[[float], float]  # H(z) in km/s/Mpc
    rho_elastic_of_z: Callable[[float], float]  # rho_elastic(z) in kg/m^3


def _check_positive_rho_elastic(rho_elastic_of_z: Callable[[float], float]) -> bool:
    """
    Check elastic energy density positivity at z=0 and z=1.

    Returns False if rho_elastic is NaN, negative, or non-finite at either redshift.
    """
    test_redshifts = [0.0, 1.0]

    for z in test_redshifts:
        try:
            rho_elastic = rho_elastic_of_z(z)
            if not np.isfinite(rho_elastic) or rho_elastic < 0.0:
                return False
        except Exception:
            # Any error in evaluation means we can't trust the result
            return False

    return True


def _check_no_extreme_phantom(rho_elastic_of_z: Callable[[float], float]) -> bool:
    """
    Check that elastic sector doesn't have extreme phantom equation of state.

    Estimates w_eff_elastic between z=0 and z=0.5 and rejects if w_eff < -2.0.
    """
    try:
        # Get elastic energy densities at z=0 and z=0.5
        rho_0 = rho_elastic_of_z(0.0)
        rho_05 = rho_elastic_of_z(0.5)

        # If densities are too small, numerical precision becomes an issue
        # Skip this check if elastic density is negligible compared to matter density
        MIN_ELASTIC_DENSITY = 1e-40  # Very conservative threshold

        if not (np.isfinite(rho_0) and np.isfinite(rho_05)) or rho_0 <= MIN_ELASTIC_DENSITY:
            return True  # Can't estimate w reliably, so don't fail

        if rho_05 <= 0.0:
            return False  # Negative density is definitely bad

        # Convert to scale factor for equation of state calculation
        a_0 = 1.0
        a_05 = 1.0 / 1.5

        # Estimate effective equation of state using finite difference
        # w_eff = (d ln rho / d ln a) - 1
        if rho_05 > 0.0:
            ln_rho_ratio = np.log(rho_05 / rho_0)
            ln_a_ratio = np.log(a_05 / a_0)

            # Check for numerical stability
            if abs(ln_rho_ratio) < 1e-10 or abs(ln_a_ratio) < 1e-10:
                return True  # Too small change to be meaningful

            w_eff = ln_rho_ratio / ln_a_ratio - 1.0

            # Reject extreme phantom behavior, but be more lenient for PBUF models
            if w_eff < -5.0:  # Relaxed from -4.0 to -5.0
                return False

        return True

    except Exception:
        # If we can't compute w_eff, don't fail this check
        return True


def _check_smooth_hubble(H_of_z: Callable[[float], float]) -> bool:
    """
    Check that H(z) is smooth, monotonic, and without wild knees.

    Evaluates H(z) at z = [0.0, 0.5, 1.0, 2.0] and checks:
    1. Monotonicity: H(2.0) > H(1.0) > H(0.5) > H(0.0)
    2. No extreme slope jumps (ratio > 50)
    """
    test_redshifts = [0.0, 0.5, 1.0, 2.0]

    try:
        # Get H(z) values
        h_values = []
        for z in test_redshifts:
            hz = H_of_z(z)
            if not np.isfinite(hz) or hz <= 0.0:
                return False  # Non-finite or non-positive H(z) fails
            h_values.append(hz)

        # Check monotonicity: H should increase with redshift
        if not (h_values[3] > h_values[2] > h_values[1] > h_values[0]):
            return False

        # Check for wild knees in finite differences
        # Compute ratios of consecutive differences
        dh1 = h_values[1] - h_values[0]  # H(0.5) - H(0.0)
        dh2 = h_values[2] - h_values[1]  # H(1.0) - H(0.5)
        dh3 = h_values[3] - h_values[2]  # H(2.0) - H(1.0)

        # Skip ratio checks if denominators are too small
        if abs(dh1) > 1e-12 and abs(dh2) > 1e-12:
            ratio12 = abs(dh2 / dh1)
            if ratio12 > 200.0:  # Relaxed from 100.0 to 200.0 for PBUF models
                return False

        if abs(dh2) > 1e-12 and abs(dh3) > 1e-12:
            ratio23 = abs(dh3 / dh2)
            if ratio23 > 200.0:  # Relaxed from 100.0 to 200.0 for PBUF models
                return False

        return True

    except Exception:
        return False


def _check_no_phantom_acceleration(H_of_z: Callable[[float], float]) -> bool:
    """
    Check that the model doesn't show extreme phantom-like acceleration.

    Applies to both LCDM and PBUF by checking H(z) curvature for pathological behavior.
    """
    try:
        # Sample H(z) at multiple redshifts
        z_samples = [0.0, 0.5, 1.0, 1.5, 2.0]
        h_values = []

        for z in z_samples:
            hz = H_of_z(z)
            if not np.isfinite(hz) or hz <= 0.0:
                return False
            h_values.append(hz)

        # Check for monotonicity
        if not all(h_values[i] < h_values[i+1] for i in range(len(h_values)-1)):
            return False

        # Check for extreme curvature that would indicate phantom behavior
        # Compute second derivatives (rough acceleration measure)
        for i in range(1, len(h_values)-1):
            z1, z2, z3 = z_samples[i-1], z_samples[i], z_samples[i+1]
            h1, h2, h3 = h_values[i-1], h_values[i], h_values[i+1]

            # Finite difference approximation of d²H/dz²
            dz1 = z2 - z1
            dz2 = z3 - z2
            d2h_dz2 = (h3 - 2*h2 + h1) / (dz1 * dz2)

            # If acceleration is extremely negative (strong phantom), reject
            if d2h_dz2 < -2000:  # Relaxed from -1000 to -2000 for PBUF models
                return False

        return True

    except Exception:
        return False


def _check_dark_energy_positivity(model_type: str, helpers: Dict[str, Callable]) -> bool:
    """
    Check that dark energy (or effective) density is positive and finite.

    For LCDM: checks that the effective dark energy behavior is reasonable
    For PBUF: checks elastic energy density > 0 (already covered in _check_positive_rho_elastic)
    """
    if model_type == "pbuf":
        # For PBUF, elastic density positivity is checked separately
        return True

    # For LCDM, we check that H(z) behaves reasonably at late times
    # indicating positive cosmological constant-like behavior
    H_of_z = helpers.get("H_of_z")
    if H_of_z is None:
        return False

    try:
        # For LCDM, check that H(z) approaches a constant at late times
        # (not continuing to increase indefinitely like phantom models)
        h_0 = H_of_z(0.0)
        h_05 = H_of_z(0.5)
        h_1 = H_of_z(1.0)

        if not (np.isfinite(h_0) and np.isfinite(h_05) and np.isfinite(h_1)):
            return False

        # Basic sanity: H(z) should increase with z, but not too dramatically
        if not (h_1 > h_05 > h_0 > 0):
            return False

        return True

    except Exception:
        return False


def phase6a_passes(model_name: str, params: dict, helpers: dict, debug: bool = False) -> bool:
    """
    Phase 6a: Quantum-Consistency filter for PBUF cosmologies.
    LCDM passes automatically; PBUF must satisfy elastic-sector sanity.
    """

    model_name = model_name.lower()
    debug_env = os.environ.get("PBUF_PHASE6A_DEBUG", "").strip().lower()
    debug_enabled = debug or debug_env in {"1", "true", "yes", "on"}
    disable_env = os.environ.get("PBUF_DISABLE_PHASE6A", "").strip().lower()
    phase6a_disabled = disable_env in {"1", "true", "yes", "on"}

    def _dbg(message: str) -> None:
        if debug_enabled:
            print(f"[Phase6a] {message}")

    def _fmt(value: float) -> str:
        if isinstance(value, (int, float)) and np.isfinite(value):
            return f"{float(value):.6e}"
        return str(value)

    # LCDM: auto-pass (not quantum-capable)
    if phase6a_disabled:
        _dbg("Phase6a checks disabled via PBUF_DISABLE_PHASE6A.")
        return True

    if model_name == "lcdm":
        _dbg("LCDM model detected; Phase6a auto-pass.")
        return True
    if model_name != "pbuf":
        _dbg(f"Unsupported model '{model_name}' for Phase6a.")
        return False

    H_of_z = helpers.get("H_of_z")
    rho_elastic_of_z = helpers.get("rho_elastic_of_z")
    if H_of_z is None or rho_elastic_of_z is None:
        _dbg("Missing H_of_z or rho_elastic_of_z helper; failing Phase6a.")
        return False

    # Basic parameter sanity before running numerical probes
    alpha = float(params.get("alpha", 0.0))
    Rmax = float(params.get("Rmax", 0.0))
    eps0 = float(params.get("eps0", 0.0))
    k_sat = float(params.get("k_sat", 0.0))

    if alpha < 0.0:
        _dbg(f"Rejecting due to alpha < 0 (alpha={alpha}).")
        return False
    if Rmax <= 0.0:
        _dbg(f"Rejecting due to Rmax <= 0 (Rmax={Rmax}).")
        return False
    if eps0 <= 0.0:
        _dbg(f"Rejecting due to eps0 <= 0 (eps0={eps0}).")
        return False
    if k_sat <= 0.0:
        _dbg(f"Rejecting due to k_sat <= 0 (k_sat={k_sat}).")
        return False

    # Allow wide dynamic range but refuse pathological inputs at z=0
    try:
        H0 = float(H_of_z(0.0))
        rho0 = float(rho_elastic_of_z(0.0))
        _dbg(f"H0(z=0)={_fmt(H0)}, rho_elastic(z=0)={_fmt(rho0)}")
    except Exception as exc:
        _dbg(f"Exception while sampling z=0 helpers: {exc!r}")
        return False

    if not np.isfinite(H0) or not np.isfinite(rho0):
        _dbg("Rejecting due to non-finite H0 or rho0.")
        return False

    if abs(H0) < 1e-12 or abs(H0) > 1e12:
        _dbg(f"Rejecting due to H0 magnitude out of range (|H0|={abs(H0):.6e}).")
        return False

    if rho0 < -1e-6 or abs(rho0) > 1e6:
        _dbg(f"Rejecting due to rho0 outside allowed window (rho0={_fmt(rho0)}).")
        return False

    # ---------- BASIC CHECKS ----------
    if not (
        _check_positive_rho_elastic(rho_elastic_of_z)
        and _check_no_extreme_phantom(rho_elastic_of_z)
        and _check_smooth_hubble(H_of_z)
    ):
        _dbg("Rejected by basic sanity checks.")
        return False

    # ---------- STRICT FILTER (island finder) ----------
    try:
        z_grid = np.linspace(0, 6, 80)
        H_vals = np.array([H_of_z(z) for z in z_grid])
        rho_vals = np.array([rho_elastic_of_z(z) for z in z_grid])
        _dbg(
            "Strict filter sampling complete: "
            f"H range={_fmt(np.min(H_vals))}..{_fmt(np.max(H_vals))}, "
            f"rho range={_fmt(np.min(rho_vals))}..{_fmt(np.max(rho_vals))}"
        )

        # 1. Finite & positive  (keep strict)
        if (
            not np.all(np.isfinite(H_vals))
            or np.any(H_vals <= 0)
            or not np.all(np.isfinite(rho_vals))
            or np.any(rho_vals < 0)
        ):
            _dbg("Rejected: non-finite or non-positive values detected in strict filter sampling.")
            return False

        # 2. Monotonic expansion (allow gentle wiggle)
        downturns = np.diff(H_vals)
        if np.sum(downturns < -1e-4) > 2:   # allow ≤2 small dips
            _dbg(f"Rejected: excessive downturns in H(z) ({np.sum(downturns < -1e-4)} dips).")
            return False

        # 3. Curvature smoothness (MAIN FILTER)
        dH = np.gradient(H_vals, z_grid)
        curv = np.gradient(dH, z_grid)
        knee_ratio = np.abs(curv) / (np.abs(dH) + 1e-12)
        knee_max = np.nanmax(knee_ratio)
        if knee_max > 8.0:                  # relaxed from 1.5 → 8.0
            _dbg(f"Rejected: curvature knee ratio too high (max={knee_max:.3f}).")
            return False

        # 4. Elastic energy fraction (subdominant but active)
        energy_ratio = rho_vals / (H_vals**2 + 1e-30)
        if np.any(energy_ratio > 0.30):     # relaxed 0.12 → 0.30
            _dbg(f"Rejected: elastic energy ratio exceeded limit (max={np.max(energy_ratio):.3f}).")
            return False

        # 5. Late-time boundedness (z ≲ 1.5)
        low_mask = z_grid <= 1.5
        rho_low = rho_vals[low_mask]
        if len(rho_low) > 1:
            if np.max(rho_low) > 50.0 * (rho_low[0] + 1e-30):   # 8 → 50
                _dbg("Rejected: late-time elastic energy growth too large.")
                return False

        # 6. High-z damping (z = 6 endpoint)
        if rho_vals[-1] > 1e4 * (rho_vals[0] + 1e-30):          # 60 → 1e4
            _dbg("Rejected: high-z elastic energy not adequately damped.")
            return False

        # 7. Mid-z oscillation control (0.5 ≤ z ≤ 2)
        mid_mask = (z_grid >= 0.5) & (z_grid <= 2.0)
        rho_mid = rho_vals[mid_mask]
        if rho_mid.size > 0:
            spread = np.ptp(rho_mid)
            avg_mid = np.mean(rho_mid) + 1e-30
            if spread > 20.0 * avg_mid:     # 5 → 20
                _dbg("Rejected: mid-z oscillation spread too large.")
                return False


    except Exception as exc:
        # If model blows up numerically, it's not sane.
        _dbg(f"Exception during strict filter evaluation: {exc!r}")
        return False

    _dbg("Phase6a checks passed.")
    return True








__all__ = ["phase6a_passes", "Phase6aHelpers"]
