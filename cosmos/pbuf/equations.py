"""
Core background equations for the Path A PBUF elastic cosmology.

	The restored formulation enforces FRW closure explicitly:

	    E²(a) = Ω_m a^-3 + Ω_r a^-4 + Ω_k a^-2 + Ω_σ(a)

	where Ω_σ(a=1) = 1 − Ω_m0 − Ω_r0 − Ω_k0 and the redshift dependence
	is provided by a smooth elastic envelope.  No cosmological constant
	fallback is present.
"""

from __future__ import annotations

import numpy as np

from .utils import _as_array, _maybe_scalar
from ..helper.guards import check_scale_factor

_MIN_POSITIVE = 1.0e-12
_CLOSURE_TOL = 1.0e-9


def _z_from_a(a: np.ndarray) -> np.ndarray:
    """Convert scale factor(s) to redshift."""
    return (1.0 / a) - 1.0


def _closure_omega_sigma0(params) -> float:
    """Return Ω_σ0 required by FRW closure."""
    import traceback
    import sys
    
    Om0 = float(params["Om0"])
    Or0 = float(params["Or0"])
    Ok0 = float(params.get("Ok0", 0.0))

    closure = 1.0 - (Om0 + Or0 + Ok0)

    if closure < -_CLOSURE_TOL:
        print("\n" + "="*80, file=sys.stderr)
        print("ERROR: PBUF CLOSURE VIOLATION", file=sys.stderr)
        print(f"Ω_m0 = {Om0}", file=sys.stderr)
        print(f"Ω_r0 = {Or0}", file=sys.stderr)
        print(f"Ω_k0 = {Ok0}", file=sys.stderr)
        print(f"Sum = {Om0 + Or0 + Ok0}", file=sys.stderr)
        print("\nTRACEBACK:", file=sys.stderr)
        traceback.print_stack(limit=10, file=sys.stderr)
        print("\n" + "="*80, file=sys.stderr)
        
        raise ValueError(
            "PBUF closure requires Ω_σ ≥ 0 but received Ω_σ0 = "
            f"{closure:.6f} (inputs Ω_m0={Om0}, Ω_r0={Or0}, Ω_k0={Ok0})"
        )
    return max(closure, 0.0)


def _elastic_shape(one_plus_z: np.ndarray, params) -> np.ndarray:
    """
    Smooth turn-on profile S(z) ∈ (0, 1], unity today and vanishing at high z.

    Parameters
    ----------
    one_plus_z : np.ndarray
        1 + z evaluated at the desired points.
    params : dict
        Elastic parameter dictionary (Rmax, n_alpha, n_eps, n_R, k_sat, alpha, eps0).
    """
    Rmax = float(params.get("Rmax", 1.0))
    Rmax = max(abs(Rmax), _MIN_POSITIVE)

    # Shape exponents – ensure they remain strictly positive.
    n_turn = float(params.get("n_alpha", 0.0)) + 1.0
    n_turn = max(n_turn, 1.0e-3)

    q_shape = float(params.get("n_eps", 0.0)) + 1.0
    q_shape = max(q_shape, 1.0e-3)

    # Optional tilt of the transition with redshift (n_R).
    tilt = float(params.get("n_R", 0.0))
    # Avoid zero / negative bases.
    one_plus_z = np.clip(one_plus_z, _MIN_POSITIVE, None)
    R_eff = Rmax * np.power(one_plus_z, -tilt)

    ratio = np.power(one_plus_z / np.maximum(R_eff, _MIN_POSITIVE), n_turn)
    base_profile = np.power(1.0 + ratio, -q_shape)

    # Saturation control: exponentiate the profile so that k_sat > 1 sharpens
    # the fall-off, whereas 0 < k_sat < 1 keeps elasticity active longer.
    k_sat = float(params.get("k_sat", 1.0))
    k_sat = np.clip(k_sat, 1.0e-3, 5.0)
    shaped = np.power(base_profile, k_sat)

    # Optional extra smoothing via alpha / eps0 if supplied.
    alpha = float(params.get("alpha", 1.0))
    alpha = max(alpha, 1.0e-3)
    eps0 = float(params.get("eps0", 1.0))
    eps0 = max(eps0, 1.0e-3)

    smoothing_exponent = 0.5 * (alpha + eps0)
    profile = np.power(shaped, smoothing_exponent)

    # Normalise so that S(z=0) = 1 regardless of the elastic knobs.
    ratio_today = np.power(1.0 / max(Rmax, _MIN_POSITIVE), n_turn)
    base_today = (1.0 + ratio_today) ** (-q_shape)
    shaped_today = base_today ** k_sat
    profile_today = max(shaped_today ** smoothing_exponent, _MIN_POSITIVE)

    normalised = profile / profile_today
    normalised = np.clip(normalised, 0.0, 1.0)
    return normalised


def omega_sigma_raw(a, params):
    """
    Closure-respecting elastic density fraction with the high-z envelope applied.

    This function returns Ω_σ(a) prior to any additional corrections; closure
    fixes Ω_σ0 = 1 − Ω_m0 − Ω_r0 − Ω_k0.  The remaining redshift dependence is
    governed by the smooth turn-on profile.
    """
    a_array, was_scalar = _as_array(a)
    check_scale_factor(a_array)

    one_plus_z = 1.0 / a_array

    omega_sigma0 = _closure_omega_sigma0(params)
    if omega_sigma0 == 0.0:
        zeros = np.zeros_like(a_array, dtype=float)
        return _maybe_scalar(zeros, was_scalar)

    profile = _elastic_shape(one_plus_z, params)
    result = omega_sigma0 * profile
    return _maybe_scalar(np.asarray(result, dtype=float), was_scalar)


def omega_sigma_radfix(a, params):
    """
    Radiation-era tweak placeholder.

    The new closure-driven elastic sector no longer requires an explicit
    radiation fix-up.  We retain the function for compatibility, returning 0.
    """
    a_array, was_scalar = _as_array(a)
    check_scale_factor(a_array)

    zeros = np.zeros_like(a_array, dtype=float)
    result = np.asarray(zeros, dtype=float)
    return _maybe_scalar(result, was_scalar)


def omega_sigma_total(a, params):
    """
    Full elastic correction Ω_σ(a) added to the Friedmann budget.

    Ω_σ(a) = k_sat · Ω_σ,raw(a) + Ω_σ,radfix(a)
    """
    a_array, was_scalar = _as_array(a)
    check_scale_factor(a_array)

    raw = np.asarray(omega_sigma_raw(a_array, params), dtype=float)
    rad = np.asarray(omega_sigma_radfix(a_array, params), dtype=float)

    total = raw + rad
    return _maybe_scalar(np.asarray(total, dtype=float), was_scalar)


def E2_pbuf(a, params):
    """
    Path A Friedmann background with no Λ fallback.

    Parameters
    ----------
    a : float or array
        Scale factor(s), 0 < a ≤ 1.
    params : dict
        Must contain H0 (km/s/Mpc), Om0, Or0, Ok0, alpha, Rmax,
        k_sat, eps0, n_alpha, n_eps, n_R.
    """
    a_array, was_scalar = _as_array(a)
    check_scale_factor(a_array)

    Om0 = float(params["Om0"])
    Or0 = float(params["Or0"])
    Ok0 = float(params.get("Ok0", 0.0))

    base = (
        Om0 * a_array**-3 +
        Or0 * a_array**-4 +
        Ok0 * a_array**-2
    )
    elastic = np.asarray(omega_sigma_total(a_array, params), dtype=float)

    E2 = np.asarray(base, dtype=float) + elastic
    if np.any(E2 <= 0.0) or np.any(~np.isfinite(E2)):
        raise ValueError(f"PBUF background produced non-positive E² for a={a}: {E2}")

    return _maybe_scalar(E2, was_scalar)


def H_pbuf_a(a, params):
    """
    Hubble rate H(a) in km/s/Mpc for the Path A PBUF background.
    """
    a_array, was_scalar = _as_array(a)
    check_scale_factor(a_array)

    E2 = np.asarray(E2_pbuf(a_array, params), dtype=float)
    Hz = float(params["H0"]) * np.sqrt(E2)
    return _maybe_scalar(Hz, was_scalar)


def H_pbuf_z(z, params):
    """
    Convenience wrapper for H(z) using the Path A background.
    """
    z_array, was_scalar = _as_array(z)
    one_plus_z = 1.0 + z_array
    if np.any(one_plus_z <= 0.0):
        raise ValueError(f"Redshift {z} gives non-positive scale factor.")

    a_array = 1.0 / one_plus_z
    Hz = H_pbuf_a(a_array, params)
    return _maybe_scalar(Hz, was_scalar)


def elastic_fraction(a, params):
    """
    Fraction of the expansion budget sourced by elasticity at scale factor a.
    """
    a_array, was_scalar = _as_array(a)
    check_scale_factor(a_array)

    elastic = np.asarray(omega_sigma_total(a_array, params), dtype=float)
    total = np.asarray(E2_pbuf(a_array, params), dtype=float)
    frac = elastic / total
    return _maybe_scalar(frac, was_scalar)
