"""Utilities for constructing physics-ready event dictionaries."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .time_conversion import compute_dt_gamma

SIGMA_T_GW_DEFAULT = 1.0e-3
SIGMA_T_GBM_DEFAULT = 2.0e-6
_CHANNEL_KEYS = ["t_obs", "sigma_t", "mass_eV", "E_eV"]


def build_gw_channel(
    sigma_t: float = SIGMA_T_GW_DEFAULT,
    mass_eV: float = 0.0,
    E_eV: float | None = None,
) -> Dict[str, float | None]:
    """Return canonical GW channel definition referenced to merger time."""
    return {
        "t_obs": 0.0,
        "sigma_t": float(sigma_t),
        "mass_eV": float(mass_eV),
        "E_eV": None if E_eV is None else float(E_eV),
    }


def build_gamma_channel(
    gw_gps: float,
    trig_met: float,
    sigma_t: float = SIGMA_T_GBM_DEFAULT,
    mass_eV: float = 0.0,
    E_eV: float | None = None,
) -> Dict[str, float | None]:
    """Build gamma-ray channel referred to the GW merger time."""
    dt = compute_dt_gamma(float(gw_gps), float(trig_met))
    return {
        "t_obs": float(dt),
        "sigma_t": float(sigma_t),
        "mass_eV": float(mass_eV),
        "E_eV": None if E_eV is None else float(E_eV),
    }


def build_intrinsic_lag_model(mean: float = 0.0, sigma: float = 5.0) -> Dict[str, float]:
    """Return intrinsic lag model specification."""
    return {
        "mean": float(mean),
        "sigma": float(sigma),
    }


def build_event(
    event_id: str,
    gw_gps: float,
    gw_sigma_t: float,
    gamma_trig_met: float,
    gamma_sigma_t: float,
    L_Mpc: float,
    intrinsic_mean: float = 0.0,
    intrinsic_sigma: float = 5.0,
    gw_mass_eV: float = 0.0,
    gw_E_eV: float | None = None,
    gamma_mass_eV: float = 0.0,
    gamma_E_eV: float | None = None,
    additional_channels: Mapping[str, Mapping[str, float | None]] | None = None,
    likelihood_channels: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Compose a complete multi-messenger event dictionary."""
    extra_channels: Dict[str, Dict[str, float | None]] = {}
    if additional_channels:
        extra_channels = {
            name: {
                "t_obs": float(data["t_obs"]),
                "sigma_t": float(data["sigma_t"]),
                "mass_eV": float(data["mass_eV"]),
                "E_eV": None if data.get("E_eV") is None else float(data["E_eV"]),
            }
            for name, data in additional_channels.items()
        }

    event = {
        "id": str(event_id),
        "L_Mpc": float(L_Mpc),
        "channels": {
            "gw": build_gw_channel(
                sigma_t=gw_sigma_t,
                mass_eV=gw_mass_eV,
                E_eV=gw_E_eV,
            ),
            "gamma": build_gamma_channel(
                gw_gps=gw_gps,
                trig_met=gamma_trig_met,
                sigma_t=gamma_sigma_t,
                mass_eV=gamma_mass_eV,
                E_eV=gamma_E_eV,
            ),
        },
        "intrinsic_lag_model": build_intrinsic_lag_model(
            mean=intrinsic_mean,
            sigma=intrinsic_sigma,
        ),
    }
    event["channels"].update(extra_channels)
    if likelihood_channels:
        event["likelihood_channels"] = list(likelihood_channels)
    else:
        event["likelihood_channels"] = ["gw", "gamma"]
    return event


def validate_event_schema(event: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate pipeline event schema prior to persistence."""
    required_keys = ["id", "L_Mpc", "channels", "intrinsic_lag_model"]
    for key in required_keys:
        if key not in event:
            return False, f"Missing required key: {key}"

    try:
        if float(event["L_Mpc"]) <= 0.0:
            return False, "L_Mpc must be positive"
    except (TypeError, ValueError):
        return False, "L_Mpc must be numeric"

    channels = event["channels"]
    if not isinstance(channels, dict):
        return False, "Channels must be provided as a dictionary"

    for required_channel in ("gw", "gamma"):
        if required_channel not in channels:
            return False, f"Missing '{required_channel}' channel"

    for channel_name, channel_data in channels.items():
        if not isinstance(channel_data, dict):
            return False, f"Channel '{channel_name}' must be a dictionary"
        for key in _CHANNEL_KEYS:
            if key not in channel_data:
                return False, f"Channel '{channel_name}' missing key: {key}"
        try:
            sigma_t = float(channel_data["sigma_t"])
            if sigma_t <= 0.0:
                return False, f"Channel '{channel_name}' sigma_t must be positive"
        except (TypeError, ValueError):
            return False, f"Channel '{channel_name}' sigma_t must be numeric"
        try:
            mass_eV = float(channel_data["mass_eV"])
            if mass_eV < 0.0:
                return False, f"Channel '{channel_name}' mass_eV cannot be negative"
        except (TypeError, ValueError):
            return False, f"Channel '{channel_name}' mass_eV must be numeric"
        E_eV = channel_data.get("E_eV")
        if E_eV is None:
            if mass_eV > 0.0:
                return False, f"Channel '{channel_name}' with mass_eV > 0 must specify E_eV"
        else:
            try:
                energy = float(E_eV)
            except (TypeError, ValueError):
                return False, f"Channel '{channel_name}' E_eV must be numeric when provided"
            if energy <= 0.0:
                return False, f"Channel '{channel_name}' E_eV must be positive"
            if energy < mass_eV:
                return False, f"Channel '{channel_name}' E_eV must exceed mass_eV"

    lag_model = event["intrinsic_lag_model"]
    if not isinstance(lag_model, dict):
        return False, "Intrinsic lag model must be a dictionary"

    for key in ("mean", "sigma"):
        if key not in lag_model:
            return False, f"Intrinsic lag model missing key: {key}"

    try:
        float(lag_model["mean"])
    except (TypeError, ValueError):
        return False, "Intrinsic lag mean must be numeric"

    try:
        sigma = float(lag_model["sigma"])
        if sigma < 0.0:
            return False, "Intrinsic lag sigma must be non-negative"
    except (TypeError, ValueError):
        return False, "Intrinsic lag sigma must be numeric"

    likelihood = event.get("likelihood_channels")
    if likelihood is not None:
        if not isinstance(likelihood, (list, tuple)):
            return False, "likelihood_channels must be a sequence of channel names"
        if len(likelihood) < 2:
            return False, "likelihood_channels must list at least two channels"
        for name in likelihood:
            if name not in channels:
                return False, f"likelihood channel '{name}' not found in channels"

    return True, ""


__all__ = [
    "SIGMA_T_GW_DEFAULT",
    "SIGMA_T_GBM_DEFAULT",
    "build_event",
    "build_gamma_channel",
    "build_gw_channel",
    "build_intrinsic_lag_model",
    "validate_event_schema",
]
