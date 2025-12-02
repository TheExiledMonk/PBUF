"""Parameter dataclass for the PBUF V11 cosmology (ported from cosmos_old)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class PBUFParams:
    """PBUF background parameters with user-supplied R_max."""

    H0: float
    Omega_m0: float
    Rmax: float
    Omega_b0: float = 0.0
    Omega_r0: float = 9e-5
    alpha: float = 0.0
    omega_normalization: str = "flat_today"
    sigma_rescale: float = 1.0


def _normalize_mode_name(mode: Any) -> str:
    text = str(mode).strip().lower() if mode is not None else ""
    if text in {"free"}:
        return "free"
    if text in {"flat_today", "flat-today", "flat", ""}:
        return "flat_today"
    raise ValueError(f"Unknown omega_normalization '{mode}'. Expected 'free' or 'flat_today'.")


def coerce_pbuf_parameters(params: Dict[str, Any], *, normalization_mode: Optional[str] = None) -> Dict[str, float]:
    """
    Return a sanitized dictionary of PBUF parameters, inferring normalization mode.
    """

    required = ["H0", "Rmax"]
    missing: List[str] = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing required PBUF parameters: {missing}")

    cleaned: Dict[str, float] = {}
    for key in required:
        try:
            cleaned[key] = float(params[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PBUF parameter '{key}' must be numeric.") from exc

    if cleaned["Rmax"] <= 0.0:
        raise ValueError("Rmax must be positive and supplied explicitly.")

    cleaned.setdefault("Omega_r0", 9e-5)
    cleaned["alpha"] = float(params.get("alpha", 0.0))
    cleaned["Omega_b0"] = float(params.get("Omega_b0", 0.0))
    # Omega_m0 is derived later, so we just stash a placeholder here for compatibility.
    cleaned["Omega_m0"] = float(params.get("Omega_m0", 0.0))

    legacy = [key for key in ("eps0", "Omega_k0") if key in params]
    if legacy:
        warnings.warn(
            "PBUFModel ignores legacy eps0/alpha inputs; Quantum microphysics now supplies these.",
            DeprecationWarning,
            stacklevel=2,
        )

    candidates: Iterable[Any] = (
        (params.get("omega_normalization"), params.get("normalization_mode"), normalization_mode)
    )
    requested_mode = next((value for value in candidates if value is not None), None)
    cleaned["omega_normalization"] = _normalize_mode_name(requested_mode)
    cleaned["sigma_rescale"] = 1.0

    return cleaned


__all__ = ["PBUFParams", "coerce_pbuf_parameters"]
