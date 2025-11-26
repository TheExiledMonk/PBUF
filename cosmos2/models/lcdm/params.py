"""Parameter dataclass for the LCDM cosmology (per-model layout)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class LCDMParams:
    H0: float
    Omega_m0: float
    Omega_b0: float
    Omega_r0: float = 9.0e-5
    Omega_k0: float = 0.0
    sigma8_0: float = 0.811


def coerce_lcdm_parameters(params: Dict[str, Any]) -> Dict[str, float]:
    """Return a sanitized dictionary of LCDM parameters."""

    required = ["H0", "Omega_m0", "Omega_b0"]
    missing: List[str] = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing required LCDM parameters: {missing}")

    cleaned: Dict[str, float] = {}
    for key in required:
        try:
            cleaned[key] = float(params[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LCDM parameter '{key}' must be numeric.") from exc

    cleaned.setdefault("Omega_r0", 9.0e-5)
    cleaned.setdefault("Omega_k0", 0.0)
    cleaned.setdefault("sigma8_0", 0.811)
    return cleaned


__all__ = ["LCDMParams", "coerce_lcdm_parameters"]
