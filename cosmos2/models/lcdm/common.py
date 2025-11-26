"""Lightweight shared structures for cosmos2 models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CMBOutput:
    """Common result container for CMB distance priors."""

    R: float
    l_A: float
    Omega_b_h2: float
    theta_star: float
    z_star: float
    D_M_Mpc: float
    D_A_Mpc: float
    r_s_Mpc: float
    extras: Dict[str, Any] = field(default_factory=dict)


__all__ = ["CMBOutput"]
