"""
Model-agnostic interfaces that fits may rely on.

The goal is to keep this module tiny and stable so fits can speak to any model
through the same protocol without learning model-specific details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence

import numpy as np

@dataclass
class CMBOutput:
    """Common result container for CMB distance prior calculations."""

    R: float
    l_A: float
    Omega_b_h2: float
    theta_star: float
    z_star: float
    D_M_Mpc: float
    D_A_Mpc: float
    r_s_Mpc: float
    extras: Dict[str, Any] = field(default_factory=dict)


class CosmologyModel(Protocol):
    """Protocol implemented by every cosmology model."""

    def cmb(self, data: Any) -> CMBOutput:
        """
        Compute CMB distance prior observables for this model instance.

        The fit provides a data/configuration object. Models must not read files
        during this call and must rely entirely on their parameters plus the
        supplied data blob.
        """

    @property
    def parameters(self) -> Dict[str, float]:
        """Return the current set of cosmological parameters as a flat dict."""

    def omega_m0(self) -> float:
        """Present-day matter density fraction (Ωₘ,₀)."""

    def sigma8(self) -> float:
        """Present-day linear matter clustering amplitude σ₈."""

    def distance_modulus(self, z: float | Sequence[float]) -> float | np.ndarray:
        """
        Compute the distance modulus μ(z) for the supplied redshift(s).
        """

    def DV(self, z: float | Sequence[float]) -> float | np.ndarray:
        """Compute the isotropic BAO volume distance D_V(z) for the supplied redshift(s)."""

    def DM(self, z: float | Sequence[float]) -> float | np.ndarray:
        """Transverse comoving distance D_M(z) for the supplied redshift(s)."""

    def DA(self, z: float | Sequence[float]) -> float | np.ndarray:
        """Angular diameter distance D_A(z) = D_M(z)/(1+z)."""

    def DH(self, z: float | Sequence[float]) -> float | np.ndarray:
        """Radial BAO distance D_H(z) = c / H(z) for the supplied redshift(s)."""

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        """Hubble rate H(z) in km/s/Mpc for the supplied redshift(s)."""

    def sound_horizon(self) -> float:
        """Return the comoving sound horizon at the drag epoch (r_d)."""

    def growth_factor(self, z: float | Sequence[float]) -> float | np.ndarray:
        """Normalized linear growth factor D(z)."""

    def growth_rate(self, z: float | Sequence[float]) -> float | np.ndarray:
        """Growth rate f(z) = d ln D / d ln a."""

    def fs8(self, z: float | Sequence[float]) -> float | np.ndarray:
        """Model prediction for fσ₈(z)."""

    def S8(self, gamma: float = 0.5) -> float:
        """Return S₈ defined as σ₈(Ωₘ/0.3)^γ for the supplied exponent."""

    def is_valid(self) -> bool:
        """Return True when the model satisfies its internal sanity checks."""
