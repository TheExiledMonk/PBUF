"""
Anisotropic BAO fitting interface for cosmological models.

This module fits BAO measurements that separately constrain
transverse and radial distances, e.g.:

    D_M(z) / r_d
    D_H(z) / r_d = c / [H(z) * r_d]

It supports per-redshift covariance and multi-redshift block covariance.

Exports:
- compute_bao_anisotropic_observables
- chi_squared_bao_aniso
- fit_bao_aniso
"""

from .observables import compute_bao_anisotropic_observables, chi2_bao_aniso
from .chi2 import chi_squared_bao_aniso
from .optimizer import fit_bao_aniso

__all__ = [
    "compute_bao_anisotropic_observables",
    "chi2_bao_aniso",
    "chi_squared_bao_aniso",
    "fit_bao_aniso",
]
