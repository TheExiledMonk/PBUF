"""
Redshift-Space Distortion (RSD) fitting interface.

RSD datasets measure fσ8(z), the product of the growth rate f(z)
and the RMS matter fluctuation σ8(z).

Exports:
- compute_rsd_observable
- chi_squared_rsd
- fit_rsd
"""

from .observables import compute_rsd_observable, chi2_rsd
from .chi2 import chi_squared_rsd
from .optimizer import fit_rsd

__all__ = [
    "compute_rsd_observable",
    "chi2_rsd",
    "chi_squared_rsd",
    "fit_rsd",
]
