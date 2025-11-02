"""
Joint cosmological fitter.

This package combines multiple dataset χ² likelihoods
(CMB, SN, BAO iso/aniso, CC, RSD, etc.)
into a unified optimizer for LCDM or PBUF.

Exports:
- available_datasets()
- compute_joint_chi2()
- fit_joint()
- run_joint_capture()
"""

from .registry import available_datasets
from .likelihoods import compute_joint_chi2
from .optimizer import fit_joint
from .joint_capture import run_joint_capture

__all__ = [
    "available_datasets",
    "compute_joint_chi2",
    "fit_joint",
    "run_joint_capture",
]
