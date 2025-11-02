"""
Cosmological fitting utilities.

This package contains modules for fitting cosmological models
to observational data, including CMB, supernova, BAO, cosmic
chronometer, and redshift-space distortion constraints.
"""

from . import cmb
from . import sn
from . import bao
from . import cc
from . import rsd

# Expose key CMB fitting functions
from .cmb import cmb_observables, chi_squared_cmb, fit_cmb

# Expose key SN fitting functions
from .sn import compute_sn_mu_model, chi_squared_sn, fit_sn

# Expose key BAO fitting functions
from .bao import compute_bao_dv_over_rd, chi_squared_bao_iso, fit_bao_iso

# Expose key CC fitting functions
from .cc import compute_cc_hubble_model, chi_squared_cc, fit_cc

# Expose key RSD fitting functions
from .rsd import compute_rsd_observable, chi_squared_rsd, fit_rsd

__all__ = [
    "cmb",
    "sn",
    "bao",
    "cc",
    "rsd",
    "cmb_observables",
    "chi_squared_cmb",
    "fit_cmb",
    "compute_sn_mu_model",
    "chi_squared_sn",
    "fit_sn",
    "compute_bao_dv_over_rd",
    "chi_squared_bao_iso",
    "fit_bao_iso",
    "compute_cc_hubble_model",
    "chi_squared_cc",
    "fit_cc",
    "compute_rsd_observable",
    "chi_squared_rsd",
    "fit_rsd",
]
