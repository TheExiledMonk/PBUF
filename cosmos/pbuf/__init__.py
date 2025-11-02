"""
PBUF package.

This package implements the Planck-Bound Unified Framework (PBUF),
in which late-time cosmic acceleration is sourced by elastic
spacetime response instead of a cosmological constant.

Key objects:
- PBUF (model.py): background FRW cosmology with elastic term
- omega_sigma_raw / omega_sigma_total (equations.py): standalone elastic helpers
- validation helpers (validators.py)
"""
from .model import PBUF
from .equations import (
    omega_sigma_raw,
    omega_sigma_radfix,
    omega_sigma_total,
    E2_pbuf,
    H_pbuf_a,
    H_pbuf_z,
    elastic_fraction,
)
