"""
ΛCDM (Lambda-Cold-Dark-Matter) baseline cosmology.

Defines:
- LCDM class for background expansion
- Core Friedmann equations and utilities
"""

from .model import LCDM
from .equations import (
    H_lcdm_a,
    H_lcdm_z,
    E_lcdm_a,
)
