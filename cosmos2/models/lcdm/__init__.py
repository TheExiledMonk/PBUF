"""LCDM per-model package for cosmos2 (aligned with PBUF layout)."""

from .common import CMBOutput
from .model import LCDMModel
from .params import LCDMParams, coerce_lcdm_parameters
from .distances import (
    E,
    E_squared,
    H,
    H_z,
    angular_diameter_distance,
    comoving_distance,
    comoving_distance_grid,
    omega_total_at_a,
)
from .cmb import (
    compute_cmb_output,
    h,
    Omega_b_h2,
    Omega_m_h2,
    sound_horizon,
    sound_horizon_drag,
    z_star_hu_sugiyama,
)
from .utils import C_LIGHT, as_redshift, as_scale_factor, simpson_integral

__all__ = [
    "LCDMModel",
    "CMBOutput",
    "LCDMParams",
    "coerce_lcdm_parameters",
    "C_LIGHT",
    "as_redshift",
    "as_scale_factor",
    "simpson_integral",
    "omega_total_at_a",
    "E_squared",
    "E",
    "H",
    "H_z",
    "comoving_distance",
    "comoving_distance_grid",
    "angular_diameter_distance",
    "compute_cmb_output",
    "h",
    "Omega_b_h2",
    "Omega_m_h2",
    "sound_horizon",
    "sound_horizon_drag",
    "z_star_hu_sugiyama",
]
