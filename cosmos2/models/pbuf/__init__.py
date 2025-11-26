"""PBUF per-model package (rebuild in progress)."""

from .distances import E, E_squared, H, H_z, angular_diameter_distance, comoving_distance, omega_total_at_a
from .elastic import alpha_of_a, epsilon_of_a, kmax_of_a, omega_sigma_of_a, omega_sigma_raw_of_a
from .growth import growth_ode_rhs
from .params import PBUFParams, coerce_pbuf_parameters
from .temperature import T_of_a, T_of_z
from .thermal_table import ThermalTable
from .utils import C_LIGHT, as_redshift, as_scale_factor, bisection_root, simpson_integral
from .sanity_base import SanityResult
from .phase6a_common import (
    CurvatureStats,
    Phase6aConfig,
    compute_H_grid,
    compute_curvature_stats,
    curvature_check,
    curvature_gate,
    load_phase6a_config,
    make_curvature_grid,
)
from .sanity import (
    Phase6aElasticStats,
    check_pbuf_sanity,
    phase6a_early_sanity,
    phase6a_elastic_curvature,
)
from .phase6a import make_phase6a_checker
from .phase7a import (
    Phase7aConfig,
    check_pbuf_phase7a_sanity,
    load_phase7a_config,
    make_phase7a_checker,
)
from .model import PBUFModel
from .normalization import apply_omega_normalization, normalize_parameters, resolve_alpha
from .fits import (
    PBUF_FIT_REGISTRY,
    build_pbuf_joint_chi2,
    get_pbuf_fit,
    resolve_pbuf_joint_fits,
    run_bao_aniso_fit,
    run_bao_iso_fit,
    run_cc_fit,
    run_cmb_fit,
    run_galaxy_pk_fit,
    run_lensing_cross_fit,
    run_rsd_fit,
    run_sh0es_prior,
    run_sn_pantheon_fit,
    run_wl_s8_fit,
)
from .cmb import (
    compute_cmb_output,
    c_s,
    h,
    Omega_b_h2,
    Omega_m_h2,
    R_b,
    sound_horizon,
    sound_horizon_drag,
    z_drag_eh,
    z_star_hu_sugiyama,
)

__all__ = [
    "C_LIGHT",
    "as_redshift",
    "as_scale_factor",
    "bisection_root",
    "simpson_integral",
    "alpha_of_a",
    "epsilon_of_a",
    "kmax_of_a",
    "omega_sigma_raw_of_a",
    "omega_sigma_of_a",
    "omega_total_at_a",
    "E_squared",
    "E",
    "H",
    "H_z",
    "comoving_distance",
    "angular_diameter_distance",
    "growth_ode_rhs",
    "PBUFParams",
    "coerce_pbuf_parameters",
    "apply_omega_normalization",
    "normalize_parameters",
    "resolve_alpha",
    "Phase6aConfig",
    "CurvatureStats",
    "compute_H_grid",
    "compute_curvature_stats",
    "curvature_check",
    "curvature_gate",
    "make_curvature_grid",
    "Phase6aElasticStats",
    "phase6a_early_sanity",
    "phase6a_elastic_curvature",
    "check_pbuf_sanity",
    "SanityResult",
    "make_phase6a_checker",
    "Phase7aConfig",
    "check_pbuf_phase7a_sanity",
    "load_phase7a_config",
    "make_phase7a_checker",
    "T_of_a",
    "T_of_z",
    "ThermalTable",
    "h",
    "Omega_b_h2",
    "Omega_m_h2",
    "R_b",
    "c_s",
    "z_star_hu_sugiyama",
    "z_drag_eh",
    "sound_horizon",
    "sound_horizon_drag",
    "compute_cmb_output",
    "PBUFModel",
    "PBUF_FIT_REGISTRY",
    "build_pbuf_joint_chi2",
    "get_pbuf_fit",
    "resolve_pbuf_joint_fits",
    "run_bao_aniso_fit",
    "run_bao_iso_fit",
    "run_cc_fit",
    "run_cmb_fit",
    "run_galaxy_pk_fit",
    "run_lensing_cross_fit",
    "run_rsd_fit",
    "run_sh0es_prior",
    "run_sn_pantheon_fit",
    "run_wl_s8_fit",
]
