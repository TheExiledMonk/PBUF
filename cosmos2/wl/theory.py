"""Shared WL theory layer: kernels, Limber C_ell, and xi± transforms."""

from __future__ import annotations

import math
import warnings
from typing import Iterable, Tuple

import numpy as np

from .bessel import bessel_j0, bessel_jn
from .fftlog import safe_xi_fftlog
from .kids import KIDS_DATA_ORDER, flatten_xi_block, tomo_pairs
from .ia import IntrinsicAlignmentModel
from .photoz import apply_photoz_shifts


def _normalize_nz(nz: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    nz = np.asarray(nz, dtype=float)
    if nz.shape[-1] != z_grid.shape[-1]:
        raise ValueError("n(z) grid dimension mismatch.")
    normalized = np.empty_like(nz, dtype=float)
    for idx, row in enumerate(nz):
        area = np.trapezoid(row, z_grid)
        if not np.isfinite(area) or area <= 0.0:
            normalized[idx] = row
        else:
            normalized[idx] = row / area
    return normalized


def _lensing_kernel_single(
    nz: np.ndarray,
    z_grid: np.ndarray,
    chi: np.ndarray,
    H: np.ndarray,
    omega_m0: float,
    H0: float,
    c_light: float,
) -> np.ndarray:
    prefactor = 1.5 * (H0 / c_light) ** 2 * omega_m0
    kernel = np.zeros_like(z_grid, dtype=float)
    for idx, (z_i, chi_i, H_i) in enumerate(zip(z_grid, chi, H)):
        mask = z_grid >= z_i
        if not np.any(mask):
            continue
        chi_s = chi[mask]
        n_s = nz[mask]
        if chi_s.size == 0 or np.any(chi_s <= 0.0):
            continue
        integrand = n_s * (chi_s - chi_i) / chi_s
        integral = np.trapezoid(integrand, z_grid[mask])
        kernel[idx] = prefactor * (1.0 + z_i) * (H_i / c_light) * chi_i * integral
    return kernel


def build_lensing_kernels(
    n_of_z: np.ndarray,
    z_grid: np.ndarray,
    chi: np.ndarray,
    H: np.ndarray,
    omega_m0: float,
    H0: float,
    c_light: float,
) -> np.ndarray:
    n_bins = n_of_z.shape[0]
    kernels = np.zeros((n_bins, z_grid.size), dtype=float)
    for idx in range(n_bins):
        kernels[idx] = _lensing_kernel_single(
            n_of_z[idx], z_grid, chi, H, omega_m0, H0, c_light
        )
    return kernels


def compute_cl_matrix(
    backend,
    n_of_z: np.ndarray,
    z_grid: np.ndarray,
    kernels: np.ndarray,
    chi: np.ndarray,
    ell_grid: np.ndarray,
    nonlinear: bool = False,
    ia_model: IntrinsicAlignmentModel | None = None,
) -> np.ndarray:
    n_bins = n_of_z.shape[0]
    cls = np.zeros((n_bins, n_bins, ell_grid.size), dtype=float)
    chi_safe = np.clip(chi, 1e-6, np.inf)
    ia_present = ia_model is not None
    ia_amplitude = ia_model.amplitude(z_grid) if ia_present else None
    for i in range(n_bins):
        for j in range(i, n_bins):
            pref = kernels[i] * kernels[j] / (chi_safe * chi_safe)
            if ia_present:
                pref_gi = (kernels[i] * n_of_z[j] + kernels[j] * n_of_z[i]) / (chi_safe * chi_safe)
                pref_ii = n_of_z[i] * n_of_z[j] / (chi_safe * chi_safe)
            for ell_idx, ell in enumerate(ell_grid):
                k = (ell + 0.5) / chi_safe
                P_nl = backend.P_m_of_kz(k, z_grid, nonlinear=nonlinear)
                integrand = pref * P_nl
                if ia_present:
                    P_lin = backend.P_m_of_kz(k, z_grid, nonlinear=False)
                    P_GI = ia_amplitude * P_lin  # GI term ∝ F(z) P_lin
                    P_II = (ia_amplitude * ia_amplitude) * P_lin  # II term ∝ F(z)^2 P_lin
                    integrand = integrand + pref_gi * P_GI + pref_ii * P_II
                cls[i, j, ell_idx] = np.trapezoid(integrand, z_grid)
                cls[j, i, ell_idx] = cls[i, j, ell_idx]
    return cls


def xi_from_cls_bessel(
    cls: np.ndarray,
    ell_grid: np.ndarray,
    theta_bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xi_plus = np.zeros((cls.shape[0], cls.shape[1], theta_bins.size), dtype=float)
    xi_minus = np.zeros_like(xi_plus)
    pref = (2.0 * ell_grid + 1.0) / (4.0 * math.pi)
    for t_idx, theta in enumerate(theta_bins):
        argument = ell_grid * theta
        J0 = bessel_j0(argument)
        J4 = bessel_jn(4, argument)
        weight = pref
        weighted_J0 = weight * J0
        weighted_J4 = weight * J4
        for i in range(cls.shape[0]):
            for j in range(cls.shape[1]):
                cl_ij = cls[i, j]
                xi_plus[i, j, t_idx] = float(np.sum(cl_ij * weighted_J0))
                xi_minus[i, j, t_idx] = float(np.sum(cl_ij * weighted_J4))
    return xi_plus, xi_minus


xi_from_cls = xi_from_cls_bessel


def compute_shear_predictions(
    backend,
    data_vector: np.ndarray,
    n_of_z: np.ndarray,
    z_grid: np.ndarray,
    theta_bins: np.ndarray,
    shear_m: np.ndarray | None = None,
    photo_z_shifts: np.ndarray | None = None,
    ia_params: dict | None = None,
    include_ia: bool = False,
    use_fftlog: bool = True,
    fftlog_ell_samples: int | None = None,
    ell_min: int = 2,
    ell_max: int = 3000,
    nonlinear: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build xi⁺/xi⁻ predictions and the flattened model vector in KiDS ordering.
    """
    n_bins = n_of_z.shape[0]
    pairs = tomo_pairs(n_bins)
    chi = backend.chi_of_z(z_grid)
    H = backend.H_of_z(z_grid)
    nz_shifted = apply_photoz_shifts(n_of_z, z_grid, photo_z_shifts)
    nz_norm = _normalize_nz(nz_shifted, z_grid)
    kernels = build_lensing_kernels(
        nz_norm,
        z_grid,
        chi,
        H,
        backend.omega_m0,
        backend.H0,
        backend.c_light,
    )

    if use_fftlog:
        n_ell = int(fftlog_ell_samples or max(ell_max - ell_min + 1, 128))
        ell_grid = np.logspace(np.log10(max(ell_min, 1e-1)), math.log10(max(ell_max, 1.0)), num=n_ell, dtype=float)
    else:
        ell_grid = np.arange(float(ell_min), float(ell_max) + 1.0, dtype=float)
    ia_model = None
    if include_ia:
        params = ia_params or {}
        ia_model = IntrinsicAlignmentModel(
            A_IA=float(params.get("A_IA", params.get("A", 0.0))),
            eta_IA=float(params.get("eta_IA", params.get("eta", 0.0))),
            z0=float(params.get("z0", 0.62)),
            C1_rho_crit=float(params.get("C1_rho_crit", params.get("C1rho", None) or 0.01389)),
            omega_m0=backend.omega_m0,
            growth_function=backend.growth_D_of_z,
        )
    cls = compute_cl_matrix(
        backend,
        nz_norm,
        z_grid,
        kernels,
        chi,
        ell_grid,
        nonlinear=nonlinear,
        ia_model=ia_model,
    )
    if use_fftlog:
        try:
            xi_plus, xi_minus = safe_xi_fftlog(cls, ell_grid, theta_bins)
        except Exception:
            warnings.warn("Falling back to Bessel xi± transform.", RuntimeWarning, stacklevel=2)
            xi_plus, xi_minus = xi_from_cls_bessel(cls, ell_grid, theta_bins)
    else:
        xi_plus, xi_minus = xi_from_cls_bessel(cls, ell_grid, theta_bins)

    if shear_m is None or len(shear_m) == 0:
        shear_m = np.zeros(n_bins, dtype=float)
    shear_m = np.asarray(shear_m, dtype=float)
    if shear_m.size != n_bins:
        raise ValueError("shear_m length mismatch with tomographic bins.")
    for idx, (i, j) in enumerate(pairs):
        calibration = (1.0 + shear_m[i]) * (1.0 + shear_m[j])
        xi_plus[i, j, :] *= calibration
        xi_minus[i, j, :] *= calibration
        xi_plus[j, i, :] = xi_plus[i, j, :]
        xi_minus[j, i, :] = xi_minus[i, j, :]

    xi_plus_flat = flatten_xi_block(xi_plus, pairs)
    xi_minus_flat = flatten_xi_block(xi_minus, pairs)
    model_vector = np.concatenate([xi_plus_flat, xi_minus_flat], axis=0)
    if model_vector.shape != data_vector.shape:
        raise ValueError(f"Model vector shape {model_vector.shape} mismatches data {data_vector.shape}")
    return xi_plus, xi_minus, model_vector


__all__ = [
    "compute_shear_predictions",
    "build_lensing_kernels",
    "compute_cl_matrix",
    "xi_from_cls",
    "xi_from_cls_bessel",
    "KIDS_DATA_ORDER",
]
