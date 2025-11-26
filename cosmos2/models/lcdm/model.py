"""LCDM model backed by per-model helpers (aligned with PBUF layout)."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from cosmos2.kernels import lcdm_math
from cosmos2.kernels.common import distances
from cosmos2.models.lcdm.common import CMBOutput

from . import cmb as cmb_module
from . import distances as lcdm_distances
from .params import LCDMParams, coerce_lcdm_parameters
from .utils import C_LIGHT


class LCDMModel:
    """
    Minimal LCDM implementation that satisfies the CosmologyModel protocol while
    delegating heavy lifting to the cosmos2 kernels.
    """

    def __init__(self, *, use_cosmos_backend: bool | None = None, **params: float) -> None:
        clean = {
            "H0": params.get("H0", 67.4),
            "Omega_m0": params.get("Omega_m0", 0.315),
            "Omega_b0": params.get("Omega_b0", 0.049),
            "Omega_r0": params.get("Omega_r0", 9.0e-5),
            "Omega_k0": params.get("Omega_k0", 0.0),
            "sigma8_0": params.get("sigma8_0", 0.811),
        }
        self._params_dict: Dict[str, float] = {k: float(v) for k, v in clean.items()}
        self._params = LCDMParams(**self._params_dict)
        self._build_grids()

    # --------------------------
    # CosmologyModel properties
    # --------------------------
    @property
    def parameters(self) -> Dict[str, float]:
        return dict(self._params_dict)

    def omega_m0(self) -> float:
        # Keep the accessor consistent with the grid defaults (Omega_m0=0.3 when unspecified).
        return float(self._params_dict.get("Omega_m0", 0.3))

    def sigma8(self) -> float:
        return self._sigma8

    def cmb(self, data: Any) -> CMBOutput:
        use_legacy = os.getenv("COSMOS2_CMB_LEGACY", "").strip().lower() in {"1", "true", "yes", "on"}

        z_star = cmb_module.z_star_hu_sugiyama(self._params)
        H0 = float(self._params.H0)
        Om0 = float(self._params.Omega_m0)
        Ob0 = float(self._params.Omega_b0)
        Ok0 = float(self._params.Omega_k0)
        Or0_fixed = float(self._params.Omega_r0)
        Or0_dyn, Og0 = lcdm_math.omega_r0_from_Tcmb(H0, T_cmb=2.7255, N_eff=3.046)
        Or0 = Or0_dyn if Or0_dyn > 0.0 else Or0_fixed
        Ol0 = 1.0 - Om0 - Or0 - Ok0
        DM = lcdm_math.comoving_distance_to_z(float(z_star), Om0, Or0, Ok0, Ol0, H0, steps=50000)
        r_s = lcdm_math.sound_horizon_high_z(
            float(z_star),
            1.0e5,
            Ob0,
            Om0,
            Or0,
            Og0,
            Ok0,
            H0,
            steps=200000,
        )

        D_A = DM / (1.0 + float(z_star))
        R = math.sqrt(self.omega_m0()) * (self.parameters.get("H0", 70.0) * DM / C_LIGHT)
        l_A = math.pi * DM / r_s if r_s > 0.0 else float("inf")
        theta_star = r_s / DM if DM > 0.0 else float("inf")
        omega_b_h2 = float(self.parameters.get("Omega_b0", 0.0)) * (self.parameters.get("H0", 70.0) / 100.0) ** 2
        return CMBOutput(
            R=R,
            l_A=l_A,
            Omega_b_h2=omega_b_h2,
            theta_star=theta_star,
            z_star=float(z_star),
            D_M_Mpc=DM,
            D_A_Mpc=D_A,
            r_s_Mpc=r_s,
            extras={"engine": "cosmos2"},
        )

    def distance_modulus(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        DM = np.interp(a, self._a_grid, self._DM)
        mu = _distance_modulus_from_DM(DM, z_arr)
        return float(mu[0]) if scalar else mu

    def DV(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        DM = np.interp(a, self._a_grid, self._DM)
        H = np.interp(a, self._a_grid, self._H)
        dv = _dv_from_DM_H(z_arr, DM, H)
        return float(dv[0]) if scalar else dv

    def DM(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        vals = np.interp(a, self._a_grid, self._DM)
        return float(vals[0]) if scalar else vals

    def DA(self, z: float | Sequence[float]) -> float | np.ndarray:
        vals = self.DM(z)
        z_arr, scalar = _to_array(z)
        vals = np.asarray(vals, dtype=float) / (1.0 + z_arr)
        return float(vals[0]) if scalar else vals

    def DH(self, z: float | Sequence[float]) -> float | np.ndarray:
        hubble = self.Hubble(z)
        hubble = np.asarray(hubble, dtype=float)
        vals = C_LIGHT / hubble
        z_arr, scalar = _to_array(z)
        return float(vals[0]) if scalar else vals

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        vals = np.interp(a, self._a_grid, self._H)
        return float(vals[0]) if scalar else vals

    def sound_horizon(self) -> float:
        return self._r_d

    def growth_factor(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a = 1.0 / (1.0 + z_arr)
        vals = np.interp(a, self._a_grid, self._D)
        return float(vals[0]) if scalar else vals

    def growth_rate(self, z: float | Sequence[float]) -> float | np.ndarray:
        z_arr, scalar = _to_array(z)
        a_vals = 1.0 / (1.0 + z_arr)
        vals = np.interp(a_vals, self._a_grid, self._f_grid)
        return float(vals[0]) if scalar else vals

    def fs8(self, z: float | Sequence[float]) -> float | np.ndarray:
        D_vals = self.growth_factor(z)
        f_vals = self.growth_rate(z)
        z_arr, scalar = _to_array(z)
        vals = np.asarray(f_vals, dtype=float) * self._sigma8 * np.asarray(D_vals, dtype=float)
        return float(vals[0]) if scalar else vals

    def S8(self, gamma: float = 0.5) -> float:
        return float(self._sigma8 * (self.omega_m0() / 0.3) ** gamma)

    def is_valid(self) -> bool:
        return bool(np.all(self._E > 0.0))

    # --------------------------
    # Internal helpers
    # --------------------------
    def _build_grids(self) -> None:
        n_grid = int(self._params_dict.get("n_grid", 80000))
        a_min = 1.0e-6
        self._a_grid = np.logspace(np.log10(a_min), 0.0, n_grid, dtype=np.float64)
        par_arr = np.array(
            [
                self._params_dict.get("H0", 70.0),
                self._params_dict.get("Omega_m0", 0.3),
                self._params_dict.get("Omega_b0", 0.05),
                self._params_dict.get("Omega_k0", 0.0),
                self._params_dict.get("Omega_r0", 9.0e-5),
                self._params_dict.get(
                    "Omega_Lambda",
                    1.0 - self._params_dict.get("Omega_m0", 0.3) - self._params_dict.get("Omega_r0", 9.0e-5) - self._params_dict.get("Omega_k0", 0.0),
                ),
            ],
            dtype=np.float64,
        )
        E_of_a, H_of_a, D_of_a, r_d, sigma8 = lcdm_math.kernel_lcdm_math(par_arr, self._a_grid)
        self._E = E_of_a
        self._H = H_of_a
        self._D = D_of_a
        # Growth rate f = d ln D / d ln a = (a/D) * dD/da
        dD_da = np.gradient(self._D, self._a_grid)
        safe_D = np.clip(self._D, 1e-6, None)
        f_grid = (self._a_grid / safe_D) * dD_da
        # Guard against any numerical spillover.
        self._f_grid = np.nan_to_num(f_grid, nan=0.0, posinf=0.0, neginf=0.0)
        self._r_d = float(r_d)
        sigma8_kernel = float(sigma8)
        self._sigma8 = float(self._params_dict.get("sigma8_0", sigma8_kernel))
        curvature = float(self._params_dict.get("Omega_k0", 0.0))
        # High-accuracy comoving distance in z-space
        chi = distances.comoving_distance_simpson_z(self._a_grid, self._E, float(self._params_dict.get("H0", 70.0)))
        DM = np.empty_like(chi)
        for i, val in enumerate(chi):
            DM[i] = distances.transverse_comoving_distance(val, float(self._params_dict.get("H0", 70.0)), curvature)
        self._DM = DM


def _to_array(z: float | Sequence[float]) -> tuple[np.ndarray, bool]:
    arr = np.atleast_1d(np.asarray(z, dtype=float))
    return arr, np.isscalar(z)


def _dv_from_DM_H(z: np.ndarray, DM: np.ndarray, H: np.ndarray) -> np.ndarray:
    dv = np.empty_like(z, dtype=float)
    for i in range(z.shape[0]):
        z_val = float(z[i])
        dm = float(DM[i])
        hubble = float(H[i])
        if z_val < 0.0 or not math.isfinite(dm) or not math.isfinite(hubble) or hubble <= 0.0:
            dv[i] = float("inf")
            continue
        denom = 1.0 + z_val
        if denom <= 0.0:
            dv[i] = float("inf")
            continue
        da = dm / denom
        factor = z_val * (1.0 + z_val) * (1.0 + z_val) * da * da * C_LIGHT / hubble
        dv[i] = float("inf") if factor <= 0.0 else float(factor ** (1.0 / 3.0))
    return dv


def _distance_modulus_from_DM(DM: np.ndarray, z: np.ndarray) -> np.ndarray:
    mu = np.empty_like(z, dtype=float)
    dL = np.empty_like(z, dtype=float)
    for i in range(z.shape[0]):
        dm = float(DM[i])
        z_val = float(z[i])
        if not math.isfinite(dm) or dm <= 0.0 or z_val < -0.999999:
            mu[i] = float("inf")
            dL[i] = float("inf")
            continue
        dL[i] = dm * (1.0 + z_val)
        mu[i] = float(5.0 * (math.log10(dm * (1.0 + z_val)) + 5.0))
    if z.shape[0] >= 2:
        sort_idx = np.argsort(z)
        sorted_dL = dL[sort_idx]
        diffs = np.diff(sorted_dL)
        diffs = np.where(np.isfinite(diffs), diffs, -2.0e-8)
        if not np.all(diffs >= -1.0e-8):
            return np.full_like(mu, float("inf"))
    return mu


__all__ = ["LCDMModel"]
