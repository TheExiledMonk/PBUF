"""KiDS-style angular scale cuts for ξ± data vectors and covariances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


def _arcmin_to_rad(theta_arcmin: float | None) -> float | None:
    if theta_arcmin is None:
        return None
    return math.radians(float(theta_arcmin) / 60.0)


@dataclass(frozen=True)
class ScaleCutMask:
    xi_plus: np.ndarray
    xi_minus: np.ndarray
    combined: np.ndarray

Pair = Tuple[int, int]
CutTuple = Tuple[Tuple[float | None, float | None], Tuple[float | None, float | None]]


def kids_default_scale_cuts(
    n_bins: int,
    *,
    xi_plus_min_arcmin: float = 0.5,
    xi_minus_min_arcmin: float = 4.2,
    xi_plus_max_arcmin: float | None = 300.0,
    xi_minus_max_arcmin: float | None = 300.0,
    xi_minus_min_overrides_arcmin: Dict[Pair, float] | None = None,
    use_official_kids_minus: bool = True,
) -> Dict[Pair, CutTuple]:
    """
    Build a per-pair θ cut dictionary using a uniform KiDS-style minimum (can be overridden per pair).
    """

    theta_min_plus = _arcmin_to_rad(xi_plus_min_arcmin)
    theta_max_plus = _arcmin_to_rad(xi_plus_max_arcmin) if xi_plus_max_arcmin is not None else None
    theta_min_minus = _arcmin_to_rad(xi_minus_min_arcmin)
    theta_max_minus = _arcmin_to_rad(xi_minus_max_arcmin) if xi_minus_max_arcmin is not None else None
    overrides = dict(xi_minus_min_overrides_arcmin or {})
    if use_official_kids_minus and n_bins >= 5:
        # Official KiDS-1000 xi- theta_min (arcmin) per tomographic pair (i<=j).
        official = {
            (0, 0): 4.2,
            (0, 1): 4.2,
            (0, 2): 4.2,
            (0, 3): 4.2,
            (0, 4): 5.6,
            (1, 1): 4.2,
            (1, 2): 4.2,
            (1, 3): 5.6,
            (1, 4): 5.6,
            (2, 2): 4.2,
            (2, 3): 5.6,
            (2, 4): 5.6,
            (3, 3): 5.6,
            (3, 4): 5.6,
            (4, 4): 6.0,
        }
        overrides.update(official)
    cuts: Dict[Pair, CutTuple] = {}
    for i in range(n_bins):
        for j in range(i, n_bins):
            key = (i, j)
            theta_min_minus_pair = theta_min_minus
            if key in overrides:
                theta_min_minus_pair = _arcmin_to_rad(overrides[key])
            cuts[key] = ((theta_min_plus, theta_max_plus), (theta_min_minus_pair, theta_max_minus))
    return cuts


def _parse_cut_value(
    value: Tuple[float | None, ...],
) -> tuple[Tuple[float | None, float | None], Tuple[float | None, float | None]]:
    if len(value) == 2:
        plus = (value[0], value[1])
        return plus, plus
    if len(value) == 4:
        plus = (value[0], value[1])
        minus = (value[2], value[3])
        return plus, minus
    raise ValueError("Scale-cut tuple must have length 2 (shared) or 4 (plus/minus).")


def build_custom_scale_cuts(
    n_bins: int,
    table: Dict[Pair, Tuple[float | None, ...]],
    *,
    default: Dict[Pair, CutTuple] | None = None,
) -> Dict[Pair, CutTuple]:
    """
    Build a per-pair cut dictionary from a user-supplied table keyed by (i,j) pairs.

    Each value can be (theta_min, theta_max) shared or (theta_min_plus, theta_max_plus, theta_min_minus, theta_max_minus).
    Theta units are radians; caller should convert arcmin/deg upstream.
    """

    cuts: Dict[Pair, CutTuple] = dict(default or {})
    for key, value in table.items():
        if not isinstance(key, tuple):
            continue
        if len(key) != 2:
            continue
        i, j = int(key[0]), int(key[1])
        if i > j or i < 0 or j >= n_bins:
            continue
        plus_cut, minus_cut = _parse_cut_value(value)
        cuts[(i, j)] = (plus_cut, minus_cut)
    return cuts


def build_scale_cut_mask(
    theta_bins: Iterable[float],
    tomo_pairs: Iterable[Tuple[int, int]],
    cuts: Dict[Pair, CutTuple]
    ) -> ScaleCutMask:
    theta = np.asarray(theta_bins, dtype=float)
    n_theta = theta.size
    pairs = list(tomo_pairs)
    mask_plus = np.ones(len(pairs) * n_theta, dtype=bool)
    mask_minus = np.ones_like(mask_plus)
    for pair_idx, pair in enumerate(pairs):
        raw = cuts.get(tuple(pair))
        if raw is None:
            raw = ((None, None), (None, None))
        if isinstance(raw, tuple) and len(raw) in {2, 4} and not isinstance(raw[0], tuple):
            plus_cut, minus_cut = _parse_cut_value(raw)  # type: ignore[arg-type]
        else:
            plus_cut, minus_cut = raw  # type: ignore[assignment]
        theta_min_plus, theta_max_plus = plus_cut
        theta_min_minus, theta_max_minus = minus_cut
        for t_idx, th in enumerate(theta):
            flag_plus = True
            if theta_min_plus is not None and th < theta_min_plus:
                flag_plus = False
            if theta_max_plus is not None and th > theta_max_plus:
                flag_plus = False
            flag_minus = True
            if theta_min_minus is not None and th < theta_min_minus:
                flag_minus = False
            if theta_max_minus is not None and th > theta_max_minus:
                flag_minus = False
            mask_plus[pair_idx * n_theta + t_idx] = flag_plus
            mask_minus[pair_idx * n_theta + t_idx] = flag_minus
    combined = np.concatenate([mask_plus, mask_minus], axis=0)
    return ScaleCutMask(xi_plus=mask_plus, xi_minus=mask_minus, combined=combined)


def apply_scale_cuts(data_vector: np.ndarray, covariance: np.ndarray, mask: ScaleCutMask) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a boolean mask to the data vector and covariance.
    """

    dv = np.asarray(data_vector, dtype=float).reshape(-1)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square for scale cuts.")
    if cov.shape[0] != dv.shape[0]:
        raise ValueError("Covariance dimension must match data vector length.")
    keep = np.asarray(mask.combined, dtype=bool)
    dv_cut = dv[keep]
    cov_cut = cov[np.ix_(keep, keep)]
    return dv_cut, cov_cut


__all__ = [
    "ScaleCutMask",
    "kids_default_scale_cuts",
    "build_scale_cut_mask",
    "apply_scale_cuts",
]
