"""Shared Phase-6a helpers (curvature probes, configs, and checks)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from cosmos2.kernels.pbuf_phase6 import curvature_stats_kernel

@dataclass(frozen=True)
class Phase6aConfig:
    a_curv_min: float = 1e-6
    a_curv_max: float = 1.0
    n_curv_points: int = 200
    curv_factor: float = 1.2
    curv_fraction_tolerance: float = 0.05
    curv_eps: float = 1e-30
    baseline_max_ratio: float = 2.0
    a_min_global: float = 1e-8
    a_min_curv: float = 1e-3
    n_global_points: int = 200
    n_early_points: int = 100
    n_elastic_points: int = 200
    eps_early_max: float = 1e-3
    early_monotonic_tol: float = 1e-8
    df_max: float = 0.5
    f_el_min: float = 0.01
    f_el_max: float = 0.3

    @property
    def blocked_grid(self) -> np.ndarray:
        return np.linspace(np.log(self.a_curv_min), np.log(self.a_curv_max), self.n_curv_points)


@dataclass
class CurvatureStats:
    a_grid: np.ndarray
    a_centers: np.ndarray
    H_values: np.ndarray
    H_prime: np.ndarray
    H_double_prime: np.ndarray
    ratio: np.ndarray
    valid_mask: np.ndarray
    max_abs_ratio: float
    min_abs_ratio: float


def load_phase6a_config(model_name: str) -> Phase6aConfig:
    path = Path("configs/phase6a") / f"{model_name.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase-6a config not found at {path}")
    payload = json.loads(path.read_text())
    return Phase6aConfig(
        a_curv_min=float(payload.get("a_curv_min", 1e-6)),
        a_curv_max=float(payload.get("a_curv_max", 1.0)),
        n_curv_points=int(payload.get("n_curv_points", 200)),
        curv_factor=float(payload.get("curv_factor", 1.2)),
        curv_fraction_tolerance=float(payload.get("curv_fraction_tolerance", 0.05)),
        curv_eps=float(payload.get("curv_eps", 1e-30)),
        baseline_max_ratio=float(payload.get("baseline_max_ratio", 2.0)),
        a_min_global=float(payload.get("a_min_global", 1e-8)),
        a_min_curv=float(payload.get("a_min_curv", 1e-3)),
        n_global_points=int(payload.get("n_global_points", 200)),
        n_early_points=int(payload.get("n_early_points", 100)),
        n_elastic_points=int(payload.get("n_elastic_points", 200)),
        eps_early_max=float(payload.get("eps_early_max", 1e-3)),
        early_monotonic_tol=float(payload.get("early_monotonic_tol", 1e-8)),
        df_max=float(payload.get("df_max", 0.5)),
        f_el_min=float(payload.get("f_el_min", 0.01)),
        f_el_max=float(payload.get("f_el_max", 0.3)),
    )


def make_curvature_grid(config: Phase6aConfig) -> np.ndarray:
    return np.exp(np.linspace(np.log(config.a_curv_min), np.log(config.a_curv_max), config.n_curv_points))


def compute_H_grid(
    H_func: Callable[[float], float],
    config: Phase6aConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the scale-factor grid and corresponding H(a) values for a Phase-6a config."""

    a_grid = make_curvature_grid(config)
    H_values = np.array([H_func(a) for a in a_grid], dtype=float)
    return a_grid, H_values


def compute_curvature_stats(
    H_func: Callable[[float], float],
    config: Phase6aConfig,
) -> CurvatureStats:
    a_grid, H_values = compute_H_grid(H_func, config)

    if a_grid.size < 3:
        raise ValueError("Phase-6a curvature grid must contain at least three points.")

    a_centers = a_grid[1:-1]

    Hp, Hpp, ratio, valid_mask, max_abs_ratio, min_abs_ratio = curvature_stats_kernel(
        a_grid, H_values, config.curv_eps
    )

    return CurvatureStats(
        a_grid=a_grid,
        a_centers=a_centers,
        H_values=H_values,
        H_prime=Hp,
        H_double_prime=Hpp,
        ratio=ratio,
        valid_mask=valid_mask,
        max_abs_ratio=max_abs_ratio,
        min_abs_ratio=min_abs_ratio,
    )


def curvature_check(
    H_func: Callable[[float], float],
    config: Phase6aConfig,
    *,
    require_fraction: bool = True,
) -> Tuple[bool, str | None, CurvatureStats]:
    stats = compute_curvature_stats(H_func, config)
    allowed = config.baseline_max_ratio * config.curv_factor
    valid = stats.valid_mask & np.isfinite(stats.ratio)
    if not np.any(valid):
        return True, None, stats
    bad = np.abs(stats.ratio[valid]) > allowed
    bad_fraction = float(np.count_nonzero(bad)) / float(np.count_nonzero(valid))
    if require_fraction and bad_fraction > config.curv_fraction_tolerance:
        reason = (
            f"|H''/H'| > {allowed:.3g} on {bad_fraction:.1%} of the grid "
            f"(max={stats.max_abs_ratio:.3f})"
        )
        return False, reason, stats
    return True, None, stats


curvature_gate = curvature_check


__all__ = [
    "Phase6aConfig",
    "CurvatureStats",
    "load_phase6a_config",
    "make_curvature_grid",
    "compute_H_grid",
    "compute_curvature_stats",
    "curvature_check",
    "curvature_gate",
]
