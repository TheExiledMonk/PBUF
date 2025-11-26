"""Probe Phase-6a curvature metrics for LCDM and PBUF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from pprint import pformat
import sys
from typing import Callable, Mapping, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cosmos2.models.lcdm import LCDMModel


@dataclass(frozen=True)
class Phase6aConfig:
    a_curv_min: float = 1e-6
    a_curv_max: float = 1.0
    n_curv_points: int = 200
    curv_factor: float = 1.2
    curv_fraction_tolerance: float = 0.05
    curv_eps: float = 1e-30
    baseline_max_ratio: float = 2.0


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
    payload = json.loads(path.read_text())
    return Phase6aConfig(
        a_curv_min=float(payload.get("a_curv_min", 1e-6)),
        a_curv_max=float(payload.get("a_curv_max", 1.0)),
        n_curv_points=int(payload.get("n_curv_points", 200)),
        curv_factor=float(payload.get("curv_factor", 1.2)),
        curv_fraction_tolerance=float(payload.get("curv_fraction_tolerance", 0.05)),
        curv_eps=float(payload.get("curv_eps", 1e-30)),
        baseline_max_ratio=float(payload.get("baseline_max_ratio", 2.0)),
    )


def _make_curvature_grid(config: Phase6aConfig) -> np.ndarray:
    return np.exp(np.linspace(np.log(config.a_curv_min), np.log(config.a_curv_max), config.n_curv_points))


def compute_curvature_stats(H_func: Callable[[float], float], config: Phase6aConfig) -> CurvatureStats:
    a_grid = _make_curvature_grid(config)
    H_values = np.array([H_func(a) for a in a_grid], dtype=float)

    if a_grid.size < 3:
        raise ValueError("Phase-6a curvature grid must contain at least three points.")

    a_centers = a_grid[1:-1]

    Hp_prev = (H_values[1:-1] - H_values[:-2]) / (a_grid[1:-1] - a_grid[:-2])
    Hp_next = (H_values[2:] - H_values[1:-1]) / (a_grid[2:] - a_grid[1:-1])
    Hp = (H_values[2:] - H_values[:-2]) / (a_grid[2:] - a_grid[:-2])
    Hpp = (Hp_next - Hp_prev) / (a_grid[2:] - a_grid[:-2])

    ratio = np.zeros_like(Hpp)
    valid_mask = np.abs(Hp) > config.curv_eps
    ratio[valid_mask] = Hpp[valid_mask] / Hp[valid_mask]

    finite_mask = np.isfinite(ratio) & valid_mask
    if np.any(finite_mask):
        max_abs_ratio = float(np.nanmax(np.abs(ratio[finite_mask])))
        min_abs_ratio = float(np.nanmin(np.abs(ratio[finite_mask])))
    else:
        max_abs_ratio = 0.0
        min_abs_ratio = 0.0

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

PRESETS: Mapping[str, Mapping[str, Mapping[str, float]]] = {
    "lcdm": {
        "planck": {
            "H0": 67.36,
            "Omega_m0": 0.315,
            "Omega_b0": 0.0493,
            "Omega_r0": 9.0e-5,
            "Omega_k0": 0.0,
        }
    },
}


def compute_bad_fraction(
    stats: CurvatureStats,
    config: Phase6aConfig,
) -> tuple[float, float, int]:
    allowed = config.baseline_max_ratio * config.curv_factor
    valid = stats.valid_mask & np.isfinite(stats.ratio)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return 0.0, allowed, valid_count
    bad = np.abs(stats.ratio[valid]) > allowed
    bad_count = int(np.count_nonzero(bad))
    return float(bad_count) / float(valid_count), allowed, valid_count


def find_extrema_indices(stats: CurvatureStats) -> tuple[int | None, int | None]:
    valid = stats.valid_mask & np.isfinite(stats.ratio)
    indices = np.nonzero(valid)[0]
    if indices.size == 0:
        return None, None
    values = np.abs(stats.ratio[indices])
    max_idx = int(indices[int(np.nanargmax(values))])
    min_idx = int(indices[int(np.nanargmin(values))])
    return max_idx, min_idx


def format_table_row(stats: CurvatureStats, idx: int) -> str:
    a = stats.a_centers[idx]
    z = 1.0 / a - 1.0
    H_val = float(np.interp(a, stats.a_grid, stats.H_values))
    H_prime = float(stats.H_prime[idx])
    ratio = float(stats.ratio[idx])
    return f"{a:10.3e}   {z:9.2e}   {H_val:9.3f}   {H_prime:9.3f}   {ratio:9.3f}"


def print_peak_sample(stats: CurvatureStats, center_idx: int) -> None:
    delimiter = "-" * 76
    print("Sample around max:")
    print("    a          z         H(a)       H'(a)    H''(a)/H'(a)")
    for offset in (-1, 0, 1):
        row_idx = center_idx + offset
        if row_idx < 0 or row_idx >= stats.ratio.size:
            continue
        print(format_table_row(stats, row_idx))
    print(delimiter)


def print_phase6a_summary(
    *,
    model_name: str,
    preset: str | None,
    params: Mapping[str, float],
    config: Phase6aConfig,
    stats: CurvatureStats,
    ok: bool,
    reason: str | None,
) -> None:
    header = "=" * 76
    divider = "-" * 76
    bad_fraction, allowed, valid_count = compute_bad_fraction(stats, config)
    max_idx, min_idx = find_extrema_indices(stats)

    status = "PASS" if ok else "FAIL"
    grid_info = f"{config.a_curv_min:.1e} ... {config.a_curv_max:.1e}"

    print(header)
    print("Phase-6a curvature probe")
    print(f"Model: {model_name}")
    print(f"Preset: {preset or 'custom'}")
    print("Parameters:")
    print(pformat(dict(params), indent=2))
    print(divider)
    print(f"a_curv range: [{grid_info}]  (N = {config.n_curv_points})")
    print(f"baseline_max_ratio = {config.baseline_max_ratio:.3g}")
    print(f"curv_factor = {config.curv_factor:.3g}  allowed ≤ {allowed:.3g}")
    print(
        f"bad fraction = {bad_fraction:.1%} "
        f"(tol {config.curv_fraction_tolerance:.1%}, valid entries = {valid_count})"
    )
    print(f"Result: {status}")
    if reason:
        print(f"Reason: {reason}")
    if max_idx is not None:
        a = stats.a_centers[max_idx]
        z = 1.0 / a - 1.0
        max_ratio = abs(stats.ratio[max_idx])
        print(f"max |H''/H'| = {max_ratio:.3g} at a = {a:.3e} (z ≈ {z:.2e})")
    if min_idx is not None:
        a_min = stats.a_centers[min_idx]
        z_min = 1.0 / a_min - 1.0
        min_ratio = abs(stats.ratio[min_idx])
        print(f"min |H''/H'| = {min_ratio:.3g} at a = {a_min:.3e} (z ≈ {z_min:.2e})")
    print(divider)
    if max_idx is not None:
        print_peak_sample(stats, max_idx)


def print_legacy_summary(stats: CurvatureStats, grid_points: int) -> None:
    header = "=" * 76
    divider = "-" * 76
    max_idx, min_idx = find_extrema_indices(stats)
    grid_info = f"{stats.a_grid[0]:.1e} ... {stats.a_grid[-1]:.1e}"

    print(header)
    print("Legacy full-range log grid (a≈1e-9 … 1.0)")
    print(f"a-grid: [{grid_info}]  (N = {grid_points})")
    if max_idx is not None:
        a = stats.a_centers[max_idx]
        z = 1.0 / a - 1.0
        print(f"max |H''/H'| = {abs(stats.ratio[max_idx]):.3g} at a = {a:.3e} (z ≈ {z:.2e})")
    if min_idx is not None:
        a_min = stats.a_centers[min_idx]
        z_min = 1.0 / a_min - 1.0
        print(f"min |H''/H'| = {abs(stats.ratio[min_idx]):.3g} at a = {a_min:.3e} (z ≈ {z_min:.2e})")
    print(divider)
    if max_idx is not None:
        print_peak_sample(stats, max_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Phase-6a curvature metrics")
    parser.add_argument("--model", required=True, choices=["lcdm"])
    parser.add_argument("--preset", help="Preset name to start from")
    parser.add_argument("--legacy-grid", action="store_true", help="Also inspect the legacy a∈[1e-9,1] grid")
    parser.add_argument("--grid-points", type=int, default=200)
    parser.add_argument("--H0", type=float)
    parser.add_argument("--Omega_m0", type=float)
    parser.add_argument("--Omega_b0", type=float)
    parser.add_argument("--Omega_r0", type=float)
    parser.add_argument("--Omega_k0", type=float)
    return parser.parse_args()


def build_parameters(args: argparse.Namespace) -> Mapping[str, float]:
    base: dict[str, float] = {}
    model_presets = PRESETS.get(args.model, {})
    if args.preset:
        preset = model_presets.get(args.preset)
        if preset is None:
            raise ValueError(f"Unknown preset '{args.preset}' for model '{args.model}'.")
        base.update(preset)
    for key in ("H0", "Omega_m0", "Omega_b0", "Omega_r0", "Omega_k0"):
        value = getattr(args, key)
        if value is not None:
            base[key] = value
    required = {"H0", "Omega_m0", "Omega_r0", "Omega_k0", "Omega_b0"}
    missing = required - base.keys()
    if missing:
        raise ValueError(f"Missing parameters: {sorted(missing)}")
    return base


def build_H_func(args: argparse.Namespace, params: Mapping[str, float]) -> Callable[[float], float]:
    if args.model != "lcdm":
        raise ValueError("PBUF model support has been removed during the rebuild; only lcdm is available.")
    model = LCDMModel(**params)
    return lambda a: float(model.Hubble(1.0 / a - 1.0))


def main() -> None:
    args = parse_args()
    params = build_parameters(args)
    config = load_phase6a_config(args.model)
    H_func = build_H_func(args, params)
    ok, reason, stats = curvature_check(H_func, config)
    print_phase6a_summary(
        model_name=args.model,
        preset=args.preset,
        params=params,
        config=config,
        stats=stats,
        ok=ok,
        reason=reason,
    )

    if args.legacy_grid:
        legacy_points = max(args.grid_points, 3)
        legacy_config = Phase6aConfig(
            a_curv_min=1e-9,
            a_curv_max=1.0,
            n_curv_points=legacy_points,
            curv_factor=config.curv_factor,
            curv_fraction_tolerance=config.curv_fraction_tolerance,
            curv_eps=config.curv_eps,
            baseline_max_ratio=config.baseline_max_ratio,
        )
        legacy_stats = compute_curvature_stats(H_func, legacy_config)
        print_legacy_summary(legacy_stats, legacy_points)


if __name__ == "__main__":
    main()
