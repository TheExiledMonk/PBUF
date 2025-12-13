"""Stable growth-index predictor returning masked γ(z) series and metadata."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np

from cosmos2.kernels.common.growth import solve_growth

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult

logger = logging.getLogger(__name__)

_DEFAULT_Z_MAX = 3.0
_DEFAULT_POINTS = 200
_MIN_SCALE_FACTOR = 1e-6
_MIN_VALUE = 1e-12
_MIN_VALID_F = 1e-6
_MIN_GROWTH_STEPS = 2048
_MAX_GROWTH_STEPS = 20000
_DESCRIPTION_TEXT = (
    "Growth index γ(z) is defined only where the growth rate f(z) > 0 and 0 < Ω_m(z) < 1. "
    "The module applies strict masking and stable log ratios to avoid numerical artifacts. "
    "Predictions follow the PBUF standardized format used by all other prediction modules."
)


def _build_z_grid(zmax: float, points: int) -> np.ndarray:
    if points < 2:
        raise ValueError("points must be at least 2.")
    if zmax < 0.0:
        raise ValueError("zmax must be non-negative.")
    return np.linspace(0.0, float(zmax), num=points, dtype=float)


def _build_integration_grid(a_targets: np.ndarray) -> np.ndarray:
    if a_targets.size == 0:
        raise ValueError("Need at least one scale factor target to solve growth.")
    safe_targets = np.clip(a_targets, _MIN_SCALE_FACTOR, 1.0)
    a_min = float(np.min(safe_targets))
    start = max(_MIN_SCALE_FACTOR, min(a_min * 0.1, 1e-3))
    steps = max(int(min(a_targets.size * 10, _MAX_GROWTH_STEPS)), _MIN_GROWTH_STEPS)
    return np.logspace(np.log10(start), 0.0, num=steps, dtype=float)


def _interpolate_log_space(grid: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if grid.size == 0 or targets.size == 0:
        return np.array([], dtype=float)
    log_grid = np.log(np.clip(grid, _MIN_SCALE_FACTOR, np.inf))
    clipped_targets = np.clip(targets, grid[0], grid[-1])
    return np.interp(np.log(clipped_targets), log_grid, values)


def _solve_growth_series(model: PredictionModelAdapter, z_values: np.ndarray) -> np.ndarray:
    if z_values.size == 0:
        return np.array([], dtype=float)
    a_targets = np.clip(1.0 / np.clip(1.0 + z_values, 1e-9, np.inf), _MIN_SCALE_FACTOR, 1.0)
    a_grid = _build_integration_grid(a_targets)
    H0 = float(model.parameters.get("H0", 67.4))
    if H0 <= 0.0 or not np.isfinite(H0):
        raise ValueError("Model reports an invalid H0.")
    H_vals = model.H(a_grid)
    if H_vals.shape != a_grid.shape:
        raise RuntimeError("H(a) grid size mismatch during growth solve.")
    E_vals = np.clip(np.asarray(H_vals, dtype=float) / H0, _MIN_VALUE, np.inf)
    omega_m0 = float(model.omega_m0())
    D_grid, _ = solve_growth(a_grid, E_vals, omega_m0=omega_m0)
    return _interpolate_log_space(a_grid, D_grid, a_targets)


def _compute_growth_rate(delta: np.ndarray, a_values: np.ndarray) -> np.ndarray:
    safe_delta = np.clip(delta, _MIN_VALUE, np.inf)
    ln_delta = np.log(safe_delta)
    ln_a = np.log(np.clip(a_values, _MIN_SCALE_FACTOR, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_order = 2 if a_values.size > 2 else 1
        gradient = np.gradient(ln_delta, ln_a, edge_order=edge_order)
    return np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)


def _build_meta(z_max: float, n_points: int, model_name: str) -> dict[str, object]:
    return {
        "z_max": float(z_max),
        "n_points": int(n_points),
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
        "notes": "Growth index computed with masked stability conditions.",
        "description": _DESCRIPTION_TEXT,
    }


def _error_result(message: str) -> PredictionResult:
    return PredictionResult(
        name="growth-index",
        version="1.0",
        metadata={"error": message},
        results={},
        status="error",
    )


@register_prediction
class GrowthIndexPrediction(PredictionModule):
    name = "growth-index"
    version = "1.0"
    description = "Predicts the masked growth index γ(z) with stable numerical derivatives."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmax",
            type=float,
            default=_DEFAULT_Z_MAX,
            help="Maximum redshift for the γ(z) grid (default 3.0)",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=_DEFAULT_POINTS,
            help="Number of redshift samples (default 200)",
        )
        super().register(parser)

    def run_prediction(self, model: PredictionModelAdapter, config: dict[str, object]) -> PredictionResult:
        try:
            z_max = float(config.get("zmax", _DEFAULT_Z_MAX))
            points = max(2, int(config.get("points", _DEFAULT_POINTS)))
        except (TypeError, ValueError) as exc:
            return _error_result(f"Invalid grid configuration: {exc}")

        try:
            z_grid = _build_z_grid(z_max, points)
            delta = _solve_growth_series(model, z_grid)
        except Exception as exc:  # pragma: no cover - best-effort guard
            logger.exception("Growth index solver failed: %s", exc)
            return _error_result("growth_index_solve_failure")

        a_values = 1.0 / np.clip(1.0 + z_grid, 1e-9, np.inf)
        f_series = _compute_growth_rate(delta, a_values)
        omega_series = model.Omega_m_of_z(z_grid)

        valid_delta = (delta > 0.0) & np.isfinite(delta)
        valid_omega = (omega_series > 0.0) & (omega_series < 1.0) & np.isfinite(omega_series)
        valid_f = f_series >= _MIN_VALID_F
        mask_valid = valid_delta & valid_omega & valid_f

        gamma = np.full_like(z_grid, np.nan)
        safe_indices = np.nonzero(mask_valid)[0]
        if safe_indices.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                safe_gamma = np.log(f_series[mask_valid]) / np.log(omega_series[mask_valid])
            gamma[mask_valid] = safe_gamma

        valid_points = int(mask_valid.sum())
        if valid_points < 5:
            logger.warning("Growth index mask only has %d valid points", valid_points)

        model_name = getattr(model.raw_model.__class__, "__name__", "model")
        payload_meta = _build_meta(z_max, points, model_name)

        payload = {
            "name": "growth_index",
            "z": z_grid.tolist(),
            "f": f_series.tolist(),
            "gamma": gamma.tolist(),
            "mask_valid": mask_valid.tolist(),
            "meta": payload_meta,
        }

        plot_data = {
            "z": z_grid[mask_valid].tolist(),
            "gamma": gamma[mask_valid].tolist(),
        }
        plots = []
        plots.append(
            PredictionPlot(
                name="growth_index_plot",
                data=plot_data,
                description="Growth Index predictions (growth_index_plot)",
                metadata={"xlabel": "redshift z", "ylabel": "γ(z)"},
            )
        )

        metadata: dict[str, object] = {
            "model": model_name,
            "points": len(z_grid),
            "valid_points": valid_points,
        }
        if valid_points < 5:
            metadata["warnings"] = [
                "Growth index mask has fewer than 5 valid points; treat γ(z) with care." 
            ]

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=payload,
            plots=plots,
        )
