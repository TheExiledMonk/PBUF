"""Prediction module presenting the statefinder pair (r, s)."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult

if TYPE_CHECKING:
    from ..model_api import PredictionModelAdapter

logger = logging.getLogger(__name__)

_DEFAULT_Z_MIN = 0.0
_DEFAULT_Z_MAX = 3.0
_DEFAULT_POINTS = 200
_MIN_POINTS = 3
_MIN_VALID_POINTS = 10
_TINY = 1e-8
_META_NOTES = "Statefinder diagnostics r(z), s(z) computed from E(a) and its derivatives."
_META_DESCRIPTION = (
    "Statefinder diagnostics (r, s) as functions of redshift. "
    "For ΛCDM with a cosmological constant, the statefinder pair sits at the fixed point (r, s) = (1, 0). "
    "This module computes r and s from the expansion history E(a) and its derivatives, "
    "and records their present day values for model comparison."
)


def _infer_h0(model: "PredictionModelAdapter") -> float:
    """Infer H0 by querying H(a=1.0) and verifying it is positive."""
    values = np.asarray(model.H(1.0), dtype=float)
    if values.size == 0:
        raise RuntimeError("Unable to infer H0 from the model.")
    h0 = float(values.flat[0])
    if not np.isfinite(h0) or h0 <= 0.0:
        raise RuntimeError("Model reported an invalid H0 value.")
    return h0


def _numerical_derivative(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Forward/backward/central finite differences that handle reversed grids."""
    arr = np.asarray(values, dtype=float)
    grid = np.asarray(coords, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError("Coordinate and value grids must share the same shape.")
    if arr.size <= 1:
        return np.full_like(arr, np.nan)

    reversed_grid = grid[0] > grid[-1]
    if reversed_grid:
        arr = arr[::-1]
        grid = grid[::-1]

    edge_order = 2 if arr.size >= 3 else 1
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = np.gradient(arr, grid, edge_order=edge_order)

    if reversed_grid:
        derivative = derivative[::-1]

    return derivative


@register_prediction
class StatefinderPrediction(PredictionModule):
    """Compute the statefinder diagnostics (r, s) from the normalized expansion history."""

    name = "statefinder"
    version = "1.0"
    description = "Statefinder diagnostics (r, s) derived from E(a)."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmin",
            type=float,
            default=_DEFAULT_Z_MIN,
            help=f"Minimum redshift to sample (default {_DEFAULT_Z_MIN}).",
        )
        parser.add_argument(
            "--zmax",
            type=float,
            default=_DEFAULT_Z_MAX,
            help=f"Maximum redshift to sample (default {_DEFAULT_Z_MAX}).",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=_DEFAULT_POINTS,
            help=f"Number of redshift samples (default {_DEFAULT_POINTS}).",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        z_min = float(config.get("zmin", _DEFAULT_Z_MIN))
        z_max = float(config.get("zmax", _DEFAULT_Z_MAX))
        points = int(config.get("points", _DEFAULT_POINTS))
        if z_min < 0.0:
            raise ValueError("zmin must be non-negative.")
        if z_max <= z_min:
            raise ValueError("zmax must be greater than zmin.")
        if points < _MIN_POINTS:
            raise ValueError(f"points must be at least {_MIN_POINTS}.")

        z_grid = np.linspace(z_min, z_max, points, dtype=float)
        a_grid = np.clip(1.0 / (1.0 + z_grid), 0.0, 1.0)

        h0 = _infer_h0(model)
        h_vals = np.asarray(model.H(a_grid), dtype=float)
        if h_vals.shape != a_grid.shape:
            raise RuntimeError("Model returned H(a) with unexpected shape.")
        e_vals = np.asarray(h_vals / h0, dtype=float)

        dE_da = _numerical_derivative(e_vals, a_grid)
        d2E_da2 = _numerical_derivative(dE_da, a_grid)

        mask_base = (
            np.isfinite(e_vals)
            & np.isfinite(dE_da)
            & np.isfinite(d2E_da2)
            & (e_vals > 0.0)
        )

        q_vals = np.full_like(e_vals, np.nan)
        r_vals = np.full_like(e_vals, np.nan)
        safe_mask = mask_base
        if safe_mask.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                a_safe = a_grid[safe_mask]
                dE_safe = dE_da[safe_mask]
                d2E_safe = d2E_da2[safe_mask]
                E_safe = e_vals[safe_mask]
                q_vals[safe_mask] = -a_safe * dE_safe / E_safe
                r_vals[safe_mask] = (
                    1.0
                    + 3.0 * a_safe * dE_safe / E_safe
                    + (a_safe**2) * d2E_safe / E_safe
                )

        s_vals = np.full_like(e_vals, np.nan)
        denominator = 3.0 * (q_vals - 0.5)
        mask_s = (
            mask_base
            & np.isfinite(q_vals)
            & np.isfinite(r_vals)
            & np.isfinite(denominator)
            & (np.abs(denominator) > _TINY)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            s_vals[mask_s] = (r_vals[mask_s] - 1.0) / denominator[mask_s]

        mask_valid = mask_s
        valid_count = int(np.count_nonzero(mask_valid))
        if valid_count < _MIN_VALID_POINTS:
            logger.warning(
                "Statefinder mask valid points = %d (< %d); results may be noisy.",
                valid_count,
                _MIN_VALID_POINTS,
            )

        r0: float | None = None
        s0: float | None = None
        if mask_valid.any():
            valid_indices = np.where(mask_valid)[0]
            closest = int(valid_indices[np.argmin(np.abs(z_grid[valid_indices] - 0.0))])
            candidate_r = r_vals[closest]
            candidate_s = s_vals[closest]
            if np.isfinite(candidate_r):
                r0 = float(candidate_r)
            if np.isfinite(candidate_s):
                s0 = float(candidate_s)

        prediction = {
            "name": self.name,
            "z": z_grid.tolist(),
            "a": a_grid.tolist(),
            "r": r_vals.tolist(),
            "s": s_vals.tolist(),
            "q": q_vals.tolist(),
            "mask_valid": mask_valid.tolist(),
            "summary": {"r0": r0, "s0": s0},
            "meta": {
                "z_min": float(z_min),
                "z_max": float(z_max),
                "n_points": len(z_grid),
                "model_name": getattr(model.raw_model.__class__, "__name__", "model"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": self.version,
                "notes": _META_NOTES,
                "description": _META_DESCRIPTION,
            },
        }

        plot_data: dict[str, list[float]] = {}
        plots: list[PredictionPlot] = []
        if mask_valid.any():
            s_valid = s_vals[mask_valid]
            r_valid = r_vals[mask_valid]
            z_valid = z_grid[mask_valid]
            if s_valid.size and r_valid.size:
                plot_data = {
                    "x": s_valid.tolist(),
                    "r": r_valid.tolist(),
                    "z_trace": z_valid.tolist(),
                }
                plots.append(
                    PredictionPlot(
                        name="statefinder_rs",
                        description="Statefinder trajectory in the (r, s) plane (statefinder prediction)",
                        data=plot_data,
                        metadata={"xlabel": "s", "ylabel": "r"},
                    )
                )

        metadata: dict[str, object] = {
            "model": prediction["meta"]["model_name"],
            "points": len(z_grid),
            "valid_points": valid_count,
        }
        if valid_count < _MIN_VALID_POINTS:
            metadata["warnings"] = [
                f"Only {valid_count} valid samples (need ≥ {_MIN_VALID_POINTS}); r(z)/s(z) may be unreliable."
            ]

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=prediction,
            plots=plots,
        )
