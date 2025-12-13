"""Prediction module producing the jerk parameter j(z) from the background expansion."""

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
_NOTES = "Jerk parameter j(z) computed from E(a) with second-order finite differences."
_DESCRIPTION = (
    "Jerk parameter prediction j(z). "
    "The jerk j ≡ d^3a/(a H^3 dt^3) equals 1 for ΛCDM with a cosmological constant. "
    "This module computes j(z) from the normalized expansion rate E(a) using second derivatives, "
    "and summarizes j at z=0 and its mean value over 0 <= z <= 1."
)


def _infer_h0(model: "PredictionModelAdapter") -> float:
    """Query the model for H(a=1) to normalize the expansion history."""
    h0_values = np.asarray(model.H(1.0), dtype=float)
    if h0_values.size == 0:
        raise RuntimeError("Unable to infer H0 from the model.")
    h0 = float(h0_values.flat[0])
    if not np.isfinite(h0) or h0 <= 0.0:
        raise RuntimeError("Model returned invalid H0.")
    return h0


def _numerical_derivative(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Finite differences like numpy.gradient, safe for decreasing grids."""
    arr = np.asarray(values, dtype=float)
    grid = np.asarray(coords, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError("Coordinate grid and values must share the same shape.")
    if arr.size <= 1:
        return np.full_like(arr, np.nan)

    reversed_order = grid[0] > grid[-1]
    if reversed_order:
        arr = arr[::-1]
        grid = grid[::-1]

    edge_order = 2 if arr.size >= 3 else 1
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = np.gradient(arr, grid, edge_order=edge_order)

    if reversed_order:
        derivative = derivative[::-1]

    return derivative


def _build_plot_data(z_valid: np.ndarray, j_valid: np.ndarray, j0_value: float | None) -> dict[str, list[float]]:
    """Prepare plot series for valid jerk samples."""
    data: dict[str, list[float]] = {
        "z": z_valid.tolist(),
        "j": j_valid.tolist(),
        "j_ref": np.ones_like(j_valid, dtype=float).tolist(),
    }
    marker = np.full_like(j_valid, np.nan)
    if j0_value is not None and z_valid.size > 0:
        idx = int(np.nanargmin(np.abs(z_valid - 0.0)))
        if np.isfinite(j_valid[idx]):
            marker[idx] = float(j_valid[idx])
    data["j0_marker"] = marker.tolist()
    return data


@register_prediction
class JerkPrediction(PredictionModule):
    """Compute the jerk parameter j(z) using the normalized expansion history."""

    name = "jerk"
    version = "1.0"
    description = "Predicts the jerk parameter j(z) derived from the background E(a)."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmin",
            type=float,
            default=_DEFAULT_Z_MIN,
            help=f"Minimum redshift (default {_DEFAULT_Z_MIN}).",
        )
        parser.add_argument(
            "--zmax",
            type=float,
            default=_DEFAULT_Z_MAX,
            help=f"Maximum redshift (default {_DEFAULT_Z_MAX}).",
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
        E_vals = np.asarray(h_vals / h0, dtype=float)

        dE_da = _numerical_derivative(E_vals, a_grid)
        d2E_da2 = _numerical_derivative(dE_da, a_grid)

        mask_valid = (
            np.isfinite(E_vals)
            & np.isfinite(dE_da)
            & np.isfinite(d2E_da2)
            & (E_vals > 0.0)
        )

        j_vals = np.full_like(E_vals, np.nan)
        safe = mask_valid
        if safe.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                a_safe = a_grid[safe]
                dE_safe = dE_da[safe]
                d2E_safe = d2E_da2[safe]
                E_safe = E_vals[safe]
                j_vals[safe] = (
                    1.0
                    + 3.0 * a_safe * dE_safe / E_safe
                    + (a_safe**2) * d2E_safe / E_safe
                )

        valid_count = int(np.count_nonzero(mask_valid))
        if valid_count < _MIN_VALID_POINTS:
            logger.warning(
                "Jerk prediction only has %d valid samples (< %d); results may be noisy.",
                valid_count,
                _MIN_VALID_POINTS,
            )

        j0: float | None = None
        if valid_count > 0:
            z_valid = z_grid[mask_valid]
            j_valid = j_vals[mask_valid]
            idx0 = int(np.argmin(np.abs(z_valid - 0.0)))
            candidate = j_valid[idx0]
            if np.isfinite(candidate):
                j0 = float(candidate)

        mask_range = mask_valid & (z_grid >= 0.0) & (z_grid <= 1.0)
        if np.count_nonzero(mask_range) > 0:
            j_mean_0_1 = float(np.nanmean(j_vals[mask_range]))
        else:
            j_mean_0_1 = None

        prediction = {
            "name": self.name,
            "z": z_grid.tolist(),
            "a": a_grid.tolist(),
            "j": j_vals.tolist(),
            "mask_valid": mask_valid.tolist(),
            "summary": {"j0": j0, "j_mean_0_1": j_mean_0_1},
            "meta": {
                "z_min": float(z_min),
                "z_max": float(z_max),
                "n_points": len(z_grid),
                "model_name": getattr(model.raw_model.__class__, "__name__", "model"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": self.version,
                "notes": _NOTES,
                "description": _DESCRIPTION,
            },
        }

        z_plot = np.asarray(prediction["z"], dtype=float)
        j_plot = np.asarray(prediction["j"], dtype=float)
        valid_plot_mask = np.asarray(prediction["mask_valid"], dtype=bool)
        z_valid = z_plot[valid_plot_mask]
        j_valid = j_plot[valid_plot_mask]

        plots = []
        if z_valid.size > 0 and np.any(np.isfinite(j_valid)):
            plot_data = _build_plot_data(z_valid, j_valid, j0)
            plots.append(
                PredictionPlot(
                    name="jerk_vs_z",
                    description="Jerk parameter j(z) (jerk prediction)",
                    data=plot_data,
                    metadata={"xlabel": "redshift z", "ylabel": "j(z)"},
                )
            )

        metadata: dict[str, object] = {
            "model": prediction["meta"]["model_name"],
            "points": len(z_grid),
            "valid_points": valid_count,
        }
        if valid_count < _MIN_VALID_POINTS:
            metadata["warnings"] = [
                f"Only {valid_count} reliable samples (need ≥ {_MIN_VALID_POINTS})."
            ]

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=prediction,
            plots=plots,
        )
