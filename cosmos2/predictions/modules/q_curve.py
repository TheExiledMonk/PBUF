"""Deceleration parameter q(z) prediction computed from E(a)."""

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
_META_DESCRIPTION = (
    "Deceleration parameter prediction q(z). "
    "Computed from the normalized expansion rate E(a) using finite differences. "
    "The module also estimates the redshift where q crosses zero, which marks the onset of cosmic acceleration."
)
_META_NOTES = "Deceleration parameter q(z) computed from E(a) with finite differences."


def _numerical_derivative(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Finite difference derivative mimicking numpy.gradient behavior."""
    arr = np.asarray(values, dtype=float)
    grid = np.asarray(coords, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError("scale factor grid and E(a) values must share the same shape.")
    if arr.size <= 1:
        return np.full_like(arr, np.nan)
    edge_order = 2 if arr.size >= 3 else 1
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient = np.gradient(arr, grid, edge_order=edge_order)
    return gradient


def _find_acceleration_redshift(z: np.ndarray, q: np.ndarray) -> float | None:
    """Return the redshift where q crosses zero when scanning from high z towards z=0."""
    if z.size < 2:
        return None
    z_rev = z[::-1]
    q_rev = q[::-1]
    for idx in range(1, len(q_rev)):
        q_lo = q_rev[idx - 1]
        q_hi = q_rev[idx]
        if q_lo > 0.0 and q_hi <= 0.0:
            if q_hi != q_lo:
                frac = -q_lo / (q_hi - q_lo)
                return float(z_rev[idx - 1] + frac * (z_rev[idx] - z_rev[idx - 1]))
            return float(z_rev[idx])
    return None


def _build_plot_data(
    z_valid: np.ndarray, q_valid: np.ndarray, z_acc: float | None
) -> dict[str, list[float]]:
    """Prepare a plot-friendly dictionary with q(z), q=0 line, and acceleration marker."""
    data: dict[str, list[float]] = {}
    data["z"] = z_valid.tolist()
    data["q"] = q_valid.tolist()
    data["q_zero"] = np.zeros_like(z_valid, dtype=float).tolist()
    marker = np.full(z_valid.shape, np.nan, dtype=float)
    if z_acc is not None and z_valid.size > 0:
        idx = int(np.nanargmin(np.abs(z_valid - z_acc)))
        marker[idx] = 0.0
    data["q_acc_marker"] = marker.tolist()
    return data


def _infer_h0(model: "PredictionModelAdapter") -> float:
    """Infer H0 by querying H(a=1); ensures positivity for normalization."""
    h0_array = np.asarray(model.H(1.0), dtype=float)
    if h0_array.size == 0:
        raise ValueError("Unable to infer H0 from the model.")
    h0 = float(h0_array.flat[0])
    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError("Model reported an invalid H0 value.")
    return h0


@register_prediction
class QCurvePrediction(PredictionModule):
    """Prediction module that samples q(z) and locates the acceleration onset."""

    name = "q-curve"
    version = "1.0"
    description = "Deceleration parameter q(z) derived from the expansion history."

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

        a_for_derivative = a_grid[::-1]
        e_for_derivative = e_vals[::-1]
        reversed_derivative = _numerical_derivative(e_for_derivative, a_for_derivative)
        dE_da = reversed_derivative[::-1]
        mask_valid = np.isfinite(e_vals) & np.isfinite(dE_da) & (e_vals > 0.0)

        q_grid = np.full_like(e_vals, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_a = a_grid[mask_valid]
            safe_dE = dE_da[mask_valid]
            safe_E = e_vals[mask_valid]
            q_grid[mask_valid] = -1.0 - (safe_a * safe_dE / safe_E)

        z_valid = z_grid[mask_valid]
        q_valid = q_grid[mask_valid]
        valid_points = int(mask_valid.sum())
        z_acc = None
        if valid_points >= _MIN_VALID_POINTS:
            z_acc = _find_acceleration_redshift(z_valid, q_valid)
        else:
            logger.warning(
                "q-curve mask has only %d valid points; need ≥ %d for reliable crossing.",
                valid_points,
                _MIN_VALID_POINTS,
            )

        prediction = {
            "name": self.name,
            "z": z_grid.tolist(),
            "a": a_grid.tolist(),
            "q": q_grid.tolist(),
            "mask_valid": mask_valid.tolist(),
            "summary": {"z_acc": float(z_acc) if z_acc is not None else None},
            "meta": {
                "z_min": float(z_min),
                "z_max": float(z_max),
                "n_points": len(z_grid),
                "model_name": getattr(model.raw_model.__class__, "__name__", "model"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "notes": _META_NOTES,
                "description": _META_DESCRIPTION,
            },
        }

        plot_data = _build_plot_data(z_valid, q_valid, z_acc)
        plots = [
            PredictionPlot(
                name="q_curve_plot",
                description="Deceleration parameter q(z) (q-curve prediction)",
                data=plot_data,
                metadata={"xlabel": "redshift z", "ylabel": "q(z)"},
            )
        ]

        model_name = prediction["meta"]["model_name"]
        metadata: dict[str, object] = {
            "model": model_name,
            "points": len(z_grid),
            "valid_points": valid_points,
        }
        if valid_points < _MIN_VALID_POINTS:
            metadata["warnings"] = [
                f"Only {valid_points} valid samples (need ≥ {_MIN_VALID_POINTS}); q(z) may be unreliable."
            ]

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=prediction,
            plots=plots,
        )
