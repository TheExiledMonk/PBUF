"""Predict the redshift-space distortion observable fσ₈(z)."""

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

_DEFAULT_Z_MIN = 0.0
_DEFAULT_Z_MAX = 2.0
_DEFAULT_POINTS = 200
_MIN_VALID_POINTS = 10
_DESCRIPTION = (
    "Redshift-space distortion prediction fσ8(z). "
    "Computed from the linear growth factor D(a), normalized to D(z=0)=1, and the present-day σ8_0. "
    "The module provides f(z), σ8(z), and fσ8(z) on a redshift grid, along with key summary values at z=0, 0.5, and 1."
)
_NOTES = "fσ8(z) computed from normalized growth D(a) and σ8_0."


def _build_z_grid(z_min: float, z_max: float, points: int) -> np.ndarray:
    if z_min < 0.0:
        raise ValueError("zmin must be non-negative.")
    if z_max <= z_min:
        raise ValueError("zmax must be greater than zmin.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    return np.linspace(float(z_min), float(z_max), num=points, dtype=float)


def _numerical_derivative(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    derivative = np.full_like(values, np.nan)
    mask = np.isfinite(values) & np.isfinite(coords)
    if mask.sum() < 2:
        return derivative
    sorted_indices = np.where(mask)[0]
    if sorted_indices.size == 0:
        return derivative
    # split into contiguous segments to avoid NaN leakage
    segments = np.split(sorted_indices, np.where(np.diff(sorted_indices) != 1)[0] + 1)
    for segment in segments:
        if segment.size < 2:
            continue
        y = values[segment]
        x = coords[segment]
        edge_order = 2 if segment.size > 2 else 1
        with np.errstate(divide="ignore", invalid="ignore"):
            gradient = np.gradient(y, x, edge_order=edge_order)
        derivative[segment] = gradient
    return derivative


def _error_result(message: str) -> PredictionResult:
    return PredictionResult(
        name="fsigma8",
        version="1.0",
        metadata={"error": message},
        results={},
        status="error",
    )


@register_prediction
class FSigma8Prediction(PredictionModule):
    name = "fsigma8"
    version = "1.0"
    description = "Predicts fσ₈(z) from the normalized growth rate and σ₈₀."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmin",
            type=float,
            default=_DEFAULT_Z_MIN,
            help="Minimum redshift (default 0.0)",
        )
        parser.add_argument(
            "--zmax",
            type=float,
            default=_DEFAULT_Z_MAX,
            help="Maximum redshift (default 2.0)",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=_DEFAULT_POINTS,
            help="Number of grid points (default 200)",
        )
        super().register(parser)

    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, object]
    ) -> PredictionResult:
        try:
            z_min = float(config.get("zmin", _DEFAULT_Z_MIN))
            z_max = float(config.get("zmax", _DEFAULT_Z_MAX))
            points = int(config.get("points", _DEFAULT_POINTS))
            z_grid = _build_z_grid(z_min, z_max, points)
        except (TypeError, ValueError) as exc:
            return _error_result(f"invalid_grid_configuration: {exc}")

        a_grid = 1.0 / np.clip(1.0 + z_grid, 1e-12, np.inf)
        try:
            H0 = float(model.parameters.get("H0", 67.4))
        except (TypeError, ValueError):
            return _error_result("invalid_H0_value")

        if not np.isfinite(H0) or H0 <= 0.0:
            return _error_result("H0_must_be_positive")

        raw_H = model.H(a_grid)
        H_vals = np.asarray(raw_H, dtype=float)
        if H_vals.shape != a_grid.shape:
            return _error_result("H_shape_mismatch")

        omega_m0 = model.omega_m0()
        E_vals = H_vals / H0
        sorted_idx = np.argsort(a_grid)
        sorted_a = a_grid[sorted_idx]
        sorted_E = E_vals[sorted_idx]
        try:
            D_sorted, _ = solve_growth(sorted_a, sorted_E, omega_m0=omega_m0)
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("fsigma8 solve_growth failed: %s", exc)
            return _error_result("growth_solver_failure")

        inverse_idx = np.argsort(sorted_idx)
        D = D_sorted[inverse_idx]

        mask_base = np.isfinite(D) & (D > 0.0)
        D_norm = np.full_like(D, np.nan)
        if mask_base.any():
            valid_z = z_grid[mask_base]
            idx0 = np.argmin(np.abs(valid_z - 0.0))
            D0 = D[mask_base][idx0]
            if np.isfinite(D0) and D0 > 0.0:
                D_norm = D / D0

        mask_growth = mask_base & np.isfinite(D_norm) & (D_norm > 0.0)
        ln_D = np.full_like(D_norm, np.nan)
        ln_D[mask_growth] = np.log(D_norm[mask_growth])
        ln_a = np.log(np.clip(a_grid, 1e-12, np.inf))
        d_lnD_d_lna = _numerical_derivative(ln_D, ln_a)

        f = np.full_like(D_norm, np.nan)
        mask_f = mask_growth & np.isfinite(d_lnD_d_lna)
        f[mask_f] = d_lnD_d_lna[mask_f]

        sigma8_z = np.full_like(D_norm, np.nan)
        sigma8_valid = False
        sigma8_value = None
        try:
            sigma8_value = float(model.sigma8_today())
            sigma8_valid = np.isfinite(sigma8_value) and sigma8_value > 0.0
        except Exception:
            sigma8_valid = False

        if sigma8_valid:
            sigma8_z[mask_growth] = sigma8_value * D_norm[mask_growth]

        mask_sigma8 = mask_growth & np.isfinite(sigma8_z) & (sigma8_z > 0.0)
        mask_fs8 = mask_f & mask_sigma8
        fs8 = np.full_like(D_norm, np.nan)
        fs8[mask_fs8] = f[mask_fs8] * sigma8_z[mask_fs8]

        mask_valid = mask_fs8
        valid_points = int(mask_valid.sum())
        if valid_points < _MIN_VALID_POINTS:
            logger.warning("fsigma8 mask only has %d valid points", valid_points)

        def value_at_z(target: float) -> float | None:
            if not mask_valid.any():
                return None
            valid_idx = np.where(mask_valid)[0]
            best = valid_idx[np.argmin(np.abs(z_grid[valid_idx] - target))]
            val = fs8[best]
            if not np.isfinite(val):
                return None
            return float(val)

        summary = {
            "fs8_z0": value_at_z(0.0),
            "fs8_z0p5": value_at_z(0.5),
            "fs8_z1": value_at_z(1.0),
        }

        model_name = getattr(model.raw_model.__class__, "__name__", "model")
        meta = {
            "z_min": float(z_min),
            "z_max": float(z_max),
            "n_points": int(points),
            "sigma8_0": float(sigma8_value) if sigma8_valid else None,
            "model_name": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "notes": _NOTES,
            "description": _DESCRIPTION,
        }

        prediction_payload = {
            "name": "fsigma8",
            "z": z_grid.tolist(),
            "a": a_grid.tolist(),
            "D_norm": D_norm.tolist(),
            "f": f.tolist(),
            "sigma8_z": sigma8_z.tolist(),
            "fs8": fs8.tolist(),
            "mask_valid": mask_valid.tolist(),
            "summary": summary,
            "meta": meta,
            "fs8_z0": summary["fs8_z0"],
            "fs8_z0p5": summary["fs8_z0p5"],
            "fs8_z1": summary["fs8_z1"],
            "sigma8_0": meta["sigma8_0"],
        }

        metadata: dict[str, object] = {
            "model": model_name,
            "generated_at": meta["created_at"],
            "points": len(z_grid),
            "valid_points": valid_points,
            "description": _DESCRIPTION,
            "summary": summary,
        }
        if valid_points < _MIN_VALID_POINTS:
            metadata["warnings"] = [
                "fσ8 mask has fewer than 10 valid points; treat with caution."
            ]

        plots = []
        if valid_points > 0:
            plot_data = {
                "z": z_grid[mask_valid].tolist(),
                "fsigma8": fs8[mask_valid].tolist(),
            }
            plots.append(
                PredictionPlot(
                    name="fsigma8_plot",
                    data=plot_data,
                    description="fσ8(z) growth rate prediction (fsigma8)",
                    metadata={"xlabel": "redshift z", "ylabel": "f σ₈(z)"},
                )
            )

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=prediction_payload,
            plots=plots,
        )
