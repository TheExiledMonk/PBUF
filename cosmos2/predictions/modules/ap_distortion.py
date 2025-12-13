"""Predict the Alcock–Paczynski distortion F(z) on a configurable redshift grid."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult

logger = logging.getLogger(__name__)

_DEFAULT_Z_MIN = 0.0
_DEFAULT_Z_MAX = 3.0
_DEFAULT_POINTS = 200
_MIN_VALID_POINTS = 10
_DESCRIPTION = (
    "Alcock–Paczynski distortion parameter F(z) = (1+z) D_A(z) H(z) / c. "
    "Widely used in BAO anisotropy and clustering analyses, F(z) provides a model-dependent relation "
    "between radial and transverse distances. This module predicts F(z) for the current cosmology "
    "on a configurable redshift grid."
)
_NOTES = "Alcock–Paczynski distortion parameter F(z) computed from H(z) and D_A(z)."


def _build_z_grid(z_min: float, z_max: float, points: int) -> np.ndarray:
    if z_min < 0.0:
        raise ValueError("zmin must be non-negative.")
    if z_max <= z_min:
        raise ValueError("zmax must be greater than zmin.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    return np.linspace(float(z_min), float(z_max), num=int(points), dtype=float)


def _error_result(message: str) -> PredictionResult:
    return PredictionResult(
        name="ap-distortion",
        version="1.0",
        metadata={"error": message},
        results={},
        status="error",
    )


@register_prediction
class APDistortionPrediction(PredictionModule):
    name = "ap-distortion"
    version = "1.0"
    description = "Predicts the Alcock–Paczynski distortion parameter F(z) from H(z) and D_A(z)."

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
            help="Maximum redshift (default 3.0)",
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

        H_vals = np.full(z_grid.shape, np.nan, dtype=float)
        D_A_vals = np.full(z_grid.shape, np.nan, dtype=float)
        c_value = float("nan")
        background = model.background

        try:
            candidate = np.asarray(background.H(z_grid), dtype=float)
            if candidate.shape == z_grid.shape:
                H_vals = candidate
            else:
                logger.warning(
                    "ap-distortion H(z) returned shape %s for %d points",
                    candidate.shape,
                    len(z_grid),
                )
        except Exception as exc:
            logger.exception("ap-distortion H(z) evaluation failed: %s", exc)

        try:
            candidate = np.asarray(background.D_A(z_grid), dtype=float)
            if candidate.shape == z_grid.shape:
                D_A_vals = candidate
            else:
                logger.warning(
                    "ap-distortion D_A(z) returned shape %s for %d points",
                    candidate.shape,
                    len(z_grid),
                )
        except Exception as exc:
            logger.exception("ap-distortion D_A(z) evaluation failed: %s", exc)

        try:
            c_value = float(background.c_value())
        except Exception as exc:  # pragma: no cover - best-effort guard
            logger.exception("ap-distortion c_value access failed: %s", exc)
            c_value = float("nan")

        mask_base = (
            np.isfinite(H_vals)
            & np.isfinite(D_A_vals)
            & (H_vals > 0.0)
            & (D_A_vals > 0.0)
            & np.isfinite(c_value)
            & (c_value > 0.0)
        )

        F_vals = np.full_like(H_vals, np.nan)
        valid = mask_base
        if np.isfinite(c_value) and c_value > 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                F_vals[valid] = (1.0 + z_grid[valid]) * D_A_vals[valid] * H_vals[valid] / c_value

        mask_valid = valid & np.isfinite(F_vals)
        valid_points = int(np.count_nonzero(mask_valid))
        if valid_points < _MIN_VALID_POINTS:
            logger.warning("ap-distortion mask only has %d valid points", valid_points)

        def value_at_z(target: float) -> float | None:
            if not mask_valid.any():
                return None
            valid_idx = np.where(mask_valid)[0]
            best = valid_idx[np.argmin(np.abs(z_grid[valid_idx] - target))]
            val = F_vals[best]
            if not np.isfinite(val):
                return None
            return float(val)

        summary = {
            "F_z0p5": value_at_z(0.5),
            "F_z1": value_at_z(1.0),
            "F_z2": value_at_z(2.0),
        }

        model_name = getattr(model.raw_model.__class__, "__name__", "model")
        created_at = datetime.now(timezone.utc).isoformat()
        meta = {
            "z_min": float(z_min),
            "z_max": float(z_max),
            "n_points": int(points),
            "c": float(c_value) if np.isfinite(c_value) else None,
            "model_name": model_name,
            "created_at": created_at,
            "version": "1.0",
            "notes": _NOTES,
            "description": _DESCRIPTION,
        }

        prediction_payload = {
            "name": "ap-distortion",
            "z": z_grid.tolist(),
            "H": H_vals.tolist(),
            "D_A": D_A_vals.tolist(),
            "F": F_vals.tolist(),
            "mask_valid": mask_valid.tolist(),
            "summary": summary,
            "meta": meta,
        }

        metadata: dict[str, object] = {
            "model": model_name,
            "generated_at": created_at,
            "points": len(z_grid),
            "valid_points": valid_points,
            "description": _DESCRIPTION,
            "summary": summary,
        }
        if valid_points < _MIN_VALID_POINTS:
            metadata["warnings"] = [
                "ap-distortion has fewer than 10 valid points; results may be unreliable."
            ]

        plots: list[PredictionPlot] = []
        if valid_points > 0:
            plot_series = {
                "z": z_grid[mask_valid].tolist(),
                "F": F_vals[mask_valid].tolist(),
            }
            plots.append(
                PredictionPlot(
                    name="ap_distortion_Fz",
                    description="Alcock–Paczynski distortion parameter F(z)",
                    data=plot_series,
                    metadata={"xlabel": "redshift z", "ylabel": "F(z)"},
                )
            )

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=prediction_payload,
            plots=plots,
        )
