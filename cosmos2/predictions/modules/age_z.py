"""Cosmic age prediction derived from the background expansion H(z)."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult

logger = logging.getLogger(__name__)


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray, initial: float = 0.0) -> np.ndarray:
    """Zero-based cumulative trapezoidal integration over x."""

    if y.size == 0:
        return np.array([], dtype=float)

    cumulative = np.empty_like(y, dtype=float)
    cumulative[0] = float(initial)
    total = 0.0
    for idx in range(1, y.size):
        dx = x[idx] - x[idx - 1]
        total += 0.5 * dx * (y[idx] + y[idx - 1])
        cumulative[idx] = float(initial + total)
    return cumulative


def _value_at_z(
    array: np.ndarray, z_grid: np.ndarray, mask_valid: np.ndarray, target_z: float
) -> float | None:
    """Return the value nearest to `target_z` among valid entries."""

    if not np.any(mask_valid):
        return None
    valid_idx = np.where(mask_valid)[0]
    nearest = valid_idx[np.argmin(np.abs(z_grid[valid_idx] - target_z))]
    value = array[nearest]
    return float(value) if np.isfinite(value) else None


@register_prediction
class AgeZPrediction(PredictionModule):
    name = "age-z"
    version = "1.0"
    description = "Cosmic age t(z) derived from H(z) via a time integral."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift (default 0.0).")
        parser.add_argument("--zmax", type=float, default=10.0, help="Maximum redshift (default 10.0).")
        parser.add_argument("--points", type=int, default=300, help="Number of redshift samples (default 300).")
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Enable the canonical age-z plot in the prediction payload.",
        )
        super().register(parser)

    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, object]
    ) -> PredictionResult:
        z_min = float(config.get("zmin", 0.0))
        z_max = float(config.get("zmax", 10.0))
        if z_min < 0.0:
            raise ValueError("zmin must be non-negative.")
        if z_max <= z_min:
            raise ValueError("zmax must exceed zmin.")
        n_points = max(int(config.get("points", 300)), 2)
        include_plot = bool(config.get("output_plot"))

        z_grid = np.linspace(z_min, z_max, n_points, dtype=float)
        H_vals = np.asarray(model.background.H(z_grid), dtype=float)
        mask_H = np.isfinite(H_vals) & (H_vals > 0.0)

        integrand = np.full_like(H_vals, np.nan)
        valid_indices = np.where(mask_H)[0]
        if valid_indices.size > 0:
            integrand[valid_indices] = 1.0 / ((1.0 + z_grid[valid_indices]) * H_vals[valid_indices])
            tL_raw = np.full_like(H_vals, np.nan)
            cumulative = _cumulative_trapezoid(integrand[valid_indices], z_grid[valid_indices])
            tL_raw[valid_indices] = cumulative
        else:
            tL_raw = np.full_like(H_vals, np.nan)

        time_factor = model.background.get_time_conversion_to_Gyr()
        if not (np.isfinite(time_factor) and time_factor > 0.0):
            logger.warning("Invalid time conversion factor; age-z outputs masked.")
            tL_Gyr = np.full_like(tL_raw, np.nan)
        else:
            tL_Gyr = tL_raw * float(time_factor)

        mask_tL = mask_H & np.isfinite(tL_Gyr) & (tL_Gyr >= 0.0)
        t0_Gyr = None
        if np.any(mask_tL):
            idx_max = np.where(mask_tL)[0][-1]
            t0_Gyr = float(tL_Gyr[idx_max])

        t_age_Gyr = np.full_like(tL_Gyr, np.nan)
        if t0_Gyr is not None:
            t_age_Gyr[mask_tL] = t0_Gyr - tL_Gyr[mask_tL]

        mask_valid = mask_tL & np.isfinite(t_age_Gyr) & (t_age_Gyr >= 0.0)
        valid_count = int(np.count_nonzero(mask_valid))
        if valid_count < 10:
            logger.warning(
                "Age-z prediction returned only %d valid points; check H(z) sampling or masks.",
                valid_count,
            )

        t_z1_Gyr = _value_at_z(t_age_Gyr, z_grid, mask_valid, 1.0)
        t_z6_Gyr = _value_at_z(t_age_Gyr, z_grid, mask_valid, 6.0)

        z_half_age = None
        if t0_Gyr is not None and valid_count > 0:
            target = 0.5 * t0_Gyr
            z_valid = z_grid[mask_valid]
            t_valid = t_age_Gyr[mask_valid]
            for idx in range(1, len(z_valid)):
                if (t_valid[idx - 1] > target) and (t_valid[idx] <= target):
                    z1, z2 = z_valid[idx - 1], z_valid[idx]
                    t1, t2 = t_valid[idx - 1], t_valid[idx]
                    if t1 != t2:
                        frac = (t1 - target) / (t1 - t2)
                        z_half_age = float(z1 + frac * (z2 - z1))
                    else:
                        z_half_age = float(z_valid[idx])
                    break

        summary = {
            "t0_Gyr": t0_Gyr,
            "t_z1_Gyr": t_z1_Gyr,
            "t_z6_Gyr": t_z6_Gyr,
            "z_half_age": z_half_age,
        }

        meta_payload = {
            "z_min": z_min,
            "z_max": z_max,
            "n_points": n_points,
            "time_unit": "Gyr",
            "model_name": type(model.raw_model).__name__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": self.version,
            "notes": "Cosmic age t(z) and lookback time t_L(z) in Gyr, derived from H(z).",
            "description": (
                "Cosmic age prediction t(z) in Gyr. "
                "Computed from the background expansion H(z) via a time integral. "
                "The module reports the present age of the universe, the age at key redshifts, "
                "and the redshift where the universe reaches half of its current age."
            ),
        }

        plots: list[PredictionPlot] = []
        if include_plot and valid_count > 0:
            z_plot = z_grid[mask_valid]
            t_age_plot = t_age_Gyr[mask_valid]
            tL_plot = tL_Gyr[mask_valid]
            plots.append(
                PredictionPlot(
                    name="cosmic_age_vs_z",
                    description="Cosmic age t(z) with lookback time for comparison.",
                    data={"z": z_plot.tolist(), "t_age_Gyr": t_age_plot.tolist(), "tL_Gyr": tL_plot.tolist()},
                    metadata={
                        "xlabel": "redshift z",
                        "ylabel": "cosmic age t(z) [Gyr]",
                        "t0_Gyr": t0_Gyr,
                        "z_half_age": z_half_age,
                    },
                )
            )

        results_payload = {
            "name": self.name,
            "z": z_grid.tolist(),
            "t_age_Gyr": t_age_Gyr.tolist(),
            "tL_Gyr": tL_Gyr.tolist(),
            "mask_valid": mask_valid.tolist(),
            "summary": summary,
            "meta": meta_payload,
        }

        metadata_payload = {
            "model": type(model.raw_model).__name__,
            "model_name": type(model.raw_model).__name__,
            "created_at": meta_payload["created_at"],
            "z_min": z_min,
            "z_max": z_max,
            "n_points": n_points,
            "time_unit": "Gyr",
            "version": self.version,
            "notes": meta_payload["notes"],
            "description": meta_payload["description"],
            "summary": summary,
            "valid_points": valid_count,
            "mask_valid_fraction": valid_count / float(len(z_grid)) if z_grid.size > 0 else 0.0,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata_payload,
            results=results_payload,
            plots=plots,
        )
