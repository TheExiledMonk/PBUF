"""Lookback time prediction computed from the background expansion."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

logger = logging.getLogger(__name__)


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray, initial: float = 0.0) -> np.ndarray:
    """Compute a zero-based cumulative trapezoidal integral."""

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
    """Return the value nearest `target_z` among the valid samples."""

    if not np.any(mask_valid):
        return None
    valid_idx = np.where(mask_valid)[0]
    nearest = valid_idx[np.argmin(np.abs(z_grid[valid_idx] - target_z))]
    value = array[nearest]
    return float(value) if np.isfinite(value) else None


@register_prediction
class LookbackPrediction(PredictionModule):
    name = "lookback"
    version = "1.0"
    description = "Lookback time and cosmic age derived from H(z)."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift (default 0.0).")
        parser.add_argument("--zmax", type=float, default=10.0, help="Maximum redshift (default 10.0).")
        parser.add_argument("--points", type=int, default=300, help="Number of redshift samples (default 300).")
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Include canonical lookback/age plots in the prediction payload.",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Export the lookback history table (z, t_L, t_age, valid flag).",
        )
        super().register(parser)

    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, object]
    ) -> PredictionResult:
        zmin = float(config.get("zmin", 0.0))
        zmax = float(config.get("zmax", 10.0))
        if zmin < 0.0:
            raise ValueError("zmin must be non-negative.")
        if zmax <= zmin:
            raise ValueError("zmax must exceed zmin.")
        points = max(int(config.get("points", 300)), 2)
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        z_grid = np.linspace(zmin, zmax, points, dtype=float)
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
            logger.warning("Time conversion factor from background is invalid; lookback times set to NaN.")
            tL = np.full_like(tL_raw, np.nan)
        else:
            tL = tL_raw * time_factor
        mask_valid = mask_H & np.isfinite(tL) & (tL >= 0.0)
        valid_count = int(np.count_nonzero(mask_valid))
        if valid_count < 10:
            logger.warning(
                "Lookback prediction returned only %d valid points; check H(z) sampling or active masks.",
                valid_count,
            )

        t0 = None
        if valid_count:
            t0 = float(tL[mask_valid][-1])

        t_age = np.full_like(tL, np.nan)
        if t0 is not None:
            t_age[mask_valid] = t0 - tL[mask_valid]

        summary = {
            "t0_Gyr": t0,
            "t_z1_Gyr": _value_at_z(t_age, z_grid, mask_valid, 1.0),
            "t_z6_Gyr": _value_at_z(t_age, z_grid, mask_valid, 6.0),
        }

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                [
                    float(z_val),
                    float(tL[idx]) if mask_valid[idx] else None,
                    float(t_age[idx]) if mask_valid[idx] else None,
                    bool(mask_valid[idx]),
                ]
                for idx, z_val in enumerate(z_grid)
            ]
            tables.append(
                PredictionTable(
                    name="lookback_vs_z",
                    columns=["z", "t_L_Gyr", "t_age_Gyr", "mask_valid"],
                    rows=rows,
                    metadata={"points": len(z_grid), "units": "Gyr"},
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot and valid_count > 0:
            z_valid = z_grid[mask_valid]
            tL_valid = tL[mask_valid]
            plots.append(
                PredictionPlot(
                    name="lookback_time_vs_z",
                    description="Lookback time t_L(z) in gigayears",
                    data={"z": z_valid.tolist(), "tL_Gyr": tL_valid.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "lookback time t_L(z) [Gyr]"},
                )
            )
            t_age_valid = t_age[mask_valid]
            plots.append(
                PredictionPlot(
                    name="cosmic_age_vs_z",
                    description="Cosmic age t(z) in gigayears",
                    data={
                        "z": z_valid.tolist(),
                        "t_age_Gyr": t_age_valid.tolist(),
                        "t0_Gyr": [float(t0)] * len(z_valid) if t0 is not None else [],
                    },
                    metadata={"xlabel": "redshift z", "ylabel": "cosmic age t(z) [Gyr]"},
                )
            )

        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {
            "model": type(model.raw_model).__name__,
            "model_name": type(model.raw_model).__name__,
            "created_at": timestamp,
            "timestamp": timestamp,
            "z_min": zmin,
            "z_max": zmax,
            "n_points": points,
            "time_unit": "Gyr",
            "version": self.version,
            "description": (
                "Lookback time prediction t_L(z) and cosmic age t(z) in Gyr. "
                "Computed from the background expansion H(z) via a line-of-sight integral. "
                "The module reports the present age of the universe t0 and the age at key redshifts, "
                "useful for comparing to galaxy ages and star-formation histories."
            ),
            "notes": "Lookback time and cosmic age computed from H(z) via a line-of-sight integral.",
            "summary": summary,
            "valid_points": valid_count,
            "mask_valid_fraction": valid_count / float(len(z_grid)) if z_grid.size > 0 else 0.0,
        }

        results = {
            "z": z_grid.tolist(),
            "tL_Gyr": tL.tolist(),
            "t_age_Gyr": t_age.tolist(),
            "mask_valid": mask_valid.tolist(),
            "t0_Gyr": summary["t0_Gyr"],
            "t_z1_Gyr": summary["t_z1_Gyr"],
            "t_z6_Gyr": summary["t_z6_Gyr"],
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
