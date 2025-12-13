"""Prediction module tracking the Hubble evolution."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult

logger = logging.getLogger(__name__)

_META_DESCRIPTION = (
    "Horizon evolution prediction: physical and comoving Hubble radii, and a truncated "
    "comoving particle horizon as functions of redshift. These quantities are derived from "
    "the background expansion H(z) and the speed of light c, and illustrate the changing "
    "horizon scales in the chosen cosmological model."
)


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray, initial: float = 0.0) -> np.ndarray:
    """Compute a zero-based cumulative trapezoidal integral over the provided grid."""

    if y.size == 0:
        return np.array([], dtype=float)

    cumulative = np.empty_like(y, dtype=float)
    cumulative[0] = float(initial)
    total = 0.0
    for idx in range(1, y.size):
        dx = abs(x[idx] - x[idx - 1])
        total += 0.5 * dx * (y[idx] + y[idx - 1])
        cumulative[idx] = float(initial + total)
    return cumulative


def _value_at_z(
    array: np.ndarray, z_grid: np.ndarray, mask: np.ndarray, target_z: float
) -> float | None:
    """Return the nearest finite value at `target_z` among the masked samples."""

    if not np.any(mask):
        return None
    valid_idx = np.where(mask)[0]
    nearest_idx = valid_idx[np.argmin(np.abs(z_grid[valid_idx] - target_z))]
    value = array[nearest_idx]
    if not np.isfinite(value):
        return None
    return float(value)


@register_prediction
class HorizonEvolutionPrediction(PredictionModule):
    """Tracks physical/comoving Hubble radii and an optional truncated particle horizon."""

    name = "horizon-evolution"
    version = "1.0"
    description = "Evolution of the Hubble and particle horizons extracted from H(z)."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift (default 0).")
        parser.add_argument("--zmax", type=float, default=10.0, help="Maximum redshift (default 10).")
        parser.add_argument("--points", type=int, default=300, help="Redshift grid size (default 300).")
        parser.add_argument(
            "--no-plot",
            action="store_true",
            help="Skip emitting the canonical horizon evolution plot.",
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
        points = max(int(config.get("points", 300)), 2)
        skip_plot = bool(config.get("no_plot"))

        z_grid = np.linspace(z_min, z_max, points, dtype=float)
        a_grid = 1.0 / np.clip(1.0 + z_grid, 1e-9, np.inf)

        try:
            H_vals = np.asarray(model.background.H(z_grid), dtype=float)
        except Exception:
            H_vals = np.full_like(z_grid, np.nan)
        mask_H = np.isfinite(H_vals) & (H_vals > 0.0)

        try:
            c_value = float(model.background.c_value())
        except Exception:
            c_value = float("nan")
        mask_c = np.isfinite(c_value) and (c_value > 0.0)
        mask_base = mask_H & mask_c

        R_H_phys = np.full_like(H_vals, np.nan)
        R_H_comoving = np.full_like(H_vals, np.nan)
        z_arr = z_grid
        if np.any(mask_base):
            R_H_phys[mask_base] = c_value / H_vals[mask_base]
            R_H_comoving[mask_base] = (c_value * (1.0 + z_arr[mask_base])) / H_vals[mask_base]

        mask_hubble = (
            mask_base
            & np.isfinite(R_H_phys)
            & np.isfinite(R_H_comoving)
            & (R_H_phys > 0.0)
            & (R_H_comoving > 0.0)
        )

        integrand = np.full_like(H_vals, np.nan)
        integrand[mask_base] = c_value / H_vals[mask_base]

        chi_particle = np.full_like(H_vals, np.nan)
        if np.any(mask_base):
            valid_z = z_arr[mask_base]
            valid_f = integrand[mask_base]
            z_rev = valid_z[::-1]
            f_rev = valid_f[::-1]
            chi_rev = _cumulative_trapezoid(f_rev, z_rev, initial=0.0)
            chi_local = chi_rev[::-1]
            chi_particle[mask_base] = chi_local
        mask_particle = mask_base & np.isfinite(chi_particle) & (chi_particle >= 0.0)

        mask_valid = mask_hubble
        valid_count = int(np.count_nonzero(mask_valid))
        if valid_count < 10:
            logger.warning(
                "horizon-evolution prediction only produced %d valid points; check H(z) sampling.",
                valid_count,
            )

        summary = {
            "R_H0_phys": _value_at_z(R_H_phys, z_arr, mask_hubble, 0.0),
            "R_H0_comoving": _value_at_z(R_H_comoving, z_arr, mask_hubble, 0.0),
            "R_H_z1_comoving": _value_at_z(R_H_comoving, z_arr, mask_hubble, 1.0),
            "R_H_z6_comoving": _value_at_z(R_H_comoving, z_arr, mask_hubble, 6.0),
        }

        plots: list[PredictionPlot] = []
        if not skip_plot and valid_count > 0:
            z_valid = z_arr[mask_valid]
            plots.append(
                PredictionPlot(
                    name="horizon_evolution_vs_z",
                    description=(
                        "Comoving Hubble horizon plus optional physical and particle curves vs."
                        " redshift."
                    ),
                    data={
                        "z": z_valid.tolist(),
                        "R_H_comoving": R_H_comoving[mask_valid].tolist(),
                        "R_H_phys": R_H_phys[mask_valid].tolist(),
                        "chi_particle": chi_particle[mask_valid].tolist(),
                    },
                    metadata={
                        "xlabel": "redshift z",
                        "ylabel": "distance (same as c/H(z))",
                    },
                )
            )

        timestamp = datetime.now(timezone.utc).isoformat()
        model_name = type(model.raw_model).__name__
        meta = {
            "description": _META_DESCRIPTION,
            "notes": (
                "Hubble horizon (physical and comoving) and a truncated comoving particle horizon "
                "computed from H(z) and c."
            ),
            "model_name": model_name,
            "z_min": z_min,
            "z_max": z_max,
            "n_points": points,
            "distance_unit": "same as c/H(z)",
            "version": self.version,
            "created_at": timestamp,
        }

        metadata = {
            "model": model_name,
            "model_name": model_name,
            "created_at": timestamp,
            "version": self.version,
            "z_min": z_min,
            "z_max": z_max,
            "n_points": points,
            "distance_unit": "same as c/H(z)",
            "description": _META_DESCRIPTION,
            "notes": meta["notes"],
            "valid_points": valid_count,
            "mask_valid_fraction": valid_count / float(z_grid.size) if z_grid.size else 0.0,
        }

        results = {
            "name": self.name,
            "z": z_arr,
            "a": a_grid,
            "R_H_phys": R_H_phys,
            "R_H_comoving": R_H_comoving,
            "chi_particle": chi_particle,
            "mask_valid": mask_valid,
            "mask_particle": mask_particle,
            "summary": summary,
            "meta": meta,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            plots=plots,
        )
