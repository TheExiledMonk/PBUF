"""Elastic energy fraction prediction module."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult

logger = logging.getLogger(__name__)

_DEFAULT_ZMIN = 0.0
_DEFAULT_ZMAX = 5.0
_DEFAULT_POINTS = 300


def _value_at_z(array: np.ndarray, z_grid: np.ndarray, mask: np.ndarray, target_z: float) -> float | None:
    if not np.any(mask):
        return None
    valid_indices = np.where(mask)[0]
    if valid_indices.size == 0:
        return None
    differences = np.abs(z_grid[valid_indices] - target_z)
    closest = valid_indices[int(np.argmin(differences))]
    value = array[closest]
    return float(value) if np.isfinite(value) else None


def _find_half_peak(z_values: np.ndarray, f_values: np.ndarray, target_half: float, reverse: bool) -> float | None:
    length = len(z_values)
    if length < 2:
        return None
    if reverse:
        iterator = range(length - 1, 0, -1)
    else:
        iterator = range(1, length)
    for idx in iterator:
        prev_idx = idx - 1 if not reverse else idx - 1
        curr_idx = idx
        z_lo, z_hi = z_values[prev_idx], z_values[curr_idx]
        f_lo, f_hi = f_values[prev_idx], f_values[curr_idx]
        crosses = (
            (f_lo >= target_half and f_hi <= target_half)
            or (f_lo <= target_half and f_hi >= target_half)
        )
        if not crosses:
            continue
        if f_hi == f_lo:
            return float(z_values[curr_idx])
        frac = (target_half - f_lo) / (f_hi - f_lo)
        return float(z_lo + frac * (z_hi - z_lo))
    return None


@register_prediction
class ElasticFractionPrediction(PredictionModule):
    name = "elastic-fraction"
    version = "1.0"
    description = "Elastic energy density fraction Ωσ/Ω_tot(z) and its diagnostics."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--z-min", type=float, default=_DEFAULT_ZMIN, help="Minimum redshift (default 0.0)")
        parser.add_argument("--z-max", type=float, default=_DEFAULT_ZMAX, help="Maximum redshift (default 5.0)")
        parser.add_argument("--points", type=int, default=_DEFAULT_POINTS, help="Number of redshift samples (default 300)")
        super().register(parser)

    def run_prediction(self, model: PredictionModelAdapter, config: dict[str, object]) -> PredictionResult:
        z_min = float(config.get("z_min", _DEFAULT_ZMIN))
        z_max = float(config.get("z_max", _DEFAULT_ZMAX))
        points = int(config.get("points", _DEFAULT_POINTS))

        if z_min < 0.0:
            raise ValueError("z_min must be non-negative.")
        if z_max < z_min:
            raise ValueError("z_max must be greater than or equal to z_min.")
        if points < 2:
            raise ValueError("points must be at least 2.")

        z_grid = np.linspace(z_min, z_max, num=points, dtype=float)
        a_grid = np.clip(1.0 / np.clip(1.0 + z_grid, 1e-12, np.inf), 0.0, 1.0)

        omega_sigma = np.asarray(model.elastic.omega_sigma(a_grid), dtype=float)
        if omega_sigma.shape != a_grid.shape:
            raise RuntimeError("Ωσ grid size mismatch.")

        E_values = np.asarray(model.background.E(a_grid), dtype=float)
        if E_values.shape != a_grid.shape:
            raise RuntimeError("E(a) grid size mismatch.")
        omega_tot = np.square(E_values)

        mask_base = (
            np.isfinite(omega_sigma)
            & np.isfinite(omega_tot)
            & (omega_tot > 0.0)
        )

        f_sigma = np.full_like(omega_sigma, np.nan)
        safe_indices = mask_base
        f_sigma[safe_indices] = omega_sigma[safe_indices] / omega_tot[safe_indices]
        mask_valid = safe_indices & np.isfinite(f_sigma)

        valid_points = int(np.count_nonzero(mask_valid))
        if valid_points < 10:
            logger.warning(
                "elastic-fraction produced only %d valid samples for model %s",
                valid_points,
                model.raw_model.__class__.__name__,
            )

        omega_sigma_0 = _value_at_z(omega_sigma, z_grid, mask_valid, 0.0)
        f_sigma_0 = _value_at_z(f_sigma, z_grid, mask_valid, 0.0)

        f_sigma_peak = None
        z_peak = None
        z_half_peak_lo = None
        z_half_peak_hi = None

        if valid_points > 0:
            z_valid = z_grid[mask_valid]
            f_valid = f_sigma[mask_valid]
            omega_sigma_valid = omega_sigma[mask_valid]

            peak_idx = int(np.argmax(f_valid))
            f_sigma_peak = float(f_valid[peak_idx])
            z_peak = float(z_valid[peak_idx])

            target_half = 0.5 * f_sigma_peak
            left_mask = z_valid < z_peak
            if np.any(left_mask):
                z_half_peak_lo = _find_half_peak(
                    z_valid[left_mask], f_valid[left_mask], target_half, reverse=True
                )

            right_mask = z_valid > z_peak
            if np.any(right_mask):
                z_half_peak_hi = _find_half_peak(
                    z_valid[right_mask], f_valid[right_mask], target_half, reverse=False
                )

            plot_data = {
                "z": z_valid.tolist(),
                "f_sigma": f_valid.tolist(),
                "Omega_sigma": omega_sigma_valid.tolist(),
            }
            plots = [
                PredictionPlot(
                    name="f_sigma_vs_z",
                    data=plot_data,
                    description="Elastic fraction fσ(z)=Ωσ/Ω_tot",
                    metadata={
                        "xlabel": "redshift z",
                        "ylabel": "fσ(z)",
                        "secondary_ylabel": "Ωσ(z)",
                        "points": valid_points,
                        "z_peak": z_peak,
                        "f_sigma_peak": f_sigma_peak,
                    },
                )
            ]
        else:
            plots = []

        results = {
            "name": self.name,
            "z": z_grid,
            "a": a_grid,
            "Omega_sigma": omega_sigma,
            "Omega_tot": omega_tot,
            "f_sigma": f_sigma,
            "mask_valid": mask_valid,
            "summary": {
                "Omega_sigma_0": omega_sigma_0,
                "f_sigma_0": f_sigma_0,
                "f_sigma_peak": f_sigma_peak,
                "z_peak": z_peak,
                "z_half_peak_lo": z_half_peak_lo,
                "z_half_peak_hi": z_half_peak_hi,
            },
            "meta": {
                "z_min": float(z_min),
                "z_max": float(z_max),
                "n_points": int(z_grid.size),
                "model_name": model.raw_model.__class__.__name__,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": self.version,
                "description": (
                    "Elastic energy density Ωσ(z) and its fraction fσ(z)=Ωσ/Ω_tot, derived from the model's "
                    "elastic sector and background expansion. The module tracks when and how strongly the "
                    "elastic component contributes to the total cosmic energy budget."
                ),
                "notes": "Elastic energy density Ωσ(z) and fraction fσ(z)=Ωσ/Ω_tot derived from the model's elastic sector and background E(a).",
            },
        }

        module_description = (
            "Elastic energy density Ωσ(z) and its fraction fσ(z)=Ωσ/Ω_tot, derived from the model's "
            "elastic sector and background expansion."
        )
        metadata = {
            "model_name": model.raw_model.__class__.__name__,
            "grid_points": len(z_grid),
            "valid_points": valid_points,
            "z_min": float(z_min),
            "z_max": float(z_max),
            "description": module_description,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            plots=plots,
        )
