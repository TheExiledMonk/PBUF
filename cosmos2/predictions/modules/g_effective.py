"""Predict the effective gravitational strength μ(z)=G_eff(z)/G_N from the linear growth history."""

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
_DEFAULT_Z_MAX = 2.0
_DEFAULT_POINTS = 200
_MIN_VALID_POINTS_WARNING = 10
_SMALL_DENOMINATOR = 1e-12

_NOTES = "Effective gravitational strength inferred from the inverted linear growth equation."
_DESCRIPTION = (
    "Effective gravitational strength μ(z) = G_eff(z)/G_N inferred from the linear growth "
    "equation. In GR with standard matter, μ(z) ≈ 1, so deviations highlight modified or "
    "effective changes to the growth source term."
)


def _numerical_derivative(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    grid = np.asarray(coords, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError("Values and coordinates must share the same shape.")
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


def _value_at_z(array: np.ndarray, z_grid: np.ndarray, mask: np.ndarray, target_z: float) -> float | None:
    if not mask.any():
        return None
    valid_idx = np.where(mask)[0]
    best_index = valid_idx[np.argmin(np.abs(z_grid[valid_idx] - float(target_z)))]
    value = array[best_index]
    if not np.isfinite(value):
        return None
    return float(value)


def _marker_series(z_grid: np.ndarray, values: np.ndarray, mask: np.ndarray, target_z: float) -> np.ndarray:
    marker = np.full_like(values, np.nan)
    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        return marker
    best_index = valid_idx[np.argmin(np.abs(z_grid[valid_idx] - float(target_z)))]
    value = values[best_index]
    if np.isfinite(value):
        marker[best_index] = float(value)
    return marker


@register_prediction
class GEffectivePrediction(PredictionModule):
    """Predict the inverted growth source term to recover effective μ(z)=G_eff/G_N."""

    name = "g-effective"
    version = "1.0"
    description = "Infers the effective gravitational strength μ(z)=G_eff(z)/G_N from growth."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=_DEFAULT_Z_MIN, help="Minimum redshift (default 0.0)")
        parser.add_argument("--zmax", type=float, default=_DEFAULT_Z_MAX, help="Maximum redshift (default 2.0)")
        parser.add_argument("--points", type=int, default=_DEFAULT_POINTS, help="Number of redshift samples (default 200)")
        super().register(parser)

    def run_prediction(self, model: PredictionModelAdapter, config: dict[str, object]) -> PredictionResult:
        z_min = float(config.get("zmin", _DEFAULT_Z_MIN))
        z_max = float(config.get("zmax", _DEFAULT_Z_MAX))
        points = int(config.get("points", _DEFAULT_POINTS))
        if z_min < 0.0:
            raise ValueError("zmin must be non-negative.")
        if z_max <= z_min:
            raise ValueError("zmax must be greater than zmin.")
        if points < 2:
            raise ValueError("points must be at least 2.")

        z_grid = np.linspace(float(z_min), float(z_max), num=points, dtype=float)
        a_grid = np.clip(1.0 / (1.0 + z_grid), 1e-12, 1.0)

        D = np.asarray(model.growth.solve_growth(a_grid), dtype=float)
        if D.shape != a_grid.shape:
            raise RuntimeError("Growth solver returned unexpected shape.")
        E_vals = np.asarray(model.background.E(a_grid), dtype=float)
        if E_vals.shape != a_grid.shape:
            raise RuntimeError("Background E(a) grid size mismatch.")
        omega_m = np.asarray(model.background.Omega_m_a(a_grid), dtype=float)
        if omega_m.shape != a_grid.shape:
            raise RuntimeError("Omega_m(a) grid size mismatch.")

        mask_base = (
            np.isfinite(D) & (D > 0.0)
            & np.isfinite(E_vals) & (E_vals > 0.0)
            & np.isfinite(omega_m) & (omega_m > 0.0)
        )

        dD_da = _numerical_derivative(D, a_grid)
        d2D_da2 = _numerical_derivative(dD_da, a_grid)
        ln_E = np.full_like(E_vals, np.nan)
        ln_E[mask_base] = np.log(E_vals[mask_base])
        ln_a = np.log(np.clip(a_grid, 1e-12, np.inf))
        dlnE_dlnA = _numerical_derivative(ln_E, ln_a)

        mask_deriv = (
            mask_base
            & np.isfinite(dD_da)
            & np.isfinite(d2D_da2)
            & np.isfinite(dlnE_dlnA)
        )

        numerator = d2D_da2 + (3.0 / a_grid + dlnE_dlnA / a_grid) * dD_da
        denominator = 1.5 * omega_m * D / (a_grid ** 2)
        mask_den = mask_deriv & np.isfinite(denominator) & (np.abs(denominator) > _SMALL_DENOMINATOR)

        mu = np.full_like(D, np.nan)
        mu[mask_den] = numerator[mask_den] / denominator[mask_den]
        mask_valid = mask_den & np.isfinite(mu)

        valid_points = int(mask_valid.sum())
        if valid_points < _MIN_VALID_POINTS_WARNING:
            logger.warning(
                "g-effective prediction only has %d valid samples (< %d desirable).",
                valid_points,
                _MIN_VALID_POINTS_WARNING,
            )

        mu0 = _value_at_z(mu, z_grid, mask_valid, 0.0)
        mu_z0p5 = _value_at_z(mu, z_grid, mask_valid, 0.5)
        mu_z1 = _value_at_z(mu, z_grid, mask_valid, 1.0)
        mask_range = mask_valid & (z_grid >= 0.0) & (z_grid <= 1.0)
        mu_mean_0_1 = float(np.nanmean(mu[mask_range])) if np.count_nonzero(mask_range) > 0 else None

        model_name = getattr(model.raw_model.__class__, "__name__", "model")
        summary = {
            "mu0": mu0,
            "mu_z0p5": mu_z0p5,
            "mu_z1": mu_z1,
            "mu_mean_0_1": mu_mean_0_1,
        }

        meta = {
            "z_min": float(z_min),
            "z_max": float(z_max),
            "n_points": int(points),
            "model_name": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": self.version,
            "notes": _NOTES,
            "description": _DESCRIPTION,
        }

        stores = {
            "name": self.name,
            "z": z_grid,
            "a": a_grid,
            "mu": mu,
            "mask_valid": mask_valid,
            "summary": summary,
            "meta": meta,
        }

        plots: list[PredictionPlot] = []
        if mask_valid.any():
            z_valid = np.asarray(z_grid[mask_valid], dtype=float)
            mu_valid = np.asarray(mu[mask_valid], dtype=float)
            ones = np.ones_like(mu_valid, dtype=float)
            marker0 = _marker_series(z_grid, mu, mask_valid, 0.0)[mask_valid]
            marker_z0p5 = _marker_series(z_grid, mu, mask_valid, 0.5)[mask_valid]
            marker_z1 = _marker_series(z_grid, mu, mask_valid, 1.0)[mask_valid]
            plots.append(
                PredictionPlot(
                    name="g_effective_vs_z",
                    description="Effective gravitational strength μ(z) = G_eff/G_N (g-effective prediction)",
                    data={
                        "z": z_valid.tolist(),
                        "mu": mu_valid.tolist(),
                        "mu_ref": ones.tolist(),
                        "mu0_marker": marker0.tolist(),
                        "mu_z0p5_marker": marker_z0p5.tolist(),
                        "mu_z1_marker": marker_z1.tolist(),
                    },
                    metadata={"xlabel": "redshift z", "ylabel": "μ(z)"},
                )
            )

        metadata: dict[str, object] = {
            "model": model_name,
            "points": len(z_grid),
            "valid_points": valid_points,
        }
        if valid_points < _MIN_VALID_POINTS_WARNING:
            metadata["warnings"] = [
                f"Only {valid_points} reliable samples (need ≥ {_MIN_VALID_POINTS_WARNING})."
            ]

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=stores,
            plots=plots,
        )
