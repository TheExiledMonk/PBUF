"""Predict the cosmological growth factor and rate."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from cosmos2.kernels.common.growth import solve_growth

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

_ZGRID_SPLIT = re.compile(r"[\s,;]+")
_DEFAULT_ZMAX = 5.0
_DEFAULT_POINTS = 300
_MIN_GROWTH_STEPS = 2048
_MAX_GROWTH_STEPS = 20000


def _parse_zgrid(value: str) -> list[float]:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("Empty zgrid specification.")
    tokens = [token for token in _ZGRID_SPLIT.split(trimmed) if token]
    if not tokens:
        raise ValueError("Empty zgrid specification.")
    parsed: list[float] = []
    for token in tokens:
        try:
            redshift = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid redshift '{token}' in zgrid.") from exc
        if redshift <= -1.0:
            raise ValueError("Redshift values must be greater than -1.")
        parsed.append(redshift)
    return sorted(set(parsed))


def _build_z_grid(zmax: float, points: int, zgrid: str | None) -> np.ndarray:
    if zgrid is not None:
        values = _parse_zgrid(zgrid)
        if not values:
            raise ValueError("zgrid must provide at least one value.")
        return np.array(values, dtype=float)
    if points < 2:
        raise ValueError("points must be at least 2.")
    if zmax < 0.0:
        raise ValueError("zmax must be non-negative.")
    return np.linspace(0.0, float(zmax), num=max(2, points), dtype=float)


def _build_integration_grid(a_samples: Sequence[float]) -> np.ndarray:
    arr = np.asarray(a_samples, dtype=float)
    if arr.size == 0:
        raise ValueError("No scale factors provided for growth integration.")
    safe = np.clip(arr, 1e-12, np.inf)
    a_min = float(np.min(safe))
    a_max = float(np.max(safe))
    a_start = min(1e-4, a_min)
    a_start = max(a_start, 1e-10)
    a_stop = max(1.0, a_max)
    step_count = int(min(max(_MIN_GROWTH_STEPS, arr.size * 8), _MAX_GROWTH_STEPS))
    step_count = max(step_count, 2)
    if a_start <= 0.0 or a_stop <= 0.0:
        raise ValueError("Scale factor grid must be positive.")
    if a_start == a_stop:
        return np.array([a_start, a_stop], dtype=float)
    return np.logspace(np.log10(a_start), np.log10(a_stop), step_count, dtype=float)


def _interpolate_log_space(grid: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    log_grid = np.log(grid)
    clipped = np.clip(targets, grid[0], grid[-1])
    return np.interp(np.log(clipped), log_grid, values)


@register_prediction
class GrowthPrediction(PredictionModule):
    name = "growth"
    version = "v1"
    description = "Predicts the growth factor D(a), growth rate f(z), and optional fσ₈(z)."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--zmax", type=float, default=_DEFAULT_ZMAX, help="Maximum redshift (default 5.0)")
        parser.add_argument(
            "--points",
            type=int,
            default=_DEFAULT_POINTS,
            help="Number of redshift samples (default 300)",
        )
        parser.add_argument(
            "--include-s8",
            action="store_true",
            help="Compute σ₈(z) and fσ₈(z) using the model's σ₈ definition.",
        )
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Include canonical growth plots in the prediction payload.",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Export the growth history table (z, a, D, f, σ₈, fσ₈).",
        )
        parser.add_argument(
            "--zgrid",
            type=str,
            help="Comma/space/semicolon separated custom z points (overrides --points).",
        )
        super().register(parser)

    def run_prediction(self, model: PredictionModelAdapter, config: dict[str, object]) -> PredictionResult:
        zmax = float(config.get("zmax", _DEFAULT_ZMAX))
        points = int(config.get("points", _DEFAULT_POINTS))
        include_s8 = bool(config.get("include_s8", False))
        output_plot = bool(config.get("output_plot", False))
        output_table = bool(config.get("output_table", False))
        zgrid = config.get("zgrid")

        z_values = _build_z_grid(zmax, points, zgrid if isinstance(zgrid, str) else None)
        a_targets = 1.0 / (1.0 + z_values)
        a_grid = _build_integration_grid(a_targets)

        H0 = float(model.parameters.get("H0", 67.4))
        if H0 <= 0.0:
            raise ValueError("Model reports non-positive H0.")
        H_vals = model.H(a_grid)
        if H_vals.shape != a_grid.shape:
            raise RuntimeError("Model H(a) grid size mismatch.")
        E_vals = np.asarray(H_vals, dtype=float) / H0

        omega_m0 = model.omega_m0()
        D_grid, _ = solve_growth(a_grid, E_vals, omega_m0=omega_m0)
        dD_da = np.gradient(D_grid, a_grid)
        D_targets = _interpolate_log_space(a_grid, D_grid, a_targets)
        dD_targets = _interpolate_log_space(a_grid, dD_da, a_targets)
        safe_D = np.clip(D_targets, 1e-12, np.inf)
        f_targets = (a_targets / safe_D) * dD_targets

        sigma8_series: list[float] | None = None
        f_sigma8_series: list[float] | None = None
        sigma8_today = None
        if include_s8:
            sigma8_today = model.sigma8_today()
            sigma8_series = (sigma8_today * D_targets).tolist()
            f_sigma8_series = (np.asarray(f_targets) * np.asarray(sigma8_series)).tolist()

        z_samples = [0.0, 1.0, 2.0]
        results: dict[str, float | None] = {"zmax": float(np.max(z_values))}
        for z_sample in z_samples:
            results[f"f_at_z{int(z_sample)}"] = float(
                np.interp(z_sample, z_values, f_targets, left=f_targets[0], right=f_targets[-1])
            )

        if include_s8 and sigma8_series is not None and f_sigma8_series is not None:
            results["s8_today"] = float(sigma8_today)
            for label, target_z in (("fs8_at_z0", 0.0), ("fs8_at_z0.5", 0.5), ("fs8_at_z1", 1.0)):
                sigma_interp = np.interp(
                    target_z, z_values, sigma8_series, left=sigma8_series[0], right=sigma8_series[-1]
                )
                f_interp = np.interp(
                    target_z, z_values, f_targets, left=f_targets[0], right=f_targets[-1]
                )
                results[label] = float(f_interp * sigma_interp)

        tables: list[PredictionTable] = []
        if output_table:
            rows: list[list[float | None]] = []
            for idx, z_val in enumerate(z_values):
                row: list[float | None] = [
                    float(z_val),
                    float(a_targets[idx]),
                    float(D_targets[idx]),
                    float(f_targets[idx]),
                ]
                if include_s8 and sigma8_series is not None and f_sigma8_series is not None:
                    row.append(float(sigma8_series[idx]))
                    row.append(float(f_sigma8_series[idx]))
                else:
                    row.extend([None, None])
                rows.append(row)
            tables.append(
                PredictionTable(
                    name="growth_vs_z",
                    columns=["z", "a", "D", "f", "sigma8_z", "f_sigma8"],
                    rows=rows,
                    metadata={"points": len(z_values)},
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="f_vs_z",
                    data={"z": z_values.tolist(), "f": f_targets.tolist()},
                    description="Growth rate f(z)",
                    metadata={"xlabel": "redshift z", "ylabel": "f(z)"},
                )
            )
            plots.append(
                PredictionPlot(
                    name="D_vs_z",
                    data={"z": z_values.tolist(), "D": D_targets.tolist()},
                    description="Growth factor D(z)",
                    metadata={"xlabel": "redshift z", "ylabel": "D(z)"},
                )
            )
            if include_s8 and f_sigma8_series is not None:
                plots.append(
                    PredictionPlot(
                        name="fs8_vs_z",
                        data={"z": z_values.tolist(), "f_sigma8": f_sigma8_series},
                        description="fσ₈(z)",
                        metadata={"xlabel": "redshift z", "ylabel": "f σ₈(z)"},
                    )
                )

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "zmax": float(np.max(z_values)),
            "points": len(z_values),
            "include_s8": include_s8,
            "zgrid": zgrid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
