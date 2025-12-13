"""Predict the Sandage–Loeb redshift drift signal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from cosmos2.models.lcdm.utils import C_LIGHT

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

SECONDS_PER_YEAR = 31_557_600.0
KM_PER_MPC = 3.085677581491367e19
KM_TO_CM = 1e5
DEFAULT_ZMIN = 0.0
DEFAULT_ZMAX = 6.0
DEFAULT_POINTS = 300
SAMPLE_ZS = (2.0, 3.0, 4.0)


def _build_z_grid(zmin: float, zmax: float, points: int) -> np.ndarray:
    if zmax <= zmin:
        raise ValueError("zmax must be greater than zmin.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    return np.linspace(float(zmin), float(zmax), num=int(points), dtype=float)


def _interpolate_scalar(values: np.ndarray, grid: np.ndarray, target: float) -> float | None:
    if target < float(grid[0]) or target > float(grid[-1]):
        return None
    return float(np.interp(target, grid, values))


def _make_velocity_drift(dzdt0: np.ndarray, z: np.ndarray, delta_years: float) -> np.ndarray:
    """Return Δv in cm/s for the provided dz/dt0 series."""
    delta_seconds = float(delta_years) * SECONDS_PER_YEAR
    with np.errstate(invalid="ignore", divide="ignore"):
        kms = C_LIGHT * dzdt0 * delta_seconds / np.where(1.0 + z == 0.0, np.nan, (1.0 + z))
    kms = np.nan_to_num(kms, nan=0.0, posinf=0.0, neginf=0.0)
    return kms * KM_TO_CM


def _lcdm_hubble(
    z: np.ndarray,
    H0: float,
    Omega_m0: float,
    Omega_r0: float,
) -> np.ndarray:
    components = Omega_r0 * (1.0 + z) ** 4 + Omega_m0 * (1.0 + z) ** 3 + (1.0 - Omega_m0 - Omega_r0)
    safe = np.clip(components, 0.0, np.inf)
    return float(H0) * np.sqrt(safe)


@register_prediction
class RedshiftDriftPrediction(PredictionModule):
    name = "redshift-drift"
    version = "v1"
    description = "Sandage–Loeb redshift drift and ELT velocity drift."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=DEFAULT_ZMIN, help="Minimum redshift (default 0.0)")
        parser.add_argument("--zmax", type=float, default=DEFAULT_ZMAX, help="Maximum redshift (default 6.0)")
        parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="Number of redshift samples (default 300)")

        dt10_group = parser.add_mutually_exclusive_group()
        dt10_group.add_argument("--dt10", dest="dt10", action="store_true", help="Include 10 year velocity drift (default)")
        dt10_group.add_argument("--no-dt10", dest="dt10", action="store_false", help="Omit 10 year velocity drift")

        dt20_group = parser.add_mutually_exclusive_group()
        dt20_group.add_argument("--dt20", dest="dt20", action="store_true", help="Include 20 year velocity drift (default)")
        dt20_group.add_argument("--no-dt20", dest="dt20", action="store_false", help="Omit 20 year velocity drift")

        parser.add_argument("--compare-lcdm", action="store_true", help="Add ΛCDM reference using Ω_m0 and Ω_r0")
        parser.add_argument("--output-table", action="store_true", help="Export the redshift drift table")
        parser.add_argument("--output-plot", action="store_true", help="Include dz/dt0 and Δv plots in the payload")
        parser.set_defaults(dt10=True, dt20=True)
        super().register(parser)

    def run_prediction(self, model: PredictionModelAdapter, config: dict[str, object]) -> PredictionResult:
        zmin = float(config.get("zmin", DEFAULT_ZMIN))
        zmax = float(config.get("zmax", DEFAULT_ZMAX))
        points = int(config.get("points", DEFAULT_POINTS))
        compute_dt10 = bool(config.get("dt10", True))
        compute_dt20 = bool(config.get("dt20", True))
        compare_lcdm = bool(config.get("compare_lcdm", False))
        output_table = bool(config.get("output_table", False))
        output_plot = bool(config.get("output_plot", False))

        if zmin < 0.0:
            raise ValueError("zmin must be non-negative.")
        z_grid = _build_z_grid(zmin, zmax, points)

        try:
            a_grid = np.clip(1.0 / (1.0 + z_grid), 1e-12, np.inf)
            H_values = np.asarray(model.H(a_grid), dtype=float)
        except AttributeError:
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata={
                    "error": "missing_H_api",
                    "summary": "Redshift-drift prediction unsupported (missing H(z))",
                },
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        if H_values.shape != z_grid.shape:
            raise RuntimeError("Model H(z) grid size mismatch.")

        H0 = float(model.parameters.get("H0", 67.4))
        if H0 <= 0.0:
            raise ValueError("Model reports non-positive H0.")

        shifters = float(H0) * (1.0 + z_grid)
        dzdt0 = np.nan_to_num((shifters - H_values) / KM_PER_MPC, nan=0.0, posinf=0.0, neginf=0.0)

        dv10_values = _make_velocity_drift(dzdt0, z_grid, 10.0) if compute_dt10 else None
        dv20_values = _make_velocity_drift(dzdt0, z_grid, 20.0) if compute_dt20 else None

        dzdt0_lcdm: np.ndarray | None = None
        dv10_lcdm: np.ndarray | None = None
        if compare_lcdm:
            params = model.parameters
            Omega_m0 = float(params.get("Omega_m0", 0.315))
            Omega_r0 = float(params.get("Omega_r0", 9.0e-5))
            H_lcdm = _lcdm_hubble(z_grid, H0, Omega_m0, Omega_r0)
            dzdt0_lcdm = np.nan_to_num((shifters - H_lcdm) / KM_PER_MPC, nan=0.0, posinf=0.0, neginf=0.0)
            if compute_dt10 and dzdt0_lcdm is not None:
                dv10_lcdm = _make_velocity_drift(dzdt0_lcdm, z_grid, 10.0)

        tables: list[PredictionTable] = []
        if output_table:
            rows: list[list[float | None]] = []
            ratio_series: np.ndarray | None = None
            if compare_lcdm and dv10_values is not None and dv10_lcdm is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_series = np.divide(dv10_values, dv10_lcdm)
                ratio_series = np.where(np.isfinite(ratio_series), ratio_series, np.nan)
                ratio_series = np.where(dv10_lcdm == 0.0, np.nan, ratio_series)
            for idx, z_val in enumerate(z_grid):
                dv10_val = float(dv10_values[idx]) if dv10_values is not None else None
                dv20_val = float(dv20_values[idx]) if dv20_values is not None else None
                dzdt0_lcdm_val = float(dzdt0_lcdm[idx]) if dzdt0_lcdm is not None else None
                dv10_lcdm_val = float(dv10_lcdm[idx]) if dv10_lcdm is not None else None
                ratio_val = None
                if ratio_series is not None:
                    ratio_val = float(ratio_series[idx])
                    if not np.isfinite(ratio_val):
                        ratio_val = None
                rows.append(
                    [
                        float(z_val),
                        float(dzdt0[idx]),
                        dv10_val,
                        dv20_val,
                        dzdt0_lcdm_val,
                        dv10_lcdm_val,
                        ratio_val,
                    ]
                )
            tables.append(
                PredictionTable(
                    name="redshift_drift_vs_z",
                    columns=[
                        "z",
                        "dzdt0",
                        "dv10",
                        "dv20",
                        "dzdt0_LCDM",
                        "dv10_LCDM",
                        "ratio_dv",
                    ],
                    rows=rows,
                    metadata={
                        "points": len(z_grid),
                        "zmin": float(zmin),
                        "zmax": float(zmax),
                        "compare_lcdm": compare_lcdm,
                        "dt10": compute_dt10,
                        "dt20": compute_dt20,
                    },
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            dz_plot_data: dict[str, Iterable[float]] = {"z": z_grid.tolist(), "dzdt0": dzdt0.tolist()}
            if dzdt0_lcdm is not None:
                dz_plot_data["dzdt0_LCDM"] = dzdt0_lcdm.tolist()
            plots.append(
                PredictionPlot(
                    name="dzdt0_vs_z",
                    description="Redshift drift rate vs. redshift",
                    data=dz_plot_data,
                    metadata={"xlabel": "redshift z", "ylabel": "dz/dt0 [1/s]"},
                )
            )
            dv_plot_data: dict[str, Iterable[float]] = {"z": z_grid.tolist()}
            if dv10_values is not None:
                dv_plot_data["dv10"] = dv10_values.tolist()
            if dv20_values is not None:
                dv_plot_data["dv20"] = dv20_values.tolist()
            plots.append(
                PredictionPlot(
                    name="dv_vs_z",
                    description="Velocity drift for requested baselines",
                    data=dv_plot_data,
                    metadata={"xlabel": "redshift z", "ylabel": "Δv [cm/s]"},
                )
            )

        results: dict[str, float | None] = {
            "zmin": float(zmin),
            "zmax": float(zmax),
            "points": len(z_grid),
        }

        def _maybe_sample(series: np.ndarray | None, name: str, convert: bool = False) -> None:
            for z_target in SAMPLE_ZS:
                key = f"{name}_z{int(z_target)}"
                if series is None:
                    results[key] = None
                    continue
                value = _interpolate_scalar(series, z_grid, z_target)
                results[key] = float(value) if value is not None else None

        _maybe_sample(dzdt0, "dzdt0")
        if dv10_values is not None:
            results.update(
                {
                    f"dv10_z{int(z_target)}": _interpolate_scalar(dv10_values, z_grid, z_target)
                    for z_target in SAMPLE_ZS
                }
            )
        else:
            for z_target in SAMPLE_ZS:
                results[f"dv10_z{int(z_target)}"] = None
        if dv20_values is not None:
            results.update(
                {
                    f"dv20_z{int(z_target)}": _interpolate_scalar(dv20_values, z_grid, z_target)
                    for z_target in SAMPLE_ZS
                }
            )
        else:
            for z_target in SAMPLE_ZS:
                results[f"dv20_z{int(z_target)}"] = None

        if compare_lcdm:
            _maybe_sample(dzdt0_lcdm, "dzdt0_LCDM")
            if dv10_lcdm is not None:
                for z_target in SAMPLE_ZS:
                    key = f"ratio_dv10_z{int(z_target)}"
                    numerator = _interpolate_scalar(dv10_values, z_grid, z_target) if dv10_values is not None else None
                    denominator = _interpolate_scalar(dv10_lcdm, z_grid, z_target)
                    if numerator is None or denominator is None or denominator == 0.0:
                        results[key] = None
                        continue
                    results[key] = float(numerator / denominator)
            else:
                for z_target in SAMPLE_ZS:
                    results[f"ratio_dv10_z{int(z_target)}"] = None
        else:
            for z_target in SAMPLE_ZS:
                results[f"ratio_dv10_z{int(z_target)}"] = None

        metadata = {
            "model": type(model.raw_model).__name__,
            "compare_lcdm": compare_lcdm,
            "dt10": compute_dt10,
            "dt20": compute_dt20,
            "points": len(z_grid),
            "zmin": float(zmin),
            "zmax": float(zmax),
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
