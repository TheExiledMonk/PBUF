"""Predictions for wave-speed modulation in the elastic PBUF spacetime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from cosmos2.parameters.central_authority import MPC_TO_KM

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

SECONDS_PER_GYR = 3.15576e16  # seconds per gigayear


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute a zero-based cumulative trapezoidal integral."""

    if y.size == 0:
        return y.copy()
    cumulative = np.zeros_like(y, dtype=float)
    for idx in range(1, y.size):
        dz = x[idx] - x[idx - 1]
        cumulative[idx] = cumulative[idx - 1] + 0.5 * dz * (y[idx] + y[idx - 1])
    return cumulative


@register_prediction
class WaveSpeedPrediction(PredictionModule):
    name = "wave-speed"
    version = "v1"
    description = "Effective wave-propagation speed from elastic spacetime stiffness."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift for sampling.")
        parser.add_argument("--zmax", type=float, default=25.0, help="Maximum redshift for sampling.")
        parser.add_argument("--points", type=int, default=300, help="Number of sampling points in redshift.")
        parser.add_argument("--output-plot", action="store_true", help="Emit canonical plots.")
        parser.add_argument("--output-table", action="store_true", help="Export canonical tables.")
        parser.add_argument(
            "--include-delay",
            action="store_true",
            help="Compute cumulative propagation delay relative to constant-c light travel.",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        zmin = float(config.get("zmin", 0.0))
        zmax = float(config.get("zmax", 25.0))
        if zmax <= zmin:
            raise ValueError("zmax must be greater than zmin.")
        points = max(int(config.get("points", 300)), 2)
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))
        include_delay = bool(config.get("include_delay"))

        z_grid = np.linspace(zmin, zmax, points, dtype=float)
        a_grid = 1.0 / (1.0 + z_grid)
        temperature = model.temperature(a_grid)
        try:
            epsilon = model.elastic_stiffness(a_grid)
        except AttributeError:
            metadata = {
                "model": model.raw_model.__class__.__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "missing_elastic_stiffness_api",
            }
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata=metadata,
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        epsilon = np.nan_to_num(epsilon, nan=0.0, posinf=0.0, neginf=0.0)
        epsilon = np.clip(epsilon, 0.0, None)
        ceff_ratio = epsilon  # c_eff / c since c_eff = c * epsilon

        valid_min = float(np.nanmin(ceff_ratio))
        valid_max = float(np.nanmax(ceff_ratio))

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                [float(z), float(a), float(T), float(eps), float(ratio)]
                for z, a, T, eps, ratio in zip(z_grid, a_grid, temperature, epsilon, ceff_ratio)
            ]
            tables.append(
                PredictionTable(
                    name="wave_speed_vs_z",
                    columns=["z", "a", "T", "epsilon0", "c_eff_over_c"],
                    rows=rows,
                    metadata={"points": len(z_grid)},
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="c_eff_over_c_vs_z",
                    description="Elastic-wave speed ratio vs. redshift",
                    data={"z": z_grid.tolist(), "c_eff_over_c": ceff_ratio.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "c_eff(z)/c"},
                )
            )

        max_delay_gyr: float | None = None

        if include_delay:
            z_delay = np.linspace(0.0, zmax, points, dtype=float)
            a_delay = 1.0 / (1.0 + z_delay)
            epsilon_delay = model.elastic_stiffness(a_delay)
            epsilon_delay = np.nan_to_num(epsilon_delay, nan=0.0, posinf=0.0, neginf=0.0)
            epsilon_delay = np.clip(epsilon_delay, 1e-12, None)
            H_delay = model.H(a_delay)
            H_si = np.clip(H_delay, 1e-30, None) / MPC_TO_KM
            integrand = (1.0 / epsilon_delay - 1.0) / H_si
            delay_seconds = _cumulative_trapezoid(integrand, z_delay)
            delay_gyr = delay_seconds / SECONDS_PER_GYR
            max_delay_gyr = float(delay_gyr[-1])
            if output_table:
                delay_rows = [
                    [float(z), float(val)]
                    for z, val in zip(z_delay, delay_gyr)
                ]
                tables.append(
                    PredictionTable(
                        name="propagation_delay_vs_z",
                        columns=["z", "delta_t"],
                        rows=delay_rows,
                        metadata={"units": "Gyr", "points": len(z_delay)},
                    )
                )
            if output_plot:
                delay_plot = PredictionPlot(
                    name="delta_t_vs_z",
                    description="Cumulative extra travel time vs. redshift",
                    data={"z": z_delay.tolist(), "delta_t_Gyr": delay_gyr.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "extra travel time (Gyr)"},
                )
                plots.append(delay_plot)

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolution": points,
            "zmin": zmin,
            "zmax": zmax,
            "include_delay": include_delay,
        }
        if include_delay:
            metadata["delay_grid_points"] = points

        results = {
            "zmin": zmin,
            "zmax": zmax,
            "c_eff_over_c_min": valid_min,
            "c_eff_over_c_max": valid_max,
        }
        if include_delay and max_delay_gyr is not None:
            results["max_delay_Gyr"] = max_delay_gyr

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
