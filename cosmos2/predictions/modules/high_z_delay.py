"""High-redshift propagation delay relative to a constant-c baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

from cosmos2.parameters.central_authority import MPC_TO_KM

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

SECONDS_PER_GYR = 3.15576e16  # seconds in a gigayear


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-based cumulative trapezoidal integral."""

    if y.size == 0:
        return y.copy()

    cumulative = np.zeros_like(y, dtype=float)
    for idx in range(1, y.size):
        dx = x[idx] - x[idx - 1]
        cumulative[idx] = cumulative[idx - 1] + 0.5 * dx * (y[idx] + y[idx - 1])
    return cumulative


def _parse_zgrid_value(value: str) -> list[float]:
    if not value:
        return []
    items: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(float(part))
    return items


def _format_z_label(z: float) -> str:
    if np.isclose(z, round(z)):
        return f"{int(round(z))}"
    return str(z).replace(".", "_")


@register_prediction
class HighZDelayPrediction(PredictionModule):
    name = "high-z-delay"
    version = "v1"
    description = "Extra propagation time for high-redshift signals versus a constant-c reference."

    _DEFAULT_ZMAX = 20.0
    _DEFAULT_POINTS = 400
    _DEFAULT_REFERENCE = "same-H-constant-c"
    _VALID_REFERENCES = (_DEFAULT_REFERENCE,)
    _DEFAULT_SUMMARY_ZGRID = (1.0, 3.0, 6.0, 10.0, 20.0)

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmax",
            type=float,
            default=self._DEFAULT_ZMAX,
            help="Maximum redshift to include in the travel-time scan (default: 20).",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=self._DEFAULT_POINTS,
            help="Number of redshift samples for the prediction (default: 400).",
        )
        parser.add_argument(
            "--reference",
            type=str,
            choices=self._VALID_REFERENCES,
            default=self._DEFAULT_REFERENCE,
            help="Reference baseline for the constant-c comparison (v1 only supports same-H).",
        )
        parser.add_argument(
            "--zgrid",
            type=str,
            default=None,
            help="Comma-separated redshifts for the summary values (default: 1,3,6,10,20).",
        )
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Emit canonical plots for Δt(z) and c_eff(z)/c",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Return the propagation delay table.",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        zmax = float(config.get("zmax", self._DEFAULT_ZMAX))
        if zmax <= 0.0:
            raise ValueError("zmax must be positive.")
        points = max(int(config.get("points", self._DEFAULT_POINTS)), 3)
        reference = str(config.get("reference", self._DEFAULT_REFERENCE))
        if reference not in self._VALID_REFERENCES:
            raise ValueError(f"Unknown reference '{reference}'.")
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        summary_zgrid = self._resolve_summary_zgrid(config.get("zgrid"))
        if summary_zgrid and summary_zgrid[-1] > zmax:
            raise ValueError("All zgrid entries must be at or below zmax.")

        z_grid = np.linspace(0.0, zmax, points, dtype=float)
        a_grid = 1.0 / (1.0 + z_grid)

        try:
            epsilon = model.elastic_stiffness(a_grid)
        except AttributeError as exc:
            metadata = {
                "model": type(model.raw_model).__name__,
                "timestamp": self._now(),
                "reference": reference,
                "error": "missing_wave_speed_api",
                "summary": "High-z propagation delay prediction not supported for this model (missing wave-speed or stiffness API).",
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
        epsilon = np.clip(epsilon, 1e-12, None)

        H_vals = model.H(a_grid)
        H_vals = np.nan_to_num(H_vals, nan=0.0, posinf=0.0, neginf=0.0)
        H_vals = np.clip(H_vals, 1e-30, None)
        H_si = H_vals / MPC_TO_KM

        integrand = (1.0 / epsilon - 1.0) / ((1.0 + z_grid) * H_si)
        delta_t_seconds = _cumulative_trapezoid(integrand, z_grid)
        delta_t_gyr = delta_t_seconds / SECONDS_PER_GYR

        ceff_ratio = epsilon
        delay_summary = self._build_delay_summary(z_grid, delta_t_gyr, summary_zgrid)

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                [
                    float(z),
                    float(1.0 / (1.0 + z)),
                    float(H),
                    float(ratio),
                    float(delta),
                ]
                for z, H, ratio, delta in zip(z_grid, H_vals, ceff_ratio, delta_t_gyr)
            ]
            tables.append(
                PredictionTable(
                    name="propagation_delay_vs_z",
                    columns=["z", "a", "H", "c_eff_over_c", "delta_t"],
                    rows=rows,
                    metadata={
                        "units": {"H": "km/s/Mpc", "delta_t": "Gyr"},
                        "points": len(z_grid),
                        "zmax": zmax,
                    },
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="delta_t_vs_z",
                    description="Extra travel time relative to constant-c propagation",
                    data={"z": z_grid.tolist(), "delta_t_Gyr": delta_t_gyr.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "extra travel time Δt(z) [Gyr]"},
                )
            )
            plots.append(
                PredictionPlot(
                    name="c_eff_over_c_vs_z",
                    description="Effective wave speed normalized to c",
                    data={"z": z_grid.tolist(), "c_eff_over_c": ceff_ratio.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "c_eff(z)/c"},
                )
            )

        results = {
            "zmax": zmax,
            "points": points,
            "max_delay_Gyr": float(delta_t_gyr[-1]),
            **delay_summary,
        }

        metadata = {
            "model": type(model.raw_model).__name__,
            "timestamp": self._now(),
            "zmax": zmax,
            "points": points,
            "reference": reference,
            "zgrid": list(summary_zgrid),
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )

    def _resolve_summary_zgrid(self, raw: object | None) -> tuple[float, ...]:
        if raw is None:
            return tuple(self._DEFAULT_SUMMARY_ZGRID)
        if isinstance(raw, str):
            values = _parse_zgrid_value(raw)
        elif isinstance(raw, Iterable):
            values = [float(item) for item in raw]
        else:
            raise ValueError("Invalid zgrid specification.")
        cleaned = tuple(sorted(set(float(val) for val in values if val >= 0.0)))
        return cleaned or tuple(self._DEFAULT_SUMMARY_ZGRID)

    def _build_delay_summary(self, z_grid: np.ndarray, delta_t_gyr: np.ndarray, summary_z: Sequence[float]) -> dict[str, float]:
        if not summary_z:
            return {}
        interpolated = np.interp(summary_z, z_grid, delta_t_gyr, right=float(delta_t_gyr[-1]))
        return {
            f"delay_at_z{_format_z_label(z)}_Gyr": float(delay)
            for z, delay in zip(summary_z, interpolated)
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
