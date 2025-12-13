"""Predict the redshift when cosmic expansion transitions to acceleration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from ..structures import PredictionPlot, PredictionResult, PredictionTable
from ..registry import PredictionModule, register_prediction

if TYPE_CHECKING:
    from ..model_api import PredictionModelAdapter


@register_prediction
class AccelerationOnsetPrediction(PredictionModule):
    """Report when the deceleration parameter q(z) crosses zero."""

    name = "acceleration-onset"
    version = "v1"
    description = "Redshift where expansion switches from deceleration to acceleration."

    _DEFAULT_ZMAX = 5.0
    _DEFAULT_POINTS = 400
    _MIN_POINTS = 5
    _Q_OFFSET_REL = 1e-4
    _Q_OFFSET_MIN = 1e-6

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmax",
            type=float,
            default=self._DEFAULT_ZMAX,
            help=f"Maximum redshift for the q(z) grid (default: {self._DEFAULT_ZMAX}).",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=self._DEFAULT_POINTS,
            help=f"Number of grid points between z=0 and zmax (default: {self._DEFAULT_POINTS}).",
        )
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Emit the q(z) plot descriptor.",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Emit the q(z) table.",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        zmax = float(config.get("zmax", self._DEFAULT_ZMAX))
        if zmax <= 0.0:
            raise ValueError("zmax must be positive.")
        points = int(config.get("points", self._DEFAULT_POINTS))
        if points < self._MIN_POINTS:
            raise ValueError(f"At least {self._MIN_POINTS} points are required.")
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        z_grid = np.linspace(0.0, zmax, points, dtype=float)
        a_grid = 1.0 / (1.0 + z_grid)

        try:
            H_grid = np.asarray(model.H(a_grid), dtype=float)
        except AttributeError as exc:
            return self._missing_h_result(model, str(exc))

        deriv = np.gradient(H_grid, a_grid, edge_order=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            q_grid = -1.0 - (a_grid / np.where(H_grid == 0.0, np.nan, H_grid)) * deriv
        q_grid = np.nan_to_num(q_grid, nan=0.0, posinf=0.0, neginf=0.0)

        q_min_idx = int(np.nanargmin(q_grid)) if q_grid.size > 0 else 0
        q_min = float(q_grid[q_min_idx]) if q_grid.size > 0 else 0.0
        z_qmin = float(z_grid[q_min_idx]) if z_grid.size > 0 else 0.0
        q0 = float(q_grid[0]) if q_grid.size > 0 else 0.0

        z_accel = None
        a_accel = None
        z_slowdown = None
        has_slowdown = False
        accel_index = None

        for idx in range(len(z_grid) - 1):
            q_lo = q_grid[idx]
            q_hi = q_grid[idx + 1]
            if q_lo <= 0.0 <= q_hi and (q_hi > 0.0 or q_lo < 0.0):
                accel_index = idx
                break

        if accel_index is not None:
            z_low = float(z_grid[accel_index])
            z_high = float(z_grid[accel_index + 1])
            q_low = float(q_grid[accel_index])
            q_high = float(q_grid[accel_index + 1])

            if q_low == 0.0:
                z_accel = z_low
            elif q_high == 0.0:
                z_accel = z_high
            else:
                try:
                    z_accel = self._bisect_q_zero(model, z_low, z_high, q_low, q_high)
                except AttributeError as exc:
                    return self._missing_h_result(model, str(exc))
            a_accel = 1.0 / (1.0 + z_accel)

            for idx in range(accel_index + 1, len(z_grid) - 1):
                q_lo = q_grid[idx]
                q_hi = q_grid[idx + 1]
                if q_lo <= 0.0 <= q_hi and (q_hi > 0.0 or q_lo < 0.0):
                    z_low = float(z_grid[idx])
                    z_high = float(z_grid[idx + 1])
                    q_low = float(q_grid[idx])
                    q_high = float(q_grid[idx + 1])
                    if q_low == 0.0:
                        z_slowdown = z_low
                    elif q_high == 0.0:
                        z_slowdown = z_high
                    else:
                        try:
                            z_slowdown = self._bisect_q_zero(model, z_low, z_high, q_low, q_high)
                        except AttributeError as exc:
                            return self._missing_h_result(model, str(exc))
                    has_slowdown = True
                    break

        rows = []
        if output_table:
            rows = [
                [float(z), float(a), float(H), float(q)]
                for z, a, H, q in zip(z_grid, a_grid, H_grid, q_grid)
            ]
        tables: list[PredictionTable] = []
        if output_table:
            tables.append(
                PredictionTable(
                    name="q_vs_z",
                    columns=["z", "a", "H", "q"],
                    rows=rows,
                    metadata={"zmax": float(zmax), "points": points},
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="q_vs_z_plot",
                    description="Deceleration parameter vs redshift",
                    data={"z": z_grid.tolist(), "q": q_grid.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "deceleration parameter q(z)"},
                )
            )

        summary = "Deceleration parameter crosses zero"
        if z_accel is not None:
            summary += f" near z≈{z_accel:.3f}."
        else:
            summary += " within the sampled redshift range."

        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {
            "model": self._model_name(model),
            "zmax": float(zmax),
            "points": points,
            "timestamp": timestamp,
            "summary": summary,
        }

        results = {
            "zmax": float(zmax),
            "z_accel": float(z_accel) if z_accel is not None else None,
            "a_accel": float(a_accel) if a_accel is not None else None,
            "q0": q0,
            "q_min": q_min,
            "z_qmin": z_qmin,
            "has_slowdown": has_slowdown,
            "z_slowdown": float(z_slowdown) if z_slowdown is not None else None,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )

    def _bisect_q_zero(
        self,
        model: "PredictionModelAdapter",
        z_low: float,
        z_high: float,
        q_low: float,
        q_high: float,
    ) -> float:
        """Bisection on q(z) between two redshifts that bracket zero."""

        if q_low > 0.0 or q_high < 0.0:
            raise ValueError("q(z) must bracket zero for bisection.")

        z_left = z_low
        z_right = z_high
        q_left = q_low
        q_right = q_high
        for _ in range(60):
            z_mid = 0.5 * (z_left + z_right)
            q_mid = self._q_at_z(model, z_mid)
            if abs(q_mid) < 1e-12:
                return z_mid
            if q_mid > 0.0:
                z_right = z_mid
                q_right = q_mid
            else:
                z_left = z_mid
                q_left = q_mid
            if abs(z_right - z_left) < 1e-7:
                break
        return 0.5 * (z_left + z_right)

    def _q_at_z(self, model: "PredictionModelAdapter", z_value: float) -> float:
        a_value = 1.0 / (1.0 + z_value)
        return self._q_at_a(model, a_value)

    def _q_at_a(self, model: "PredictionModelAdapter", a_value: float) -> float:
        a_clamped = max(float(a_value), 1e-12)
        delta = max(a_clamped * self._Q_OFFSET_REL, self._Q_OFFSET_MIN)
        a_minus = max(a_clamped - delta, 1e-9)
        a_plus = a_clamped + delta
        arr = np.array([a_minus, a_clamped, a_plus], dtype=float)
        H_vals = np.asarray(model.H(arr), dtype=float)
        H_minus, H_center, H_plus = H_vals[0], H_vals[1], H_vals[2]
        span = a_plus - a_minus
        derivative = (H_plus - H_minus) / span if span != 0.0 else 0.0
        if H_center == 0.0:
            return -1.0
        q_val = -1.0 - (a_clamped / H_center) * derivative
        return float(np.nan_to_num(q_val, nan=0.0, posinf=0.0, neginf=0.0))

    def _missing_h_result(self, model: "PredictionModelAdapter", reason: str) -> PredictionResult:
        return PredictionResult(
            name=self.name,
            version=self.version,
            status="error",
            metadata={
                "error": "missing_H_api",
                "summary": "Acceleration-onset prediction not supported (missing H(z)/H(a) API).",
                "model": self._model_name(model),
                "reason": reason,
            },
            results={},
            tables=[],
            plots=[],
        )

    def _model_name(self, model: "PredictionModelAdapter") -> str:
        raw = getattr(model.raw_model, "name", None)
        if isinstance(raw, str) and raw:
            return raw
        return model.raw_model.__class__.__name__
