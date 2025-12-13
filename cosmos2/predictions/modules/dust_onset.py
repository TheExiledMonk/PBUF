"""Predict when dust formation becomes efficient."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from ..structures import PredictionPlot, PredictionResult, PredictionTable
from ..registry import PredictionModule, register_prediction

if TYPE_CHECKING:
    from ..model_api import PredictionModelAdapter


class _DustOnsetRequirementsError(Exception):
    """Indicates the selected model cannot support the dust-onset proxy."""


@register_prediction
class DustOnsetPrediction(PredictionModule):
    name = "dust-onset"
    version = "v1"
    description = "Predicts the redshift where elastic-enhanced enrichment yields dust."

    _DEFAULT_ZMAX = 30.0
    _DEFAULT_POINTS = 300
    _DEFAULT_THRESHOLD = 0.05
    _MIN_EPSILON = 1e-6
    _MIN_INTEGRATION_STEPS = 512
    _TARGET_RED_SHIFTS = (6.0, 10.0, 15.0)

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmax",
            type=float,
            default=self._DEFAULT_ZMAX,
            help="Maximum redshift to include in the normalized dust potential grid (default: 30).",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=self._DEFAULT_POINTS,
            help="Number of grid points for the dust potential scan (default: 300).",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="simple",
            help='Computation mode (currently only "simple" is supported).',
        )
        parser.add_argument(
            "--threshold-fraction",
            type=float,
            default=self._DEFAULT_THRESHOLD,
            help="Fraction of the total dust potential that must accumulate before dust appears (default: 0.05).",
        )
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Return the normalized dust potential vs redshift plot descriptor.",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Return the normalized dust potential table.",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        try:
            payload = self._compute_prediction(model, config)
        except _DustOnsetRequirementsError as exc:
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata={
                    "error": "missing_elastic_stiffness_api",
                    "summary": "Dust-onset prediction not supported for this model.",
                },
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                [float(z), float(a), float(eps), float(d_eff), float(p_norm)]
                for z, a, eps, d_eff, p_norm in zip(
                    payload["z_grid"],
                    payload["a_grid"],
                    payload["epsilon0_grid"],
                    payload["D_eff_grid"],
                    payload["P_norm_grid"],
                )
            ]
            tables.append(
                PredictionTable(
                    name="dust_potential_vs_z",
                    columns=["z", "a", "epsilon0", "D_eff", "P_norm"],
                    rows=rows,
                    metadata={
                        "mode": payload["mode"],
                        "points": payload["points"],
                        "threshold_fraction": payload["threshold_fraction"],
                        "zmax": payload["zmax"],
                    },
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="P_norm_vs_z",
                    description="Normalized cumulative dust potential versus redshift",
                    data={"z": payload["z_grid"], "P_norm": payload["P_norm_grid"]},
                    metadata={
                        "xlabel": "redshift z",
                        "ylabel": "normalized dust potential P_norm(z)",
                    },
                )
            )

        results = {
            "z_dust_on": payload["z_dust_on"],
            "a_dust_on": payload["a_dust_on"],
            "threshold_fraction": payload["threshold_fraction"],
            "P_norm_at_z6": payload["P_norm_targets"].get(6.0),
            "P_norm_at_z10": payload["P_norm_targets"].get(10.0),
            "P_norm_at_z15": payload["P_norm_targets"].get(15.0),
        }

        summary = f"Elastic-enhanced dust potential crosses {payload['threshold_fraction']:.3f}"
        if payload["z_dust_on"] is not None:
            summary += f" near z≈{payload['z_dust_on']:.2f}."
        else:
            summary += f" after z<{payload['zmax']:.1f}."

        metadata = {
            "model": payload["model_name"],
            "points": payload["points"],
            "threshold_fraction": payload["threshold_fraction"],
            "mode": payload["mode"],
            "timestamp": payload["timestamp"],
            "summary": summary,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )

    def _compute_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> dict[str, object]:
        zmax = float(config.get("zmax", self._DEFAULT_ZMAX))
        if zmax <= 0.0:
            raise ValueError("zmax must be positive.")
        points = int(config.get("points", self._DEFAULT_POINTS))
        if points < 3:
            raise ValueError("The dust scan requires at least 3 points.")
        mode = str(config.get("mode", "simple")).strip().lower()
        if mode != "simple":
            raise ValueError(f'Mode "{mode}" is not supported.')
        threshold_fraction = float(config.get("threshold_fraction", self._DEFAULT_THRESHOLD))
        if not (0.0 < threshold_fraction <= 1.0):
            raise ValueError("threshold_fraction must be in (0, 1].")

        z_grid = np.linspace(0.0, zmax, points, dtype=float)
        a_grid = 1.0 / (1.0 + z_grid)
        a_min = 1.0 / (1.0 + zmax)
        integration_steps = max(points * 2, self._MIN_INTEGRATION_STEPS)
        a_integration = np.linspace(a_min, 1.0, integration_steps, dtype=float)

        try:
            epsilon_integration = model.elastic_stiffness(a_integration)
            epsilon_grid = model.elastic_stiffness(a_grid)
        except AttributeError as exc:
            raise _DustOnsetRequirementsError(str(exc)) from exc

        star_eff_func = getattr(model.raw_model, "star_formation_efficiency", None)
        star_integration = self._evaluate_maybe_scalar(star_eff_func, a_integration)
        star_grid = self._evaluate_maybe_scalar(star_eff_func, a_grid)

        clipped_integration = np.clip(epsilon_integration, self._MIN_EPSILON, None)
        clipped_grid = np.clip(epsilon_grid, self._MIN_EPSILON, None)

        D_integration = (1.0 / clipped_integration) * star_integration
        D_grid = (1.0 / clipped_grid) * star_grid

        P_integration = self._cumulative_integral(a_integration, D_integration)
        total_potential = float(P_integration[-1])
        if not np.isfinite(total_potential) or total_potential <= 0.0:
            raise ValueError("Dust potential integral failed to converge.")
        P_norm_integration = P_integration / total_potential

        P_norm_grid = np.interp(a_grid, a_integration, P_norm_integration, left=0.0, right=1.0)

        threshold_idx = np.argmax(P_norm_integration >= threshold_fraction)
        if threshold_idx >= len(P_norm_integration) or P_norm_integration[threshold_idx] < threshold_fraction:
            a_dust = None
            z_dust = None
        else:
            a_dust = float(a_integration[threshold_idx])
            z_dust = float((1.0 / a_dust) - 1.0)

        target_values = self._evaluate_targets(a_integration, P_norm_integration, self._TARGET_RED_SHIFTS)

        payload = {
            "z_grid": z_grid.tolist(),
            "a_grid": a_grid.tolist(),
            "epsilon0_grid": epsilon_grid.tolist(),
            "D_eff_grid": D_grid.tolist(),
            "P_norm_grid": P_norm_grid.tolist(),
            "zmax": zmax,
            "points": points,
            "mode": mode,
            "threshold_fraction": threshold_fraction,
            "model_name": self._model_name(model),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "z_dust_on": z_dust,
            "a_dust_on": a_dust,
            "P_norm_targets": target_values,
        }
        return payload

    @staticmethod
    def _evaluate_maybe_scalar(func: Callable[[float], float] | None, a_values: Sequence[float]) -> np.ndarray:
        if not callable(func):
            return np.ones(len(a_values), dtype=float)
        values = np.empty(len(a_values), dtype=float)
        for idx, a_val in enumerate(a_values):
            values[idx] = float(func(float(a_val)))
        return values

    @staticmethod
    def _cumulative_integral(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
        arr_x = np.asarray(x, dtype=float)
        arr_y = np.asarray(y, dtype=float)
        if arr_x.size < 2:
            return np.array(arr_y, dtype=float)
        increments = 0.5 * (arr_y[:-1] + arr_y[1:]) * np.diff(arr_x)
        result = np.empty_like(arr_x)
        result[0] = 0.0
        result[1:] = np.cumsum(increments)
        return result

    @staticmethod
    def _evaluate_targets(
        a_grid: Sequence[float], P_norm: Sequence[float], targets: Sequence[float]
    ) -> dict[float, float]:
        a_values = 1.0 / (1.0 + np.asarray(targets, dtype=float))
        values = np.interp(a_values, a_grid, P_norm, left=0.0, right=1.0)
        return {float(target): float(val) for target, val in zip(targets, values)}

    @staticmethod
    def _model_name(model: "PredictionModelAdapter") -> str:
        raw = model.raw_model
        return getattr(raw, "__class__", type(raw)).__name__
