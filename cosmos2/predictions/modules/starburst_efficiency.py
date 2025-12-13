"""Starburst efficiency prediction combining collapse and enrichment boosts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable


@register_prediction
class StarburstEfficiencyPrediction(PredictionModule):
    name = "starburst-efficiency"
    version = "v1"
    description = "Predicts collapse+metallicity driven starburst efficiency amplification."

    _DEFAULT_ZMAX = 20.0
    _DEFAULT_POINTS = 300
    _DEFAULT_GAMMA = 1.0
    _MIN_INTEGRATION_STEPS = 512
    _MIN_EPSILON = 1e-12
    _TARGET_RED_SHIFTS = (6.0, 10.0, 15.0)

    def register(self, parser: argparse.ArgumentParser) -> None:  # pragma: no cover - argparse wiring
        parser.add_argument(
            "--zmax",
            type=float,
            default=self._DEFAULT_ZMAX,
            help="Maximum redshift for the starburst scan (default: 20).",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=self._DEFAULT_POINTS,
            help="Number of redshift samples for the prediction (default: 300).",
        )
        parser.add_argument(
            "--gamma-metallicity",
            type=float,
            default=self._DEFAULT_GAMMA,
            help="Metallicity amplification coefficient γ (default: 1.0).",
        )
        parser.add_argument(
            "--beta-lcdm",
            type=float,
            help="Optional β value for the ΛCDM reference (defaults to γ when omitted).",
        )
        parser.add_argument(
            "--compare-lcdm",
            action="store_true",
            help="Include a ΛCDM reference and ratio in the output.",
        )
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Emit the canonical S(z) and ratio plot descriptors.",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Export the S(z) table with collapse, metallicity, and comparison columns.",
        )
        super().register(parser)

    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, object]
    ) -> PredictionResult:
        zmax = float(config.get("zmax", self._DEFAULT_ZMAX))
        if zmax <= 0.0:
            raise ValueError("zmax must be positive.")
        points = max(int(config.get("points", self._DEFAULT_POINTS)), 2)
        gamma = float(config.get("gamma_metallicity", self._DEFAULT_GAMMA))
        beta = config.get("beta_lcdm")
        beta = float(beta) if beta is not None else gamma
        compare_lcdm = bool(config.get("compare_lcdm"))
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        z_grid = np.linspace(0.0, zmax, num=points, dtype=float)
        a_grid = 1.0 / (1.0 + z_grid)

        try:
            epsilon_grid = model.elastic_stiffness(a_grid)
        except AttributeError as exc:
            return self._unsupported_prediction(model, str(exc))

        epsilon_grid = self._sanitize_epsilon(epsilon_grid)
        C_vals = np.power(epsilon_grid, -0.5)

        Z_rel_grid, a_integration = self._build_relative_metallicity(model, zmax, points, a_grid)

        S_lcdm_vals = None
        ratio_vals: np.ndarray | None = None
        if compare_lcdm:
            Z_rel_lcdm_grid = self._relative_lcdm_history(a_integration, a_grid)
            S_lcdm_vals = 1.0 + beta * Z_rel_lcdm_grid
            ratio_vals = np.divide(
                C_vals * (1.0 + gamma * Z_rel_grid),
                S_lcdm_vals,
                out=np.full_like(C_vals, np.nan),
                where=S_lcdm_vals > 0.0,
            )

        M_vals = 1.0 + gamma * Z_rel_grid
        S_vals = C_vals * M_vals

        peak_index = int(np.nanargmax(S_vals))
        peak_value = float(S_vals[peak_index])
        peak_redshift = float(z_grid[peak_index])

        samples = self._sample_targets(z_grid, S_vals, self._TARGET_RED_SHIFTS)

        results: dict[str, float] = {
            "zmax": float(zmax),
            "S_at_z6": samples.get(6.0, float(S_vals[0])),
            "S_at_z10": samples.get(10.0, float(S_vals[-1])),
            "S_at_z15": samples.get(15.0, float(S_vals[-1])),
            "peak_S_value": peak_value,
            "peak_S_redshift": peak_redshift,
        }

        if compare_lcdm and ratio_vals is not None:
            ratio_at_z6 = self._sample_series(z_grid, ratio_vals, 6.0)
            ratio_at_z10 = self._sample_series(z_grid, ratio_vals, 10.0)
            results["S_over_LCDM_at_z6"] = ratio_at_z6
            results["S_over_LCDM_at_z10"] = ratio_at_z10

        tables: list[PredictionTable] = []
        if output_table:
            tables.append(
                PredictionTable(
                    name="starburst_efficiency_vs_z",
                    columns=["z", "a", "epsilon0", "Z_rel", "C", "M", "S", "S_LCDM", "S_ratio"],
                    rows=self._build_table_rows(
                        z_grid,
                        a_grid,
                        epsilon_grid,
                        Z_rel_grid,
                        C_vals,
                        M_vals,
                        S_vals,
                        S_lcdm_vals,
                        ratio_vals,
                    ),
                    metadata={
                        "points": len(z_grid),
                        "zmax": float(zmax),
                        "compare_lcdm": compare_lcdm,
                        "integration_steps": max(self._MIN_INTEGRATION_STEPS, points * 4),
                    },
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            s_plot_data = {"z": z_grid.tolist(), "S": S_vals.tolist()}
            if compare_lcdm and S_lcdm_vals is not None:
                s_plot_data["S_LCDM"] = S_lcdm_vals.tolist()
            plots.append(
                PredictionPlot(
                    name="S_vs_z",
                    description="Starburst efficiency amplification across redshift",
                    data=s_plot_data,
                    metadata={"xlabel": "redshift z", "ylabel": "S(z)"},
                )
            )
            if compare_lcdm and ratio_vals is not None:
                plots.append(
                    PredictionPlot(
                        name="S_ratio_vs_z",
                        description="PBUF starburst efficiency versus ΛCDM reference",
                        data={"z": z_grid.tolist(), "S_ratio": ratio_vals.tolist()},
                        metadata={"xlabel": "redshift z", "ylabel": "S(z)/S_LCDM(z)"},
                    )
                )

        metadata = {
            "model": type(model.raw_model).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "zmax": float(zmax),
            "points": points,
            "compare_lcdm": compare_lcdm,
            "gamma_metallicity": gamma,
            "beta_lcdm": beta,
            "summary": "Predicted starburst efficiency amplification from elastic-enhanced collapse and metallicity.",
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )

    # -- Helpers -----------------------------------
    def _unsupported_prediction(self, model: PredictionModelAdapter, exc_message: str) -> PredictionResult:
        metadata = {
            "model": type(model.raw_model).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "missing_stiffness_or_metallicity_api",
            "summary": "Starburst efficiency prediction unsupported (missing stiffness or metallicity API).",
            "details": exc_message,
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

    def _build_relative_metallicity(
        self,
        model: PredictionModelAdapter,
        zmax: float,
        points: int,
        a_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        integration_steps = max(self._MIN_INTEGRATION_STEPS, points * 4)
        a_min = 1.0 / (1.0 + zmax)
        a_integration = np.linspace(a_min, 1.0, num=integration_steps, dtype=float)

        epsilon_integration = self._sanitize_epsilon(model.elastic_stiffness(a_integration))
        star_eff = self._evaluate_star_efficiency(model, a_integration)
        integrand = star_eff / epsilon_integration
        cumulative = self._cumulative_trapezoid(integrand, a_integration)
        total = cumulative[-1]
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Metallicity integral failed to converge for the starburst scan.")
        Z_rel_integration = cumulative / total
        Z_rel_grid = np.interp(a_grid, a_integration, Z_rel_integration)
        return Z_rel_grid, a_integration

    @classmethod
    def _relative_lcdm_history(cls, a_integration: np.ndarray, a_grid: np.ndarray) -> np.ndarray:
        cumulative = cls._cumulative_trapezoid(a_integration, a_integration)
        total = cumulative[-1]
        if total <= 0.0:
            raise ValueError("LCDM reference integral failed to converge.")
        Z_rel = cumulative / total
        return np.interp(a_grid, a_integration, Z_rel)

    @staticmethod
    def _evaluate_star_efficiency(model: PredictionModelAdapter, a_values: Sequence[float]) -> np.ndarray:
        func = getattr(model.raw_model, "star_formation_efficiency", None)
        arr = np.asarray(a_values, dtype=float)
        if not callable(func):
            return np.ones_like(arr, dtype=float)
        values = np.empty_like(arr, dtype=float)
        for idx, a_val in enumerate(arr):
            values[idx] = float(func(float(a_val)))
        return values

    @staticmethod
    def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        if y.size == 0:
            return np.array([], dtype=float)
        result = np.empty_like(y, dtype=float)
        result[0] = 0.0
        result[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
        return result

    @staticmethod
    def _sample_targets(z_grid: np.ndarray, values: np.ndarray, targets: Sequence[float]) -> dict[float, float]:
        result: dict[float, float] = {}
        for target in targets:
            interp = StarburstEfficiencyPrediction._sample_series(z_grid, values, target)
            result[target] = interp
        return result

    @staticmethod
    def _sample_series(z_grid: np.ndarray, values: np.ndarray, target: float) -> float:
        return float(np.interp(target, z_grid, values, left=values[0], right=values[-1]))

    @staticmethod
    def _build_table_rows(
        z_grid: np.ndarray,
        a_grid: np.ndarray,
        epsilon: np.ndarray,
        Z_rel: np.ndarray,
        C_vals: np.ndarray,
        M_vals: np.ndarray,
        S_vals: np.ndarray,
        S_lcdm_vals: np.ndarray | None,
        ratio_vals: np.ndarray | None,
    ) -> list[list[float | None]]:
        rows: list[list[float | None]] = []
        for idx in range(len(z_grid)):
            s_lcdm = S_lcdm_vals[idx] if S_lcdm_vals is not None else None
            ratio = ratio_vals[idx] if ratio_vals is not None else None
            rows.append(
                [
                    float(z_grid[idx]),
                    float(a_grid[idx]),
                    StarburstEfficiencyPrediction._safe_float(epsilon[idx]),
                    StarburstEfficiencyPrediction._safe_float(Z_rel[idx]),
                    StarburstEfficiencyPrediction._safe_float(C_vals[idx]),
                    StarburstEfficiencyPrediction._safe_float(M_vals[idx]),
                    StarburstEfficiencyPrediction._safe_float(S_vals[idx]),
                    StarburstEfficiencyPrediction._safe_float(s_lcdm),
                    StarburstEfficiencyPrediction._safe_float(ratio),
                ]
            )
        return rows

    @staticmethod
    def _safe_float(value: float | np.floating | None) -> float | None:
        if value is None:
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    @staticmethod
    def _sanitize_epsilon(values: np.ndarray) -> np.ndarray:
        sanitized = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(sanitized, StarburstEfficiencyPrediction._MIN_EPSILON, None)
