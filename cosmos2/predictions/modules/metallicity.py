"""Predict relative metallicity evolution using elastic-enhanced enrichment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Callable, Sequence

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable


class _MetallicityRequirementsError(Exception):
    """Signals that the selected model cannot support metallicity enrichment."""


@register_prediction
class MetallicityPrediction(PredictionModule):
    name = "metallicity"
    version = "v1"
    description = "Predicts the metallicity growth history driven by elastic stiffness."

    _TARGET_RED_SHIFTS = (2.0, 6.0, 10.0)
    _MIN_ENRICH_STEPS = 512
    _MIN_EPSILON = 1e-6
    _LCDM_BETA = 1.0

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmin",
            type=float,
            default=2.0,
            help="Lowest redshift for the output grid (default: 2.0).",
        )
        parser.add_argument(
            "--zmax",
            type=float,
            default=20.0,
            help="Highest redshift for the output grid (default: 20.0).",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=200,
            help="Number of redshift steps (default: 200).",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="simple",
            help='Enrichment mode (currently only "simple" is supported).',
        )
        parser.add_argument(
            "--compare-lcdm",
            action="store_true",
            help="Compute a ΛCDM-like reference enrichment curve for comparison.",
        )
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Return plot descriptors for the metallicity evolution and boost.",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Export metallicity values as a CSV-style table.",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        try:
            payload = self._compute_prediction(model, config)
        except _MetallicityRequirementsError as exc:
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata={"error": "missing_elastic_stiffness_api", "summary": str(exc)},
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                [float(z), float(a), float(eps), float(e_eff), float(z_rel)]
                for z, a, eps, e_eff, z_rel in zip(
                    payload["z_grid"],
                    payload["a_grid"],
                    payload["epsilon"],
                    payload["E_eff"],
                    payload["Z_rel"],
                )
            ]
            tables.append(
                PredictionTable(
                    name="metallicity_vs_z",
                    columns=["z", "a", "epsilon0", "E_eff", "Z_rel"],
                    rows=rows,
                    metadata={
                        "mode": payload["mode"],
                        "compare_lcdm": payload["compare_lcdm"],
                        "points": payload["points"],
                    },
                )
            )
            if payload["compare_lcdm"] and payload.get("Z_rel_lcdm") is not None:
                lcdm_rows = [
                    [float(z), float(val)]
                    for z, val in zip(payload["z_grid"], payload["Z_rel_lcdm"])
                ]
                tables.append(
                    PredictionTable(
                        name="metallicity_vs_z_lcdm",
                        columns=["z", "Z_rel_lcdm"],
                        rows=lcdm_rows,
                        metadata={"beta": self._LCDM_BETA},
                    )
                )

        plots: list[PredictionPlot] = []
        if output_plot:
            plot_data = {"z": payload["z_grid"], "Z_rel": payload["Z_rel"]}
            if payload["compare_lcdm"] and payload.get("Z_rel_lcdm") is not None:
                plot_data["Z_rel_lcdm"] = payload["Z_rel_lcdm"]
            plots.append(
                PredictionPlot(
                    name="Z_rel_vs_z",
                    description="Relative PBUF metallicity growth compared to today",
                    data=plot_data,
                    metadata={
                        "xlabel": "redshift z",
                        "ylabel": "relative metallicity Z(z) / Z(0)",
                    },
                )
            )
            if payload["compare_lcdm"] and payload.get("Z_boost") is not None:
                plots.append(
                    PredictionPlot(
                        name="Z_boost_vs_z",
                        description="PBUF metallicity boost relative to a simple ΛCDM curve",
                        data={"z": payload["z_grid"], "Z_boost": payload["Z_boost"]},
                        metadata={
                            "xlabel": "redshift z",
                            "ylabel": "metallicity boost (PBUF / ΛCDM)",
                        },
                    )
                )

        results: dict[str, object] = {
            "zmin": payload["zmin"],
            "zmax": payload["zmax"],
            "Z_over_Z0_at_z2": payload["Z_at_targets"].get(2.0),
            "Z_over_Z0_at_z6": payload["Z_at_targets"].get(6.0),
            "Z_over_Z0_at_z10": payload["Z_at_targets"].get(10.0),
            "boost_vs_lcdm_at_z6": payload["boosts"].get(6.0) if payload["compare_lcdm"] else None,
            "boost_vs_lcdm_at_z10": payload["boosts"].get(10.0) if payload["compare_lcdm"] else None,
            "z range": payload["z_range"],
            "Z(z=2) / Z(0)": payload["Z_at_targets"].get(2.0),
            "Z(z=6) / Z(0)": payload["Z_at_targets"].get(6.0),
            "Z(z=10) / Z(0)": payload["Z_at_targets"].get(10.0),
        }
        if payload["compare_lcdm"] and payload.get("boost_lines"):
            results["metallicity boost vs LCDM"] = payload["boost_lines"]

        metadata = {
            "model": payload["model_name"],
            "points": payload["points"],
            "mode": payload["mode"],
            "compare_lcdm": payload["compare_lcdm"],
            "timestamp": payload["timestamp"],
            "summary": "Predicted metallicity evolution Z(z) from elastic-enhanced enrichment",
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
        zmin = float(config.get("zmin", 2.0))
        zmax = float(config.get("zmax", 20.0))
        if zmin < 0.0 or zmax <= zmin:
            raise ValueError("Specify 0 <= zmin < zmax for the metallicity grid.")
        points = int(config.get("points", 200))
        if points < 3:
            raise ValueError("Enrichment predictions require at least 3 grid points.")
        mode = str(config.get("mode", "simple")).strip().lower()
        if mode != "simple":
            raise ValueError(f'Enrichment mode "{mode}" is not supported yet.')
        compare_lcdm = bool(config.get("compare_lcdm"))

        z_grid = np.linspace(zmin, zmax, points)
        a_grid = 1.0 / (1.0 + z_grid)

        integration_z_max = max(zmax, max(self._TARGET_RED_SHIFTS))
        a_min = 1.0 / (1.0 + integration_z_max)
        integration_steps = max(points, self._MIN_ENRICH_STEPS)
        a_integration = np.linspace(a_min, 1.0, integration_steps)

        try:
            epsilon_integration = model.elastic_stiffness(a_integration)
            epsilon_grid = model.elastic_stiffness(a_grid)
        except AttributeError as exc:
            raise _MetallicityRequirementsError(
                "Metallicity prediction not supported (missing stiffness/thermal API)."
            ) from exc

        star_eff_func = getattr(model.raw_model, "star_formation_efficiency", None)
        star_integration = self._evaluate_maybe_scalar(star_eff_func, a_integration)
        star_grid = self._evaluate_maybe_scalar(star_eff_func, a_grid)

        clipped_integration = np.clip(epsilon_integration, self._MIN_EPSILON, None)
        clipped_grid = np.clip(epsilon_grid, self._MIN_EPSILON, None)

        E_eff_integration = (1.0 / clipped_integration) * star_integration
        E_eff_grid = (1.0 / clipped_grid) * star_grid

        Z_integral = self._cumulative_integral(a_integration, E_eff_integration)
        Z_total = float(Z_integral[-1])
        if not np.isfinite(Z_total) or Z_total <= 0.0:
            raise _MetallicityRequirementsError("Enrichment integral failed to converge.")
        Z_rel_integration = Z_integral / Z_total

        Z_rel_grid = np.interp(a_grid, a_integration, Z_rel_integration)

        payload: dict[str, object] = {
            "z_grid": z_grid.tolist(),
            "a_grid": a_grid.tolist(),
            "epsilon": epsilon_grid.tolist(),
            "E_eff": E_eff_grid.tolist(),
            "Z_rel": Z_rel_grid.tolist(),
            "zmin": zmin,
            "zmax": zmax,
            "mode": mode,
            "points": points,
            "compare_lcdm": compare_lcdm,
            "z_range": f"{zmin:g} -> {zmax:g}",
            "model_name": model.raw_model.__class__.__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        payload["Z_at_targets"] = self._evaluate_targets(
            a_integration, Z_rel_integration, self._TARGET_RED_SHIFTS
        )

        if compare_lcdm:
            E_lcdm = a_integration ** self._LCDM_BETA
            Z_lcdm = self._cumulative_integral(a_integration, E_lcdm)
            Z_lcdm_total = float(Z_lcdm[-1])
            if not np.isfinite(Z_lcdm_total) or Z_lcdm_total <= 0.0:
                raise _MetallicityRequirementsError("Reference ΛCDM integral failed to converge.")
            Z_rel_lcdm = Z_lcdm / Z_lcdm_total
            payload["Z_rel_lcdm"] = np.interp(a_grid, a_integration, Z_rel_lcdm).tolist()
            payload["Z_boost"] = self._compute_boost(Z_rel_grid, payload["Z_rel_lcdm"])
            payload["boosts"] = self._evaluate_boost_targets(
                a_integration, Z_rel_integration, Z_rel_lcdm, self._TARGET_RED_SHIFTS
            )
            payload["boost_lines"] = self._build_boost_lines(payload["boosts"])
        else:
            payload["Z_rel_lcdm"] = None
            payload["Z_boost"] = None
            payload["boosts"] = {}
            payload["boost_lines"] = None

        return payload

    @staticmethod
    def _evaluate_maybe_scalar(
        func: Callable[[float], float] | None, a_values: Sequence[float]
    ) -> np.ndarray:
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
        a_grid: Sequence[float], Z_rel: Sequence[float], targets: Sequence[float]
    ) -> dict[float, float]:
        a_values = 1.0 / (1.0 + np.asarray(targets, dtype=float))
        Z_values = np.interp(a_values, a_grid, Z_rel)
        return {float(target): float(value) for target, value in zip(targets, Z_values)}

    @staticmethod
    def _compute_boost(Z_rel: Sequence[float], Z_rel_lcdm: Sequence[float]) -> list[float] | None:
        if Z_rel_lcdm is None:
            return None
        arr_rel = np.asarray(Z_rel, dtype=float)
        arr_lcdm = np.asarray(Z_rel_lcdm, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            boost = np.divide(arr_rel, arr_lcdm, out=np.full_like(arr_rel, np.nan), where=arr_lcdm > 0)
        return boost.tolist()

    @staticmethod
    def _evaluate_boost_targets(
        a_grid: Sequence[float],
        Z_rel: Sequence[float],
        Z_rel_lcdm: Sequence[float],
        targets: Sequence[float],
    ) -> dict[float, float]:
        a_values = 1.0 / (1.0 + np.asarray(targets, dtype=float))
        base = np.interp(a_values, a_grid, Z_rel)
        reference = np.interp(a_values, a_grid, Z_rel_lcdm)
        result: dict[float, float] = {}
        for target, pbuf_val, lcdm_val in zip(targets, base, reference):
            if lcdm_val and np.isfinite(lcdm_val):
                result[float(target)] = float(pbuf_val / lcdm_val)
            else:
                result[float(target)] = float("nan")
        return result

    @staticmethod
    def _build_boost_lines(boosts: dict[float, float] | None) -> str | None:
        if not boosts:
            return None
        lines = []
        for z in sorted(boosts):
            ratio = boosts[z]
            formatted = f"{ratio:.2f}x" if np.isfinite(ratio) else "n/a"
            lines.append(f"    at z={z:.0f}: {formatted}")
        return "\n" + "\n".join(lines)
