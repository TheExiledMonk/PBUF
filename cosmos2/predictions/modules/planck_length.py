"""Predict the evolution of the effective Planck length from thermal rigidity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable


@register_prediction
class PlanckLengthPrediction(PredictionModule):
    name = "planck-length"
    version = "v1"
    description = "Compute the Planck length evolution ell_Pl(z)/ell_Pl(0) from elastic rigidity."

    _SUPPORTED_MODES = {"epsilon-only"}
    _TARGET_REDSHIFTS = (2.0, 6.0, 10.0)

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift in the scan.")
        parser.add_argument("--zmax", type=float, default=30.0, help="Maximum redshift in the scan.")
        parser.add_argument("--points", type=int, default=300, help="Number of redshift samples.")
        parser.add_argument(
            "--mode",
            type=str,
            default="epsilon-only",
            help='Scaling mode (currently only "epsilon-only" is supported).',
        )
        parser.add_argument("--output-plot", action="store_true", help="Emit the canonical ell_Pl ratio plot.")
        parser.add_argument("--output-table", action="store_true", help="Export the ell_Pl evolution table.")
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        zmin = float(config.get("zmin", 0.0))
        zmax = float(config.get("zmax", 30.0))
        if zmax <= zmin:
            raise ValueError("zmax must be greater than zmin.")
        if zmin <= -1.0:
            raise ValueError("zmin must be greater than -1 to keep the scale factor finite.")
        grid_points = max(int(config.get("points", 300)), 2)
        mode_raw = str(config.get("mode", "epsilon-only") or "epsilon-only")
        mode_key = mode_raw.strip().lower()
        if not mode_key:
            mode_key = "epsilon-only"
        if mode_key not in self._SUPPORTED_MODES:
            mode_note = f"Mode '{mode_raw}' is not implemented; falling back to epsilon-only."
            mode_key = "epsilon-only"
        else:
            mode_note = None
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        z_vals = np.linspace(zmin, zmax, grid_points, dtype=float)
        a_vals = 1.0 / (1.0 + z_vals)
        try:
            epsilon_vals = np.asarray(model.elastic_stiffness(a_vals), dtype=float)
            epsilon_today = float(model.elastic_stiffness(1.0))
        except AttributeError:
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata={"error": "missing_T_of_z_or_epsilon0_of_T"},
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        if not np.all(np.isfinite(epsilon_vals)) or not np.isfinite(epsilon_today):
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata={"error": "invalid_rigidity_values", "summary": "Elastic stiffness must be finite."},
                results={},
                tables=[],
                plots=[],
                status="error",
            )
        if epsilon_today <= 0.0 or np.any(epsilon_vals <= 0.0):
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata={"error": "non_positive_rigidity", "summary": "Elastic stiffness must remain positive."},
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        temperature_vals = np.asarray(model.temperature(a_vals), dtype=float)
        ell_ratio = np.sqrt(epsilon_today / epsilon_vals)

        def _ratio_at(z_target: float) -> float | None:
            if z_target < z_vals[0] or z_target > z_vals[-1]:
                return None
            return float(np.interp(z_target, z_vals, ell_ratio))

        target_ratios = {f"ell_ratio_at_z{int(z)}": _ratio_at(z) for z in self._TARGET_REDSHIFTS}

        max_index = int(np.nanargmax(ell_ratio))
        max_ratio = float(ell_ratio[max_index])
        max_ratio_redshift = float(z_vals[max_index])

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                [
                    float(z),
                    float(a),
                    float(T),
                    float(eps),
                    float(ratio),
                ]
                for z, a, T, eps, ratio in zip(z_vals, a_vals, temperature_vals, epsilon_vals, ell_ratio)
            ]
            tables.append(
                PredictionTable(
                    name="planck_length_vs_z",
                    columns=["z", "a", "T", "epsilon0", "ell_ratio"],
                    rows=rows,
                    metadata={
                        "mode": mode_key,
                        "requested_mode": mode_raw,
                        "points": len(z_vals),
                        "zmin": zmin,
                        "zmax": zmax,
                    },
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="ell_ratio_vs_z",
                    description="Evolution of ell_Pl(z)/ell_Pl(0) following epsilon0(T) stiffness.",
                    data={"z": z_vals.tolist(), "ell_ratio": ell_ratio.tolist()},
                    metadata={
                        "xlabel": "redshift z",
                        "ylabel": "ell_Pl(z) / ell_Pl(0)",
                        "points": len(z_vals),
                    },
                )
            )

        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {
            "summary": "Predicted evolution of the effective Planck length ell_Pl(z) from thermal rigidity epsilon0(T).",
            "model": model.raw_model.__class__.__name__,
            "mode": mode_key,
            "points": len(z_vals),
            "zmin": float(zmin),
            "zmax": float(zmax),
            "timestamp": timestamp,
        }
        if mode_note:
            metadata["note"] = mode_note

        results = {
            "zmin": float(zmin),
            "zmax": float(zmax),
            "max_ell_ratio": max_ratio,
            "max_ell_ratio_redshift": max_ratio_redshift,
            **{key: value for key, value in target_ratios.items()},
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
