"""Prediction module that checks the PBUF baryon–saturation–rigidity identity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable


@register_prediction
class CurvatureIdentityPrediction(PredictionModule):
    name = "curvature-identity"
    version = "v1"
    description = "Checks the PBUF baryon–saturation–rigidity identity (Ω_b0≈2α, k_sat≈1−α, k_max≈ε₀−α)."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Export the identity components (actual, predicted, delta) as a table.",
        )
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Emit a bar chart showing the residuals for each component.",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        output_table = bool(config.get("output_table"))
        output_plot = bool(config.get("output_plot"))

        params = model.parameters
        alpha = _to_float(params.get("alpha"))
        Omega_b0_actual = _to_float(params.get("Omega_b0"))

        if alpha is None or Omega_b0_actual is None:
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata={
                    "error": "missing_alpha_or_Omega_b0",
                    "summary": "Curvature-identity prediction not supported (missing alpha or Omega_b0).",
                },
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        epsilon0_today = _safe_epsilon0_today(model)
        k_sat_actual = _to_float(params.get("k_sat"))
        k_max_actual = _to_float(params.get("k_max"))

        Omega_b0_pred = 2.0 * alpha
        k_sat_pred = 1.0 - alpha
        k_max_pred = epsilon0_today - alpha if epsilon0_today is not None else None

        delta_Omega_b0 = _delta(Omega_b0_actual, Omega_b0_pred)
        delta_k_sat = _delta(k_sat_actual, k_sat_pred)
        delta_k_max = _delta(k_max_actual, k_max_pred)

        rows: list[list[float | str | None]] = []
        rows.append(["Omega_b0", Omega_b0_actual, Omega_b0_pred, delta_Omega_b0])
        rows.append(["k_sat", k_sat_actual, k_sat_pred, delta_k_sat])
        if k_max_actual is not None and k_max_pred is not None:
            rows.append(["k_max", k_max_actual, k_max_pred, delta_k_max])

        tables: list[PredictionTable] = []
        if output_table:
            tables.append(
                PredictionTable(
                    name="curvature_identity_components",
                    columns=["quantity", "actual", "predicted", "delta"],
                    rows=rows,
                    metadata={"generated_by": self.name},
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plot_items = [
                (label, delta)
                for label, _, _, delta in rows
                if delta is not None
            ]
            if plot_items:
                plot_names, plot_deltas = zip(*plot_items)
                plots.append(
                    PredictionPlot(
                        name="curvature_identity_bar",
                        description="Residuals of the baryon–saturation–rigidity identity.",
                        data={"x": list(plot_names), "y": [float(delta) for delta in plot_deltas]},
                        metadata={"type": "bar", "xlabel": "Quantity", "ylabel": "actual - predicted"},
                    )
                )

        results: dict[str, float | None] = {
            "alpha": alpha,
            "epsilon0_today": epsilon0_today,
            "Omega_b0_actual": Omega_b0_actual,
            "Omega_b0_pred": Omega_b0_pred,
            "Delta_Omega_b0": delta_Omega_b0,
            "k_sat_actual": k_sat_actual,
            "k_sat_pred": k_sat_pred,
            "Delta_k_sat": delta_k_sat,
            "k_max_actual": k_max_actual,
            "k_max_pred": k_max_pred,
            "Delta_k_max": delta_k_max,
        }

        Omega_k0 = _to_float(params.get("Omega_k0"))
        if Omega_k0 is not None:
            results["Omega_k0"] = Omega_k0
            results["closure_today"] = 1.0 - Omega_k0

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "has_k_max": k_max_actual is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": "Check of the PBUF baryon–saturation–rigidity identity (Ω_b0 ≈ 2α, k_sat ≈ 1−α, k_max ≈ ε₀−α)",
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )


def _to_float(value: object | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _delta(actual: float | None, predicted: float | None) -> float | None:
    if actual is None or predicted is None:
        return None
    return actual - predicted


def _scalar_from_array(values: Sequence[float]) -> float:
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.size == 0:
        raise ValueError("Cannot extract scalar from empty array.")
    return float(arr.reshape(-1)[0])


def _safe_epsilon0_today(model: "PredictionModelAdapter") -> float | None:
    try:
        epsilon = model.elastic_stiffness(1.0)
    except AttributeError:
        return None
    try:
        return _scalar_from_array(epsilon)
    except ValueError:
        return None
