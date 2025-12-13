"""Curvature prediction plug-in exposing the PBUF residual curvature forecasts."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable


class _CurvatureRequirementsError(Exception):
    """Signals that the selected model cannot provide the needed geometry data."""


@register_prediction
class CurvaturePrediction(PredictionModule):
    name = "curvature"
    version = "v1"
    description = "Predicted residual spatial curvature Ω_k0 at z = 0."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Emit the standard curvature component bar chart.",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Export today's curvature components as a CSV table.",
        )
        parser.add_argument(
            "--diagnostics",
            action="store_true",
            help="Print extra diagnostics such as Omega_m0, alpha, and k_sat.",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))
        diagnostics = bool(config.get("diagnostics"))

        try:
            payload = self._compute_curvature(model)
        except _CurvatureRequirementsError as exc:
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata={"error": str(exc)},
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        H0 = payload["H0"]
        Omega_k0 = payload["Omega_k0"]
        closure = payload["closure_today"]
        components = payload["components"]
        epsilon0_today = payload.get("epsilon0_today")
        k_sat = payload.get("k_sat")
        curvature_radius = payload["curvature_radius_Mpc"]

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                ["Omega_m0", components["Omega_m0"]],
                ["Omega_b0", components["Omega_b0"]],
                ["Omega_r0", components["Omega_r0"]],
                ["Omega_sigma0", components["Omega_sigma0"]],
                ["alpha", components["alpha"]],
                ["k_sat", components["k_sat"]],
                ["closure_today", closure],
            ]
            tables.append(
                PredictionTable(
                    name="curvature_components",
                    columns=["component", "value"],
                    rows=rows,
                    metadata={"as_of": "a=1"},
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plot_names = ["Omega_m0", "Omega_r0", "alpha", "Omega_sigma0"]
            plot_values = [components["Omega_m0"], components["Omega_r0"], components["alpha"], components["Omega_sigma0"]]
            plots.append(
                PredictionPlot(
                    name="curvature_bar_chart",
                    description="Closure components that feed today's curvature bias.",
                    data={"x": plot_names, "y": plot_values},
                    metadata={
                        "type": "bar",
                        "xlabel": "Components",
                        "ylabel": "Fraction",
                    },
                )
            )

        results: dict[str, float | dict[str, float | None]] = {
            "Omega_k0": Omega_k0,
            "curvature_radius_Mpc": curvature_radius,
            "closure_today": closure,
            "components": components,
        }
        if diagnostics:
            results.update(_diagnostic_results(components, epsilon0_today, k_sat, Omega_k0))

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "diagnostics": diagnostics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": "Predicted residual spatial curvature Ω_k at z = 0",
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )

    def _compute_curvature(self, model: "PredictionModelAdapter") -> dict[str, float | dict[str, float | None]]:
        params = model.parameters
        try:
            H0 = float(params["H0"])
        except KeyError as exc:
            raise _CurvatureRequirementsError("Model missing H0 parameter") from exc
        if H0 <= 0.0 or not math.isfinite(H0):
            raise _CurvatureRequirementsError("Model H0 must be positive")

        H_today = _scalar_from_array(model.H(1.0))
        closure = (H_today / H0) ** 2 if H0 != 0.0 else float("nan")
        Omega_m0 = float(params.get("Omega_m0", 0.0))
        Omega_r0 = float(params.get("Omega_r0", 0.0))
        alpha = float(params.get("alpha", 0.0))
        Omega_b0 = float(params.get("Omega_b0", 0.0))

        Omega_sigma0 = closure - (Omega_m0 + Omega_r0 + alpha)
        Omega_k0 = 1.0 - closure
        epsilon0_today = _safe_elastic_stiffness(model)
        k_sat = (
            float(epsilon0_today - alpha) if epsilon0_today is not None else None
        )
        curvature_radius = _compute_curvature_radius(model.constants["c_km_per_s"], H0, Omega_k0)

        components = {
            "Omega_m0": Omega_m0,
            "Omega_r0": Omega_r0,
            "Omega_b0": Omega_b0,
            "Omega_sigma0": Omega_sigma0,
            "alpha": alpha,
            "k_sat": float(k_sat) if k_sat is not None else None,
        }

        return {
            "H0": H0,
            "Omega_k0": Omega_k0,
            "closure_today": closure,
            "components": components,
            "epsilon0_today": epsilon0_today,
            "k_sat": float(k_sat) if k_sat is not None else None,
            "curvature_radius_Mpc": curvature_radius,
        }


def _scalar_from_array(values: float | Iterable[float]) -> float:
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.size == 0:
        raise _CurvatureRequirementsError("Cannot evaluate Hubble rate at a=1.")
    return float(arr.reshape(-1)[0])


def _safe_elastic_stiffness(model: "PredictionModelAdapter") -> float | None:
    try:
        stiffness = model.elastic_stiffness(1.0)
    except AttributeError:
        return None
    return _scalar_from_array(stiffness)


def _compute_curvature_radius(c_km_per_s: float, H0: float, Omega_k0: float) -> float:
    if not math.isfinite(Omega_k0) or Omega_k0 == 0.0:
        return float("inf")
    return float(c_km_per_s / (H0 * math.sqrt(abs(Omega_k0))))


def _diagnostic_results(
    components: dict[str, float | None],
    epsilon0_today: float | None,
    k_sat: float | None,
    Omega_k0: float,
) -> dict[str, float | None]:
    diagnostics: dict[str, float | None] = {
        "Omega_m0": components["Omega_m0"],
        "Omega_b0": components["Omega_b0"],
        "Omega_r0": components["Omega_r0"],
        "Omega_sigma0": components["Omega_sigma0"],
        "alpha": components["alpha"],
        "k_sat": k_sat,
        "residual_curvature": Omega_k0,
    }
    if epsilon0_today is not None:
        diagnostics["epsilon0_today"] = epsilon0_today
    return diagnostics
