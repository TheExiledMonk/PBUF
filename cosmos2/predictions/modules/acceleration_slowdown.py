"""Estimate the future slowdown redshift from elastic saturation ideas."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np

from ..structures import PredictionPlot, PredictionResult, PredictionTable
from ..registry import PredictionModule, register_prediction

if TYPE_CHECKING:
    from ..model_api import PredictionModelAdapter


@register_prediction
class AccelerationSlowdownPrediction(PredictionModule):
    name = "acceleration-slowdown"
    version = "v1"
    description = "Predict the redshift when cosmic acceleration starts to slow down."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument("--amin", type=float, default=0.2, help="Minimum scale factor to survey.")
        parser.add_argument("--amax", type=float, default=1.6, help="Maximum scale factor to survey.")
        parser.add_argument("--points", type=int, default=300, help="Sampling density for the profile.")
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        amin = float(config.get("amin", 0.2))
        amax = float(max(config.get("amax", 1.6), amin + 0.1))
        points = int(config.get("points", 300))
        a_vals = np.linspace(amin, amax, max(points, 10), dtype=float)
        bg = model.background(a_vals)
        q_vals = bg["q"]
        z_vals = bg["z"]

        future_mask = a_vals >= 1.0
        slowdown_z = None
        if future_mask.any():
            q_future = q_vals[future_mask]
            z_future = z_vals[future_mask]
            positive = q_future >= 0.0
            if positive.any():
                slowdown_z = float(z_future[np.argmax(positive)])

        table = PredictionTable(
            name="deceleration_profile",
            columns=["z", "q"],
            rows=list(zip(z_vals.tolist(), q_vals.tolist())),
        )
        plot = PredictionPlot(
            name="deceleration_curve",
            description="Deceleration parameter vs redshift",
            data={"z": z_vals.tolist(), "q": q_vals.tolist()},
        )

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata={"scale_factor_range": [amin, amax]},
            results={"z_slowdown": slowdown_z},
            tables=[table],
            plots=[plot],
        )
