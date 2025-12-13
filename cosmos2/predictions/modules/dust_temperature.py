"""Dust temperature scaling predictions."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np

from ..structures import PredictionPlot, PredictionResult, PredictionTable
from ..registry import PredictionModule, register_prediction

if TYPE_CHECKING:
    from ..model_api import PredictionModelAdapter


@register_prediction
class DustTemperaturePrediction(PredictionModule):
    name = "dust-temperature"
    version = "v1"
    description = "Estimate dust temperature evolution from the model thermal history."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=2.0, help="Minimum redshift in the scan.")
        parser.add_argument("--zmax", type=float, default=20.0, help="Maximum redshift in the scan.")
        parser.add_argument("--points", type=int, default=150, help="Point count for the redshift grid.")
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        zmin = float(config.get("zmin", 2.0))
        zmax = float(config.get("zmax", 20.0))
        points = int(config.get("points", 150))
        if zmax <= zmin:
            raise ValueError("zmax must be greater than zmin.")
        z_vals = np.linspace(zmin, zmax, max(points, 2), dtype=float)
        a_vals = 1.0 / (1.0 + z_vals)
        photon_temps = model.temperature(a_vals)
        scale = 1.0 + 0.05 * model.parameters.get("Omega_m0", 0.3)
        dust_temps = photon_temps * (1.0 + 0.15 * np.log1p(z_vals)) * scale

        rows = list(zip(z_vals.tolist(), dust_temps.tolist()))
        table = PredictionTable(
            name="dust_temperature_vs_z",
            columns=["z", "T_dust_K"],
            rows=rows,
            metadata={"scale_factor_points": len(z_vals)},
        )
        plot = PredictionPlot(
            name="dust_temperature_curve",
            description="Dust temperature vs redshift",
            data={"z": z_vals.tolist(), "T_dust_K": dust_temps.tolist()},
            metadata={"model": model.raw_model.__class__.__name__},
        )
        metadata = {"zmin": zmin, "zmax": zmax, "points": len(z_vals)}

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results={
                "T_dust_zmin_K": float(dust_temps[0]),
                "T_dust_zmax_K": float(dust_temps[-1]),
            },
            tables=[table],
            plots=[plot],
        )
