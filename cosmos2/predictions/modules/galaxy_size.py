"""Galaxy-size predictions modulated by the elastic PBUF stiffness history."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

G_SI = 6.67430e-11  # Newton constant [m^3 kg^-1 s^-2]
M_SUN_KG = 1.98847e30
KPC_TO_M = 3.085677581e19
MPC_TO_M = 3.085677581e22


def _rho_crit_per_kpc3(H_km_s_Mpc: np.ndarray) -> np.ndarray:
    """Compute the critical density (Msun / kpc^3) from H(z)."""

    H_si = H_km_s_Mpc * 1e3 / MPC_TO_M
    rho_si = 3.0 * H_si ** 2 / (8.0 * math.pi * G_SI)
    return rho_si / M_SUN_KG * (KPC_TO_M ** 3)


def _interpolate_safe(z_grid: np.ndarray, values: np.ndarray, target: float) -> float:
    if target <= z_grid[0]:
        return float(values[0])
    if target >= z_grid[-1]:
        return float(values[-1])
    return float(np.interp(target, z_grid, values))


@register_prediction
class GalaxySizePrediction(PredictionModule):
    name = "galaxy-size"
    version = "v1"
    description = "Predict PBUF-corrected galaxy sizes and compare to a reference virial scale."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=2.0, help="Minimum redshift for the prediction.")
        parser.add_argument("--zmax", type=float, default=20.0, help="Maximum redshift for the prediction.")
        parser.add_argument("--points", type=int, default=200, help="Sampling density across the redshift range.")
        parser.add_argument("--mass", type=float, default=1e10, help="Characteristic halo mass [Msun].")
        parser.add_argument(
            "--compare-lcdm",
            action="store_true",
            help="Also emit the ΛCDM-style reference sizes for the same grid.",
        )
        parser.add_argument("--output-plot", action="store_true", help="Produce canonical plots for the prediction.")
        parser.add_argument("--output-table", action="store_true", help="Emit tables for the prediction.")
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        zmin = float(config.get("zmin", 2.0))
        zmax = float(config.get("zmax", 20.0))
        if zmax <= zmin:
            raise ValueError("zmax must be greater than zmin.")
        points = max(int(config.get("points", 200)), 2)
        mass_msun = float(config.get("mass", 1e10))
        if mass_msun <= 0.0:
            raise ValueError("mass must be positive.")
        compare_lcdm = bool(config.get("compare_lcdm"))
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        z_grid = np.linspace(zmin, zmax, points, dtype=float)
        a_grid = 1.0 / (1.0 + z_grid)

        try:
            epsilon0 = model.elastic_stiffness(a_grid)
        except AttributeError:
            metadata = {
                "model": model.raw_model.__class__.__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "missing_elastic_stiffness_api",
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

        H_vals = model.H(a_grid)
        rho_crit = _rho_crit_per_kpc3(H_vals)
        rho_crit = np.clip(rho_crit, 1e-40, None)

        delta_c = 200.0
        prefactor = 3.0 / (4.0 * math.pi)
        R_ref = np.power(prefactor * mass_msun / (delta_c * rho_crit), 1.0 / 3.0)
        R_ref = np.nan_to_num(R_ref, nan=0.0, posinf=0.0, neginf=0.0)

        epsilon_clipped = np.clip(epsilon0, 0.0, None)
        g_eps = np.sqrt(epsilon_clipped)
        R_gal = R_ref * g_eps
        size_ratio = np.zeros_like(R_ref)
        np.divide(R_gal, R_ref, out=size_ratio, where=R_ref > 0.0)

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                [
                    float(z),
                    float(a),
                    float(eps),
                    float(r_ref),
                    float(r_gal),
                    float(ratio),
                ]
                for z, a, eps, r_ref, r_gal, ratio in zip(z_grid, a_grid, epsilon_clipped, R_ref, R_gal, size_ratio)
            ]
            tables.append(
                PredictionTable(
                    name="galaxy_size_vs_z",
                    columns=["z", "a", "epsilon0", "R_ref_kpc", "R_gal_kpc", "ratio"],
                    rows=rows,
                    metadata={"mass_Msun": mass_msun, "points": len(z_grid)},
                )
            )
            if compare_lcdm:
                lcdm_rows = [
                    [float(z), float(r_ref)] for z, r_ref in zip(z_grid, R_ref)
                ]
                tables.append(
                    PredictionTable(
                        name="galaxy_size_vs_z_lcdm",
                        columns=["z", "R_lcdm_kpc"],
                        rows=lcdm_rows,
                        metadata={"mass_Msun": mass_msun, "points": len(z_grid)},
                    )
                )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="R_gal_vs_z",
                    data={"z": z_grid.tolist(), "R_gal_kpc": R_gal.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "galaxy size R_gal [kpc]"},
                )
            )
            plots.append(
                PredictionPlot(
                    name="size_ratio_vs_z",
                    data={"z": z_grid.tolist(), "ratio": size_ratio.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "size ratio (PBUF / ref)"},
                )
            )

        targets = {"z2": 2.0, "z10": 10.0}
        results: dict[str, float] = {
            "zmin": zmin,
            "zmax": zmax,
            "mass_Msun": mass_msun,
        }
        for label, z_val in targets.items():
            results[f"R_gal_at_z{int(z_val)}_kpc"] = _interpolate_safe(z_grid, R_gal, z_val)
        results["size_ratio_z10"] = _interpolate_safe(z_grid, size_ratio, 10.0)

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "points": points,
            "compare_lcdm": compare_lcdm,
            "output_plot": output_plot,
            "output_table": output_table,
            "zmin": zmin,
            "zmax": zmax,
        }

        if compare_lcdm:
            metadata["baseline"] = "virial_reference"

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
