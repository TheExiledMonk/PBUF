#!/usr/bin/env python3
"""
CLI entry point to generate Quantum → Cosmos thermal lookup tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from quantum.thermal import (
    ThermalGenerationError,
    ThermalModelConfig,
    ThermalTableSpec,
    generate_thermal_table,
    save_table,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a PBUF thermal lookup table.")
    parser.add_argument("--mode", choices=["off", "linear", "power", "exp"], required=True,
                        help="Thermal model mode.")
    parser.add_argument("--beta", type=float, default=0.05, help="Softening coefficient β.")
    parser.add_argument("--t-star", type=float, default=1.0e6, help="Reference temperature T* (Kelvin).")
    parser.add_argument("--power", type=float, default=1.0, help="Power index p for 'power'/'exp' modes.")
    parser.add_argument("--alpha-qm", type=float, default=0.03, help="Baseline α_QM amplitude.")
    parser.add_argument("--eps-min", type=float, default=1.0e-4, help="Minimum rigidity clamp.")
    parser.add_argument("--t-min", type=float, default=2.725, help="Minimum temperature in Kelvin.")
    parser.add_argument("--t-max", type=float, default=1.0e12, help="Maximum temperature in Kelvin.")
    parser.add_argument("--points", type=int, default=512, help="Number of base log-spaced samples.")
    parser.add_argument("--dense-points", type=int, default=24, help="Samples per refinement window.")
    parser.add_argument("--table-version", type=int, default=11, help="Table version tag.")
    parser.add_argument("--method-version", type=int, default=11, help="Method version tag.")
    parser.add_argument("--regulator", default="thermal_default", help="Regulator provenance label.")
    parser.add_argument("--field-content", default="SM_full", help="Field content provenance label.")
    parser.add_argument("--f-cut", type=float, default=1.0e12, help="Cutoff temperature metadata value.")
    parser.add_argument("--f-coup", type=float, default=1.0e8, help="Coupling temperature metadata value.")
    parser.add_argument("--notes", default="auto-generated via exporter", help="Metadata note.")
    parser.add_argument("--output", type=Path, help="Explicit output path. Defaults to artifacts/thermal/{mode}/thermal_table_vXX.json")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing file.")
    return parser


def _resolve_output(path: Optional[Path], mode: str, table_version: int) -> Path:
    if path:
        return path
    rel = Path("artifacts") / "thermal" / mode / f"thermal_table_v{table_version:02d}.json"
    return rel


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = ThermalModelConfig(
        mode=args.mode,
        beta=args.beta,
        t_star=args.t_star,
        power=args.power,
        alpha_qm=args.alpha_qm,
        eps_min=args.eps_min,
    )
    spec = ThermalTableSpec(
        model=config,
        t_min=args.t_min,
        t_max=args.t_max,
        num_points=args.points,
        dense_points=args.dense_points,
        table_version=args.table_version,
        method_version=args.method_version,
        regulator=args.regulator,
        field_content=args.field_content,
        f_cut_T=args.f_cut,
        f_coup_T=args.f_coup,
        notes=args.notes,
    )

    output_path = _resolve_output(args.output, args.mode, args.table_version)
    if output_path.exists() and not args.overwrite:
        parser.error(f"Output file {output_path} already exists. Pass --overwrite to replace it.")

    try:
        table = generate_thermal_table(spec)
    except ThermalGenerationError as exc:
        parser.error(str(exc))
        return

    save_table(table, output_path)
    print(f"Thermal table written to {output_path}")


if __name__ == "__main__":
    main()
