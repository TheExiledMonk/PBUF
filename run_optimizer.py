#!/usr/bin/env python3
"""
Coordinate descent driver for the PBUF basin walker.

Example:
    python run_optimizer.py \
        --model pbuf \
        --datasets cmb,sn_pantheon,bao_iso \
        --phase6a \
        --out data/results/basin_scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from cosmos.optim.coord_optimizer import (
    CoordinateBasinWalker,
    DEFAULT_PBUF_REFERENCE,
    SECOND_PASS_PARAMS,
)


def _parse_datasets(raw: str) -> Sequence[str]:
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _load_seed(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Seed JSON must contain an object, got {type(data).__name__}")
    normalized: Dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
        else:
            raise ValueError(f"Seed parameter '{key}' must be numeric, got {value!r}")
    return normalized


def _build_second_pass(enabled: bool) -> Sequence[str]:
    return tuple(SECOND_PASS_PARAMS) if enabled else tuple()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PBUF coordinate optimizer (basin walker).")
    parser.add_argument(
        "--model",
        default="pbuf",
        help="Cosmological model to optimize (currently only 'pbuf' supported).",
    )
    parser.add_argument(
        "--datasets",
        default="cmb,sn_pantheon,bao_iso",
        help="Comma-separated list of dataset tags to include in χ².",
    )
    parser.add_argument(
        "--phase6a",
        action="store_true",
        help="Enforce Phase 6a physical sanity filter during scans.",
    )
    parser.add_argument(
        "--delta-chi2",
        type=float,
        default=20.0,
        help="Δχ² tolerance used to define basin edges (default: 20).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path for optimizer results.",
    )
    parser.add_argument(
        "--seed-json",
        type=Path,
        help="Optional JSON file with starting parameters to override the reference cosmology.",
    )
    parser.add_argument(
        "--skip-second-pass",
        action="store_true",
        help="Disable the tightening pass over (H0, Om0).",
    )

    args = parser.parse_args(argv)

    model = args.model.strip().lower()
    if model != "pbuf":
        parser.error("Only the PBUF model is supported by the basin walker at this time.")

    datasets = _parse_datasets(args.datasets)
    if not datasets:
        parser.error("At least one dataset must be specified via --datasets.")

    reference = dict(DEFAULT_PBUF_REFERENCE)
    if args.seed_json:
        if not args.seed_json.exists():
            parser.error(f"Seed JSON not found: {args.seed_json}")
        seed_values = _load_seed(args.seed_json)
        reference.update(seed_values)

    walker = CoordinateBasinWalker(
        model_type=model,
        datasets=datasets,
        enforce_phase6a=args.phase6a,
        delta_chi2=args.delta_chi2,
        reference_params=reference,
        second_pass_params=_build_second_pass(not args.skip_second_pass),
    )

    result = walker.run_and_save(args.out)

    print("✅ Basin walk complete")
    print(f"   Model: {result['model_type']}")
    print(f"   Datasets: {', '.join(result['datasets_used'])}")
    print(f"   Output: {Path(args.out).resolve()}")
    if "fiducial_chi2" in result:
        print(f"   Fiducial χ²: {result['fiducial_chi2']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
