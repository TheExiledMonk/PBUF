#!/usr/bin/env python3

"""Dump the derived PBUF Ω_b0 and Ω_m0 values to standard output."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cosmos2.pbuf.microphysics import ensure_thermal_table
from cosmos2.models.pbuf.params import PBUFParams, coerce_pbuf_parameters
from cosmos2.models.pbuf.normalization import normalize_parameters


def _resolve_params(overrides: Mapping[str, Any]) -> PBUFParams:
    base = {"H0": 70.0, "Rmax": 9e7}
    base.update(overrides)
    coerced = coerce_pbuf_parameters(base, normalization_mode="flat_today")
    return PBUFParams(**coerced)


def main() -> None:
    table = ensure_thermal_table()
    params = _resolve_params({})
    finalized, _, _ = normalize_parameters(params, table)

    omega_b0 = finalized.Omega_b0
    omega_m0 = finalized.Omega_m0
    print(f"Derived Ω_b0 = {omega_b0:.6f}")
    print(f"Derived Ω_m0 = {omega_m0:.6f}")


if __name__ == "__main__":
    main()
