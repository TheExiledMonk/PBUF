"""Loose dataset sanity checks for CMB summary statistics."""

from __future__ import annotations

from typing import Dict

from cosmos.optim.sanity_base import SanityResult

Summary = Dict[str, float]


def check_cmb_sanity(summary: Summary) -> SanityResult:
    res = SanityResult()

    l_A = float(summary["l_A"])
    theta_star = float(summary["theta_star"])
    z_star = float(summary["z_star"])
    D_M = float(summary["D_M"])
    D_A = float(summary["D_A"])
    r_s = float(summary["r_s"])
    R = summary.get("R")

    if not (200.0 <= l_A <= 400.0):
        res.add_error(f"CMB sanity: l_A={l_A} out of [200, 400]")

    if not (0.005 <= theta_star <= 0.02):
        res.add_error(f"CMB sanity: theta_star={theta_star} out of [0.005, 0.02]")

    if not (800.0 <= z_star <= 2000.0):
        res.add_error(f"CMB sanity: z_star={z_star} out of [800, 2000]")

    if D_A <= 0.0 or D_M <= 0.0:
        res.add_error(f"CMB sanity: non-positive distances D_A={D_A}, D_M={D_M}")

    if not (80.0 <= r_s <= 200.0):
        res.add_error(f"CMB sanity: r_s={r_s} out of [80, 200]")

    if R is not None and not (0.5 <= float(R) <= 3.0):
        res.add_error(f"CMB sanity: R={R} out of [0.5, 3.0]")

    return res
