#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "numba" not in sys.modules:
    # Allow running the sanity helper without numba installed.
    import types

    def _njit(fn=None, *args, **kwargs):
        if callable(fn):
            return fn

        def decorator(f):
            return f

        return decorator

    numba_stub = types.SimpleNamespace(njit=_njit)
    sys.modules["numba"] = numba_stub

if not hasattr(np, "cumtrapz"):
    try:
        from scipy.integrate import cumulative_trapezoid

        np.cumtrapz = lambda y, x, initial=0.0: cumulative_trapezoid(y, x, initial=initial)  # type: ignore[attr-defined]
    except Exception:
        def _cumtrapz(y, x, initial=0.0):
            y_arr = np.asarray(y, dtype=float)
            x_arr = np.asarray(x, dtype=float)
            dx = np.diff(x_arr)
            integrand = dx * 0.5 * (y_arr[:-1] + y_arr[1:])
            cumulative = np.cumsum(integrand)
            return np.concatenate(([float(initial)], cumulative + float(initial)))

        np.cumtrapz = _cumtrapz  # type: ignore[attr-defined]

from cosmos2.models.pbuf.fits import run_cmb_fit
from cosmos2.models.pbuf.model import PBUFModel
from cosmos2.models.pbuf.thermal_table import ThermalTable


BASELINE_PBUF = dict(
    H0=70.0,
    Omega_m0=0.3,
    Omega_b0=0.05,
    Omega_r0=1.0e-4,
    alpha=0.001,
    Rmax=1.0e6,
)


def build_pbuf():
    table = ThermalTable(_make_rows())
    return PBUFModel(thermal_table=table, thermal_metadata={}, normalization_mode="flat_today", **BASELINE_PBUF)


def _make_rows() -> list[dict[str, float]]:
    a_grid = np.logspace(-4, 0, 64)
    rows: list[dict[str, float]] = []
    for a in a_grid:
        rows.append(
            {
                "a": float(a),
                "z": float(1.0 / a - 1.0),
                "T": 2.7255 / a,
                "epsilon0_T": 0.002 + 5.0e-4 * (1.0 - a),
                "alpha_T": 0.001 + 2.0e-4 * (1.0 - a),
                "dln_epsilon0_dlnT": 0.0,
                "dln_alpha_dlnT": 0.0,
                "g_star": 3.36,
                "g_starS": 3.9,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="CMB distance prior sanity for cosmos2 PBUF.")
    parser.add_argument("--z-star", type=float, default=1089.92, help="Override recombination redshift.")
    args = parser.parse_args()

    model = build_pbuf()
    model.thermal_table = model._thermal  # type: ignore[attr-defined] # noqa: SLF001

    print("\n=== PBUF CMB distance-prior sanity (cosmos2) ===\n")

    cmb_out = model.cmb({"z_star": args.z_star})
    R = float(cmb_out.R)
    lA = float(cmb_out.l_A)
    theta_star = float(cmb_out.theta_star)

    chi2, extras = run_cmb_fit(model, dataset=None)

    print(f"z_star         = {args.z_star}")
    print(f"R              = {R}")
    print(f"lA             = {lA}")
    print(f"theta_star     = {theta_star}")
    print(f"chi2_vs_planck = {chi2}")
    print(f"valid_model    = {getattr(model, 'is_valid', lambda: False)()}")

    if extras and "dataset" in extras:
        meta = extras["dataset"].get("meta") if isinstance(extras["dataset"], dict) else None
        if meta:
            print(f"dataset meta   = {meta}")


if __name__ == "__main__":
    main()
