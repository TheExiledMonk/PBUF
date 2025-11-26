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

from cosmos2.models.model_factory import create_model
from cosmos2.models.pbuf.distances import omega_total_at_a
from cosmos2.models.pbuf.elastic import omega_sigma_of_a
from cosmos2.models.pbuf.model import PBUFModel
from cosmos2.models.pbuf.sanity import check_pbuf_sanity
from cosmos2.models.pbuf.thermal_table import ThermalTable


BASELINE_LCDM = dict(
    H0=70.0,
    Omega_m0=0.3,
    Omega_b0=0.05,
    Omega_r0=1.0e-4,
)

BASELINE_PBUF = dict(
    H0=70.0,
    Omega_m0=0.3,
    Omega_b0=0.05,
    Omega_r0=1.0e-4,
    alpha=0.001,
    Rmax=1.0e6,
)


def build_lcdm():
    return create_model("lcdm", **BASELINE_LCDM)


def build_pbuf():
    rows = _make_rows()
    table = ThermalTable(rows)
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


def _closure_at_a1(model) -> float:
    params = model._params  # noqa: SLF001
    table = model._thermal  # noqa: SLF001
    alpha = model._alpha  # noqa: SLF001
    omega_sigma_today = omega_sigma_of_a(1.0, params, table)
    return omega_total_at_a(1.0, params, table, alpha=alpha), omega_sigma_today


def main():
    parser = argparse.ArgumentParser(description="Compare LCDM vs PBUF H(z) for sanity.")
    parser.add_argument("--z-grid", nargs="*", type=float, default=[0.0, 0.1, 0.3, 0.5, 1.0, 2.0])
    args = parser.parse_args()

    z_samples = np.asarray(args.z_grid, dtype=float)

    lcdm = build_lcdm()
    pbuf = build_pbuf()
    pbuf.thermal_table = pbuf._thermal  # type: ignore[attr-defined] # noqa: SLF001

    print("=== Background sanity: cosmos2 PBUF vs LCDM ===\n")
    print("z      H_lcdm   H_pbuf")
    for z in z_samples:
        H_l = float(lcdm.Hubble(z))
        H_p = float(pbuf.Hubble(z))
        print(f"{z:4.1f}  {H_l:7.2f}  {H_p:7.2f}")

    omega_sum, omega_sigma = _closure_at_a1(pbuf)
    print("\nClosure (a=1):")
    print(f"  Ω_total(a=1)   = {omega_sum:.6f}")
    print(f"  Ω_sigma(a=1)   = {omega_sigma:.6f}")
    print(f"  σ8 (assumed)   = {pbuf.sigma8():.6f}")

    sanity = check_pbuf_sanity(pbuf._numeric_parameters(), pbuf)  # noqa: SLF001
    print("\nPhase-7a sanity:")
    print(f"  {'PASS' if sanity.ok else 'FAIL'}")
    if not sanity.ok:
        for reason in sanity.reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
