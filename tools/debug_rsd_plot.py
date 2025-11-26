"""Quick script to inspect RSD residuals under a candidate cosmology."""

from __future__ import annotations

import argparse

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.models.model_factory import create_model as create_cosmos2_model

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting
    plt = None


def parse_params(param_items: list[str]) -> dict[str, float]:
    params: dict[str, float] = {}
    for item in param_items:
        if "=" not in item:
            raise ValueError(f"Invalid --param '{item}', expected key=value")
        key, value = item.split("=", 1)
        params[key] = float(value)
    return params


def _load_pbuf_lut() -> dict[str, np.ndarray]:
    from cosmos2.pbuf.microphysics import ensure_thermal_table

    table = ensure_thermal_table()
    return {
        "T": np.asarray(table.T, dtype=float),
        "eps": np.asarray(table.eps, dtype=float),
        "alpha": np.asarray(table.alpha, dtype=float),
        "dln_eps": np.asarray(table.dln_eps, dtype=float),
        "dln_alpha": np.asarray(table.dln_alpha, dtype=float),
        "g_star": np.asarray(table.g_star, dtype=float),
        "g_starS": np.asarray(table.g_starS, dtype=float),
        "a": np.asarray(table.a, dtype=float),
        "metadata": getattr(table, "metadata", {}),
    }


def build_model(model_name: str, overrides: dict[str, float]) -> object:
    normalized = {key: float(value) for key, value in overrides.items()}
    if model_name == "pbuf":
        lut = _load_pbuf_lut()
        return create_cosmos2_model("pbuf", lut=lut, **normalized)
    return create_cosmos2_model(model_name, **normalized)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RSD fσ₈ against a candidate model.")
    parser.add_argument("--model", choices=["lcdm", "pbuf"], default="lcdm")
    parser.add_argument("--param", action="append", default=[], help="Model override (key=value)")
    parser.add_argument("--output", help="Path to save the figure")
    args = parser.parse_args()

    if plt is None:
        parser.exit("matplotlib is required to plot RSD data. Please install it first.\n")

    params = parse_params(args.param)
    model = build_model(args.model, params)
    dataset = get_dataset("rsd")
    z = dataset["z"]
    fs8_obs = dataset["obs"]
    fs8_err = dataset.get("err")
    fs8_model = np.asarray(model.fs8(z), dtype=float)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].set_title(f"RSD fσ₈ ({args.model.upper()})")
    axes[0].plot(z, fs8_model, label="model", color="tab:orange")
    axes[0].scatter(z, fs8_obs, label="data", color="tab:blue", s=8, alpha=0.7)
    if fs8_err is not None:
        axes[0].errorbar(z, fs8_obs, yerr=fs8_err, fmt="none", ecolor="tab:blue", alpha=0.4)
    axes[0].set_ylabel("fσ₈")
    axes[0].legend(loc="best")

    residuals = fs8_obs - fs8_model
    axes[1].plot(z, residuals, color="tab:green")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("obs - model")
    axes[1].axhline(0.0, color="k", lw=0.7, linestyle="--")

    fig.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=200)
        print(f"Saved RSD plot to {args.output}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
