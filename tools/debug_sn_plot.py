"""Quick script to inspect the Pantheon+ SN residuals under a candidate model."""

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
    parser = argparse.ArgumentParser(description="Plot SN residuals for a given cosmology candidate.")
    parser.add_argument("--model", choices=["lcdm", "pbuf"], default="lcdm")
    parser.add_argument("--param", action="append", default=[], help="Model override (key=value)")
    parser.add_argument("--output", help="Path to save the figure")
    args = parser.parse_args()

    if plt is None:
        parser.exit(message="matplotlib is required to plot SN residuals, please install it first.\n")

    params = parse_params(args.param)
    model = build_model(args.model, params)
    dataset = get_dataset("sn")
    z = dataset["z"]
    mu_obs = dataset["obs"]
    mu_model = model.distance_modulus(z)
    residuals = mu_obs - mu_model

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].set_title(f"SN Pantheon+ ({args.model.upper()})")
    axes[0].plot(z, mu_model, label="model", color="tab:orange")
    axes[0].scatter(z, mu_obs, label="data", s=8, color="tab:blue", alpha=0.6)
    if dataset.get("err") is not None:
        axes[0].errorbar(z, mu_obs, yerr=dataset["err"], fmt="none", ecolor="tab:blue", alpha=0.4)
    axes[0].set_ylabel("μ (mag)")
    axes[0].legend(loc="best")

    axes[1].plot(z, residuals, color="tab:green")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("obs - model (mag)")
    axes[1].axhline(0.0, color="k", lw=0.7, linestyle="--")

    fig.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=200)
        print(f"Saved SN residual plot to {args.output}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
