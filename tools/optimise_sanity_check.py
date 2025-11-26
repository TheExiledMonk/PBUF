"""Manual helper to sanity-check the basin engine."""

from __future__ import annotations

import argparse
from typing import Sequence

from cosmos2.api.engine import run_optimisation
from cosmos2.config import load_bounds_for_model
from cosmos2.fits.registry import FIT_REGISTRY
from cosmos2.models.model_factory import create_model
from cosmos2.science_runner.runner import _load_pbuf_lut


def run(model_name: str, datasets: str, samples: int) -> None:
    dataset_list = [token.strip().lower() for token in datasets.replace(",", " ").split() if token.strip()]
    bounds = load_bounds_for_model(model_name, dataset_list)
    if model_name == "pbuf":
        lut = _load_pbuf_lut()
        factory = lambda params: create_model("pbuf", lut=lut, **params)  # noqa: E731
    else:
        factory = lambda params: create_model("lcdm", **params)  # noqa: E731

    def evaluator(params: dict[str, float]) -> float:
        model = factory(params)
        total = 0.0
        for name in dataset_list:
            fit_fn = FIT_REGISTRY[name]
            chi2 = fit_fn(model)[0]
            total += chi2
        return total

    config = {
        "name": model_name,
        "bounds": bounds,
        "evaluator": evaluator,
        "fits": dataset_list,
        "n_batches": 1,
        "batch_size": max(samples, 1),
        "grid_points": max(samples, 1),
        "rng_seed": 42,
    }
    result = run_optimisation([config])
    best = result["models"][0]
    print(f"[{model_name}] best chi²={best['best_chi2']:.4f}")
    for key, value in sorted(best.get("best_params", {}).items()):
        print(f"    {key}={value:.6g}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight optimisation sanity check.")
    parser.add_argument("--model", choices=["pbuf", "lcdm"], default="pbuf")
    parser.add_argument("--datasets", default="cmb")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    run(args.model, args.datasets, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
