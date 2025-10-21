"""
Supernova-only fitting pipeline.

This script orchestrates data loading, model selection, optimisation, and
artefact generation for a single run. It intentionally contains no physics;
all model evaluations are delegated to modules inside `core/`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np

from core import gr_models, pbuf_models
from core.diagnostics import save_correlation_heatmap
from core.pbuf_fitting import FitSettings, fit_model
from core.pbuf_state import ModelParameters, RunState, with_defaults
from dataio.loaders import load_dataset
from utils import logging as log
from utils.io import read_yaml, write_json_atomic, write_yaml_atomic
from utils.plotting import plot_pull_distribution, plot_residuals
from utils import parameters as param_utils


DEFAULT_PARAMS = {
    "lcdm": {"H0": 70.0, "Om0": 0.3},
    "pbuf": {"H0": 70.0, "Om0": 0.3, "alpha": 5.0e-4, "Rmax": 1.0e9, "eps0": 0.7, "n_eps": 0.0},
}


def load_dataset_config_snapshot(name: str, datasets_cfg: dict) -> dict:
    dataset_entries = datasets_cfg.get("datasets", {})
    if name in dataset_entries:
        return dataset_entries[name]
    for entry in dataset_entries.values():
        if isinstance(entry, dict) and name in entry:
            return entry[name]
    raise KeyError(f"Dataset '{name}' not found in configuration snapshot")


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _run_id(tag: str) -> str:
    return f"{tag}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _git_commit() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return output.decode().strip()
    except Exception:
        return "n/a"


def _select_model(name: str):
    if name == "lcdm":
        return gr_models, DEFAULT_PARAMS["lcdm"]
    if name == "pbuf":
        return pbuf_models, DEFAULT_PARAMS["pbuf"]
    raise ValueError(f"Unknown model '{name}'")


def _save_vectors(run_dir: Path, vectors: Dict[str, np.ndarray]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for key, array in vectors.items():
        path = run_dir / f"{key}.npy"
        np.save(path, array)
        paths[key] = str(path.resolve())
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SN-only fits for PBUF or ΛCDM.")
    parser.add_argument("--model", choices=["lcdm", "pbuf"], default="lcdm")
    parser.add_argument("--dataset", default="mock_supernovae")
    parser.add_argument("--out", default="proofs/results")
    parser.add_argument("--chi2-surface", dest="chi2_surface", default=None, help="paramX,paramY for surface plot")
    parser.add_argument("--grid", type=int, default=25, help="Grid resolution per axis for χ² surface")
    parser.add_argument("--generate-report", action="store_true", help="Generate single-fit HTML report")
    args = parser.parse_args()

    log.info(f"Loading configuration and dataset '{args.dataset}'")
    settings = read_yaml("config/settings.yml")
    datasets_cfg = read_yaml("config/datasets.yml")
    report_cfg = read_yaml("config/report.yml")

    dataset = load_dataset(args.dataset, datasets_cfg)
    model_module, default_params = _select_model(args.model)
    model_name = args.model.upper()

    bounds = settings.get("bounds", {})
    fit_settings = FitSettings(
        tolerance=settings.get("fitting", {}).get("tolerance", 1.0e-6),
        max_iter=settings.get("fitting", {}).get("max_iter", 5000),
        bounds=bounds,
    )

    params = with_defaults(default_params, {})
    log.info(f"Running fit for model {model_name}")
    fit_result = fit_model(model_module, dataset, params, fit_settings)
    log.fit(model_name, fit_result["metrics"]["chi2"], fit_result["metrics"]["dof"])

    run_tag = "SN_MOCK" if settings.get("mock") else "SN_REAL"
    run_id = _run_id(f"{run_tag}_{model_name}")
    run_dir = Path(args.out).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vectors = {
        "z": dataset["z"],
        "y_obs": dataset["y"],
        "y_model": fit_result["model_y"],
        "residuals": fit_result["residuals"],
        "pulls": fit_result["pulls"],
    }
    vector_paths = _save_vectors(run_dir, vectors)

    figures = {}
    figures["residuals_vs_z"] = plot_residuals(dataset["z"], fit_result["residuals"], model_name, run_dir)
    figures["pull_distribution"] = plot_pull_distribution(fit_result["pulls"], model_name, run_dir)
    cov_matrix = dataset.get("cov")
    if cov_matrix is not None and cov_matrix.size > 0:
        figures["correlation_matrix"] = save_correlation_heatmap(cov_matrix, run_dir, model_name)

    chi2_surface_meta = None
    if args.chi2_surface:
        from utils.chi2_surface import chi2_surface_scan

        param_x, param_y = [token.strip() for token in args.chi2_surface.split(",")]
        lower_x, upper_x = bounds.get(param_x, (params[param_x] * 0.5, params[param_x] * 1.5))
        lower_y, upper_y = bounds.get(param_y, (params[param_y] * 0.5, params[param_y] * 1.5))
        grid_x = np.linspace(lower_x, upper_x, args.grid)
        grid_y = np.linspace(lower_y, upper_y, args.grid)
        cov = dataset["cov"]
        if cov is None and dataset.get("sigma") is not None:
            sigma = dataset["sigma"]
            cov = np.diag(sigma ** 2)
        png, npy = chi2_surface_scan(
            model_module.mu,
            dataset["z"],
            dataset["y"],
            cov,
            param_x,
            param_y,
            grid_x,
            grid_y,
            fit_result["bestfit"],
            run_dir,
            model_name,
        )
        figures[f"chi2_surface_{param_x}_{param_y}"] = png
        chi2_surface_meta = npy
        vector_paths[f"chi2_surface_{param_x}_{param_y}"] = chi2_surface_meta

    timestamp = _timestamp()
    meta = dataset.get("meta", {})
    dataset_info = {
        "name": meta.get("name", args.dataset),
        "path": meta.get("path"),
        "covariance": meta.get("covariance"),
        "notes": meta.get("notes"),
        "source": meta.get("source"),
        "release_tag": meta.get("release_tag"),
        "raw_files": meta.get("raw_files"),
        "prepared_at": meta.get("prepared_at"),
        "transform_version": meta.get("transform_version"),
        "covariance_components_used": meta.get("covariance_components_used"),
        "z_prefer": meta.get("z_prefer"),
        "derived_files": meta.get("derived_files"),
    }
    dataset_info = {key: value for key, value in dataset_info.items() if value is not None}

    if model_name == "PBUF":
        evolution_policy = {"coupling_matrix": "identity"}
    else:
        evolution_policy = {}

    model_parameters = ModelParameters(values=fit_result["bestfit"], evolution_policy=evolution_policy)
    run_state = RunState(
        run_id=run_id,
        model=model_name,
        dataset=dataset_info,
        parameters=model_parameters,
        extras={"timestamp": timestamp},
    )

    provenance = {
        "commit": _git_commit(),
        "code_version": run_state.version,
        "constants": str(Path("config/constants.py").resolve()),
        "settings": str(Path("config/settings.yml").resolve()),
    }

    figures_payload = dict(figures)
    if chi2_surface_meta:
        label = [key for key in figures if key.startswith("chi2_surface_")][0]
        figures_payload[label] = figures[label]

    data_vectors_payload = {name: path for name, path in vector_paths.items()}

    canonical_block = param_utils.canonical_parameters(model_name)
    parameter_payload = param_utils.build_parameter_payload(
        model_name,
        fitted=fit_result["bestfit"],
        free_names=fit_result["bestfit"].keys(),
        canonical=canonical_block,
    )

    result_json = {
        "run_id": run_id,
        "timestamp": timestamp,
        "mock": bool(settings.get("mock", True)),
        "dataset": dataset_info,
        "model": model_name,
        "parameters": parameter_payload,
        "evolution_policy": evolution_policy,
        "metrics": fit_result["metrics"],
        "data_vectors": data_vectors_payload,
        "figures": figures_payload,
        "provenance": provenance,
    }

    json_path = run_dir / "fit_results.json"
    write_json_atomic(json_path, result_json)

    config_snapshot = {
        "settings": settings,
        "dataset": load_dataset_config_snapshot(args.dataset, datasets_cfg),
        "report": report_cfg,
        "model": model_name,
    }
    write_yaml_atomic(run_dir / "config_used.yml", config_snapshot)

    if args.generate_report:
        log.info("Generating single-fit HTML report")
        from pipelines.generate_report import render_report

        report_path = render_report(result_json, run_dir)
        log.info(f"Report written to {report_path}")

    log.info(f"Results stored in {json_path}")


if __name__ == "__main__":
    main()
