"""Lightweight CLI for running fits and optimisations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np

from cosmos2.config import load_bounds_for_model
from cosmos2.data.registry import get_dataset
from cosmos2.fits.bao_aniso import run_bao_aniso_fit
from cosmos2.fits.bao_iso import run_bao_iso_fit
from cosmos2.fits.cc import run_cc_fit
from cosmos2.fits.galaxy_pk import run_galaxy_pk_fit
from cosmos2.fits.lensing_cross import run_lensing_cross_fit
from cosmos2.fits.rsd import run_rsd_fit
from cosmos2.fits.wl import run_wl_s8_fit
from cosmos2.fits.cmb import run_fit as run_cmb_fit
from cosmos2.fits.sh0es import run_sh0es_prior
from cosmos2.fits.sn import run_sn_pantheon_fit
from cosmos2.fits.registry import FIT_REGISTRY as COSMOS2_FIT_REGISTRY
from cosmos2.api.engine import run_optimisation
from cosmos2.models.model_factory import create_model as create_cosmos2_model

from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.unified_runner import UnifiedScienceRunner
from cosmos2.threads.monitor_types import available_monitor_modes, normalize_monitor_mode
from toolbox import run_toolbox


def parse_params(param_list: Sequence[str]) -> dict[str, float | str]:
    params: dict[str, float | str] = {}
    for item in param_list:
        if "=" not in item:
            raise ValueError(f"Invalid --param '{item}', expected key=value")
        key, value = item.split("=", 1)
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    return params


def _preview_vector(label: str, values: Any, *, limit: int = 3) -> None:
    try:
        arr = np.asarray(values).flatten()
    except Exception:
        return
    if arr.size:
        trimmed = arr[:limit].tolist()
        print(f"{label}: {trimmed!r}")


def _load_pbuf_lut() -> dict[str, Any]:
    """Bridge legacy thermal table into the LUT mapping expected by cosmos2."""
    global _PBUF_LUT
    if _PBUF_LUT is not None:
        return _PBUF_LUT
    from cosmos2.pbuf.microphysics import ensure_thermal_table

    table = ensure_thermal_table()
    metadata = dict(getattr(table, "metadata", {}) or {})
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update({k: v for k, v in metadata.items() if k != "metadata"})
        metadata = merged
    _PBUF_LUT = {
        "T": np.asarray(table.T, dtype=float),
        "eps": np.asarray(table.eps, dtype=float),
        "alpha": np.asarray(table.alpha, dtype=float),
        "dln_eps": np.asarray(table.dln_eps, dtype=float),
        "dln_alpha": np.asarray(table.dln_alpha, dtype=float),
        "g_star": np.asarray(table.g_star, dtype=float),
        "g_starS": np.asarray(table.g_starS, dtype=float),
        "a": np.asarray(table.a, dtype=float),
        "metadata": metadata,
    }
    return _PBUF_LUT


_PBUF_LUT: dict[str, Any] | None = None


_MONITOR_CHOICES = list(available_monitor_modes())


def _monitor_arg(value: str | None) -> str | None:
    """Parse a monitor argument, allowing aliases to map to canonical modes."""
    if value is None:
        return None
    try:
        return normalize_monitor_mode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_model(model_name: str, overrides: dict[str, float | str]) -> object:
    normalized: dict[str, float | str] = {}
    for key, value in overrides.items():
        try:
            normalized[key] = float(value)
        except Exception:
            normalized[key] = value
    name = model_name.strip().lower()
    if name == "pbuf":
        lut = _load_pbuf_lut()
        return create_cosmos2_model("pbuf", lut=lut, **normalized)
    return create_cosmos2_model(name, **normalized)


def _make_model_factory(model_name: str) -> Callable[[dict[str, float]], object]:
    normalized = model_name.strip().lower()
    if normalized == "pbuf":
        def _factory(params: dict[str, float]) -> object:
            lut = _load_pbuf_lut()
            return create_cosmos2_model("pbuf", lut=lut, **params)
    else:
        def _factory(params: dict[str, float]) -> object:
            return create_cosmos2_model(normalized, **params)
    return _factory


def _make_joint_evaluator(
    model_name: str,
    dataset_names: Sequence[str],
    dataset_weights: dict[str, float],
) -> tuple[Callable[[dict[str, float]], float], Callable[[dict[str, float]], tuple[float, dict[str, Any]]]]:
    normalized = [name.strip().lower() for name in dataset_names if name.strip()]
    if not normalized:
        raise ValueError("At least one dataset must be provided via --datasets.")
    for name in normalized:
        if name not in COSMOS2_FIT_REGISTRY:
            raise ValueError(f"Unknown dataset/fit '{name}' requested.")

    specs: list[tuple[str, Callable[[Any], Any], float]] = []
    for name in normalized:
        specs.append((name, COSMOS2_FIT_REGISTRY[name], float(dataset_weights.get(name, 1.0))))

    factory = _make_model_factory(model_name)

    def evaluate(params: dict[str, float]) -> float:
        model = factory(params)
        total = 0.0
        for _, fit_fn, weight in specs:
            try:
                result = fit_fn(model)
                chi2 = result[0] if isinstance(result, tuple) else result
                if not math.isfinite(chi2):
                    return float("inf")
                total += weight * float(chi2)
            except Exception:
                return float("inf")
        return float(total)

    def evaluate_with_breakdown(params: dict[str, float]) -> tuple[float, dict[str, Any]]:
        model = factory(params)
        breakdown: dict[str, Any] = {}
        total = 0.0
        for name, fit_fn, weight in specs:
            try:
                result = fit_fn(model)
                chi2 = result[0] if isinstance(result, tuple) else result
                if not math.isfinite(chi2):
                    chi2 = float("inf")
                weighted = weight * float(chi2)
                total += weighted
                breakdown[name] = {"chi2": float(chi2), "weight": weight, "weighted_chi2": weighted}
            except Exception as exc:
                breakdown[name] = {"chi2": float("inf"), "weight": weight, "error": str(exc)}
                total = float("inf")
        return float(total), breakdown

    return evaluate, evaluate_with_breakdown


def _run_science_runner(args: argparse.Namespace) -> None:
    """Run the cosmos2 science runner using the unified runner."""
    from science_runner import _collect_paths, _collect_override_items, _interactive_confirm
    from cosmos2_science_runner import _make_progress_printer

    config_paths = _collect_paths(args.config, args.config_dir)
    if not config_paths:
        raise SystemExit("No science config files were provided.")

    for cfg_path in config_paths:
        config = ScienceRunConfig.from_path(cfg_path)
        if args.mode:
            config.mode = args.mode
        if args.engine:
            config.engine = args.engine
        if args.workers:
            config.engine_settings["workers"] = args.workers
        if args.monitor is not None:
            config.engine_settings["monitor"] = args.monitor
        if args.resume:
            config.engine_settings["resume"] = True
        override_models = _collect_override_items(args.override_models)
        override_fits = _collect_override_items(args.override_fits)
        try:
            if override_models:
                config.set_models(override_models)
            if override_fits:
                config.set_fits(override_fits)
        except ValueError as exc:
            raise SystemExit(str(exc))

        interactive = args.interactive or config.interactive
        if interactive:
            if not _interactive_confirm(config):
                print(f"Skipping {cfg_path}")
                continue

        runner = UnifiedScienceRunner(config, dry_run=args.dry_run)
        try:
            monitor_mode = normalize_monitor_mode(config.engine_settings.get("monitor"))
        except ValueError as exc:
            raise SystemExit(str(exc))

        if monitor_mode == "plugin":
            runner.execute()
        else:
            runner.execute(progress_callback=_make_progress_printer(config.run_name))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cosmos_cli.py", description="Cosmos command line interface")
    subparsers = parser.add_subparsers(dest="command")

    fit_parser = subparsers.add_parser("fit", help="Run a fit")
    fit_parser.add_argument(
        "target",
        choices=[
            "cmb",
            "sn",
            "sh0es",
            "sn_sh0es",
            "bao_iso",
            "bao_aniso",
            "cc",
            "rsd",
            "wl_s8",
            "lensing_cross",
            "galaxy_pk",
            "joint",
        ],
        help="Fit type to run",
    )
    fit_parser.add_argument("--model", required=True, choices=["pbuf", "lcdm"])
    fit_parser.add_argument("--param", action="append", default=[], help="Model parameter override (key=value)")
    fit_parser.add_argument("--datasets", help="Comma/space separated dataset names for joint fit target")
    fit_parser.add_argument(
        "--dataset-weight",
        action="append",
        default=[],
        help="Override dataset weights (dataset=weight) for joint fit (can be repeated)",
    )

    optim_parser = subparsers.add_parser("optimise", help="Run an optimisation")
    optim_parser.add_argument("--model", required=True, choices=["pbuf", "lcdm"], help="Model to optimise")
    optim_parser.add_argument("--datasets", required=True, help="Comma/space separated dataset names (e.g. 'cmb')")
    optim_parser.add_argument("--engine", default="grid_search", choices=["grid_search", "basin"], help="Optimisation engine")
    optim_parser.add_argument(
        "--dataset-weight",
        action="append",
        default=[],
        help="Override dataset weights (dataset=weight) when summing χ² (can be repeated)",
    )
    report_parser = subparsers.add_parser("report", help="Generate summary reports from a science run")
    report_parser.add_argument("--run", required=True, help="Path to the science run directory")
    report_parser.add_argument("--output-dir", help="Directory to place the generated report bundle")
    report_parser.add_argument(
        "--format",
        action="append",
        default=[],
        help="Report format to emit (json, html, latex, libre, csv). Can be repeated; defaults to json+html when unspecified. HTML format includes embedded plots.",
    )
    report_parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up legacy report files to avoid confusion"
    )
    optim_parser.add_argument("--samples", type=int, default=500, help="Sample count for grid search")
    optim_parser.add_argument("--seed", type=int, help="PRNG seed for optimisation")
    optim_parser.add_argument("--scatter", type=int, default=200, help="Scatter samples for basin engine")
    optim_parser.add_argument("--seeds", type=int, default=10, help="Seed count for basin engine")
    optim_parser.add_argument("--refine", type=int, default=50, help="Local descent steps per basin seed")
    optim_parser.add_argument("--threads", type=int, default=4, help="Worker count for basin engine")
    optim_parser.add_argument("--save-result", action="store_true", help="Persist optimisation_result.json")
    optim_parser.add_argument("--output", type=str, help="Optional path for optimisation_result.json")

    sanity_parser = subparsers.add_parser("sanity", help="Probe model and dataset sanity for specific parameters")
    sanity_parser.add_argument("--model", required=True, choices=["pbuf", "lcdm"], help="Model to evaluate")
    sanity_parser.add_argument("--datasets", required=False, default="cmb", help="Comma/space separated dataset names")
    sanity_parser.add_argument("--param", action="append", default=[], help="Model parameter override (key=value)")

    science_parser = subparsers.add_parser("science", help="Run a science config via science_runner")
    science_parser.add_argument("--config", "-c", action="append", default=[], help="Path to science config JSON/YAML")
    science_parser.add_argument("--config-dir", "-d", help="Directory containing science config sheets")
    science_parser.add_argument("--interactive", "-i", action="store_true", help="Prompt before each config")
    science_parser.add_argument("--dry-run", action="store_true", help="Build outputs without execution")
    science_parser.add_argument("--override-fits", action="append", help="Override joint fit definitions (comma separated)")
    science_parser.add_argument("--override-models", action="append", help="Override model list (comma separated)")
    science_parser.add_argument("--mode", choices=["fit", "scout"], help="Override mode from config")
    science_parser.add_argument("--engine", help="Engine override (cosmos2_basin/basin/threaded)")
    science_parser.add_argument("--workers", type=int, help="Worker processes for batch evaluation")
    science_parser.add_argument(
        "--monitor",
        nargs="?",
        const="simple",
        type=_monitor_arg,
        choices=_MONITOR_CHOICES,
        help="Enable monitoring during optimisation (options: ansi, plugin, textual).",
    )
    science_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.json when available")

    quantum_thermal_parser = subparsers.add_parser(
        "quantum-thermal",
        help="Generate a new Quantum-derived thermal table (uses configs/quantum/config.json)",
    )
    quantum_thermal_parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Optional dataset names to record in the thermal metadata (for provenance only)",
    )

    thermal_parser = subparsers.add_parser("thermal", help="Regenerate the PBUF thermal table cache")
    thermal_parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Optional dataset names to record in the thermal metadata (for provenance only)",
    )

    toolbox_parser = subparsers.add_parser("toolbox", help="Invoke the Cosmos toolbox helpers")
    toolbox_parser.add_argument(
        "action",
        choices=["data-sync", "quantum-ingest", "quantum-compact", "quantum-download"],
        help="Toolbox command to execute",
    )
    toolbox_parser.add_argument("--datasets", nargs="+", help="Dataset names to sync (data-sync)")
    toolbox_parser.add_argument(
        "--dataset-components",
        action="append",
        help="Forwarded to toolbox data-sync for dataset-specific component overrides.",
    )
    toolbox_parser.add_argument("--max-gcn", type=int, help="Limit GCN files (quantum-ingest)")
    toolbox_parser.add_argument("--summary", type=Path, help="Summary path (quantum-ingest)")
    toolbox_parser.add_argument("--output", type=Path, help="Normalized CSV output (quantum-ingest)")
    toolbox_parser.add_argument("--force-downloads", action="store_true", help="Redownload even when files exist (quantum-download)")
    toolbox_parser.add_argument("--skip-fermi", action="store_true", help="Skip Fermi downloads (quantum-download)")
    toolbox_parser.add_argument("--debug", action="store_true", help="Enable debug logging (quantum-download)")

    args = parser.parse_args(argv)

    if args.command == "science":
        _run_science_runner(args)
        return 0

    if args.command == "thermal":
        try:
            from cosmos2.pbuf.microphysics import THERMAL_CACHE_PATH, run_microphysics_bootstrap
        except Exception as exc:  # noqa: BLE001
            print(f"[cosmos2] Failed to import thermal bootstrap helpers: {exc}")
            return 1
        metadata = run_microphysics_bootstrap(datasets=args.datasets or [])
        print("[cosmos2] Thermal table regenerated")
        print(f" path : {THERMAL_CACHE_PATH}")
        print(f" beta : {metadata.get('beta')}")
        print(f" alpha_qm : {metadata.get('alpha_qm')}")
        print(f" eps0_base: {metadata.get('eps0_base')}")
        print(f" micro_hash: {metadata.get('micro_hash')}")
        return 0

    if args.command == "quantum-thermal":
        try:
            from cosmos2.pbuf.microphysics import THERMAL_CACHE_PATH, run_microphysics_bootstrap
        except Exception as exc:  # noqa: BLE001
            print(f"[cosmos2] Failed to import quantum thermal bootstrap helpers: {exc}")
            return 1
        metadata = run_microphysics_bootstrap(datasets=args.datasets or [])
        print("[cosmos2][quantum] Thermal table generated")
        print(f" path : {THERMAL_CACHE_PATH}")
        print(f" beta : {metadata.get('beta')}")
        print(f" alpha_qm : {metadata.get('alpha_qm')}")
        print(f" eps0_base: {metadata.get('eps0_base')}")
        print(f" micro_hash: {metadata.get('micro_hash')}")
        return 0

    if args.command == "toolbox":
        toolbox_args = [args.action]
        if args.action == "data-sync" and args.datasets:
            toolbox_args.extend(["--datasets", *args.datasets])
        if args.action == "data-sync" and args.dataset_components:
            for spec in args.dataset_components:
                toolbox_args.extend(["--dataset-components", spec])
        if args.action == "quantum-ingest":
            if args.max_gcn is not None:
                toolbox_args.extend(["--max-gcn", str(args.max_gcn)])
            if args.summary:
                toolbox_args.extend(["--summary", str(args.summary)])
            if args.output:
                toolbox_args.extend(["--output", str(args.output)])
        if args.action == "quantum-download":
            if args.force_downloads:
                toolbox_args.append("--force-downloads")
            if args.skip_fermi:
                toolbox_args.append("--skip-fermi")
            if args.debug:
                toolbox_args.append("--debug")
        run_toolbox(toolbox_args)
        return 0

    if args.command == "fit":
        params = parse_params(args.param)
        model = _build_model(args.model, params)
        if args.target == "joint":
            dataset_names = _parse_dataset_names(args.datasets or "")
            dataset_weights = _parse_dataset_weights(args.dataset_weight)
            _, evaluate_with_breakdown = _make_joint_evaluator(args.model, dataset_names, dataset_weights)
            chi2, breakdown = evaluate_with_breakdown({k: float(v) for k, v in params.items()})  # type: ignore[arg-type]
            print(f"Joint fit ({args.model.upper()})")
            print("-" * 40)
            print(f"total χ² : {chi2:8.3f}")
            for name in dataset_names:
                summary = breakdown.get(name, {})
                chi_val = summary.get("chi2")
                weight = summary.get("weight", 1.0)
                if chi_val is not None:
                    print(f"{name.upper():10s}: χ²={chi_val:8.3f} (w={weight})")
                elif summary.get("error"):
                    print(f"{name.upper():10s}: error={summary['error']}")
            return 0
        if args.target == "cmb":
            dataset = get_dataset("cmb")
            chi2, extras = run_cmb_fit(model, dataset)
            print(f"CMB distance prior fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2   : {chi2:8.3f}")
            _preview_vector("predictions", extras.get("predictions"))
            return 0

        if args.target == "sn":
            dataset = get_dataset("sn")
            chi2, extras = run_sn_pantheon_fit(model, dataset)
            print(f"SN Pantheon+ Fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2   : {chi2:8.3f}")
            print(f"data pts: {len(dataset['z'])}")
            _preview_vector("μ_model sample", extras.get("predictions"))
            return 0

        if args.target == "sh0es":
            dataset = get_dataset("sh0es")
            chi2, extras = run_sh0es_prior(model, dataset)
            print(f"SH0ES prior ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2       : {chi2:8.3f}")
            _preview_vector("H0 residual", extras.get("residuals"))
            return 0

        if args.target == "bao_iso":
            dataset = get_dataset("bao_iso")
            chi2, extras = run_bao_iso_fit(model, dataset)
            print(f"BAO isotropic fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2   : {chi2:8.3f}")
            print(f"data pts: {len(dataset['z'])}")
            _preview_vector("DV/r_d model sample", extras.get("predictions"))
            return 0

        if args.target == "bao_aniso":
            dataset = get_dataset("bao_aniso")
            chi2, extras = run_bao_aniso_fit(model, dataset)
            print(f"BAO anisotropic fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2   : {chi2:8.3f}")
            print(f"bins    : {len(dataset['z'])}")
            _preview_vector("model vector sample", extras.get("predictions"))
            return 0

        if args.target == "cc":
            dataset = get_dataset("cc")
            chi2, extras = run_cc_fit(model, dataset)
            print(f"Cosmic chronometer fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2   : {chi2:8.3f}")
            print(f"data pts: {len(dataset['z'])}")
            _preview_vector("H_model sample", extras.get("predictions"))
            return 0

        if args.target == "rsd":
            dataset = get_dataset("rsd")
            chi2, extras = run_rsd_fit(model, dataset)
            print(f"RSD fσ₈ fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2   : {chi2:8.3f}")
            print(f"data pts: {len(dataset['z'])}")
            _preview_vector("fσ₈_model sample", extras.get("predictions"))
            return 0

        if args.target == "wl_s8":
            dataset = get_dataset("wl_s8")
            chi2, extras = run_wl_s8_fit(model, dataset)
            print(f"WL S₈ fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2   : {chi2:8.3f}")
            _preview_vector("S₈_model sample", extras.get("predictions"))
            return 0

        if args.target == "lensing_cross":
            dataset = get_dataset("lensing_cross")
            chi2, extras = run_lensing_cross_fit(model, dataset)
            print(f"Lensing cross-correlation fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2       : {chi2:8.3f}")
            _preview_vector("A_model sample", extras.get("predictions"))
            return 0

        if args.target == "galaxy_pk":
            dataset = get_dataset("galaxy_pk")
            chi2, extras = run_galaxy_pk_fit(model, dataset)
            print(f"Galaxy P(k) compressed fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2   : {chi2:8.3f}")
            _preview_vector("model vector sample", extras.get("predictions"))
            return 0

        if args.target == "sn_sh0es":
            dataset_sn = get_dataset("sn")
            dataset_sh0es = get_dataset("sh0es")
            chi2_sn, _ = run_sn_pantheon_fit(model, dataset_sn)
            chi2_sh0es, _ = run_sh0es_prior(model, dataset_sh0es)
            total_chi2 = chi2_sn + chi2_sh0es
            print(f"SN Pantheon+ + SH0ES Joint Fit ({args.model.upper()})")
            print("-" * 40)
            print(f"chi^2 SN    : {chi2_sn:8.3f}")
            print(f"chi^2 SH0ES : {chi2_sh0es:8.3f}")
            print(f"total χ²    : {total_chi2:8.3f}")
            return 0

        return 1

    if args.command == "optimise":
        return _handle_optimise(args)

    if args.command == "report":
        return _handle_report(args)

    if args.command == "sanity":
        return _handle_sanity(args)

    parser.print_help()
    return 1


def _handle_optimise(args: argparse.Namespace) -> int:
    dataset_names = _parse_dataset_names(args.datasets)
    dataset_weights = _parse_dataset_weights(args.dataset_weight)
    bounds = load_bounds_for_model(args.model, dataset_names)
    evaluator, _ = _make_joint_evaluator(args.model, dataset_names, dataset_weights)
    model_config: dict[str, Any] = {
        "name": args.model,
        "bounds": bounds,
        "evaluator": evaluator,
        "fits": dataset_names,
        "fit_weights": dataset_weights,
        "rng_seed": args.seed,
    }
    if args.model.strip().lower() == "pbuf":
        model_config["model_kwargs"] = {"lut": _load_pbuf_lut()}
    if args.engine == "grid_search":
        model_config["grid_points"] = max(1, args.samples)
        model_config["n_batches"] = 1
        model_config["batch_size"] = max(1, args.scatter)
    else:
        model_config["n_batches"] = max(1, args.seeds)
        model_config["batch_size"] = max(1, args.refine)
        model_config["n_scatter"] = max(0, args.scatter)
    result = run_optimisation([model_config])

    _print_optimisation_summary(args.model, result)

    if args.save_result:
        output_path = Path(args.output or f"cosmos/models/{args.model}/optimisation_result.json")
        payload = {
            "model": args.model,
            "engine": args.engine,
            "datasets": dataset_names,
            "result": result,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved optimisation result to {output_path}")

    if dataset_weights:
        print("Dataset weight overrides:")
        for name, weight in sorted(dataset_weights.items()):
            print(f"  {name}: {weight}")

    return 0


def _handle_sanity(args: argparse.Namespace) -> int:
    dataset_names = _parse_dataset_names(args.datasets)
    params = parse_params(args.param)
    _, evaluate_with_breakdown = _make_joint_evaluator(args.model, dataset_names, {})
    chi2, breakdown = evaluate_with_breakdown({k: float(v) for k, v in params.items()})  # type: ignore[arg-type]

    if not math.isfinite(chi2):
        print(f"Sanity failed for {args.model.upper()} candidate ({chi2:.4e} χ²)")
        for dataset, summary in breakdown.items():
            if summary.get("error"):
                print(f"  - {dataset}: {summary['error']}")
        return 1

    print(f"Sanity checks passed ({chi2:.4f} χ²)")
    for dataset, summary in breakdown.items():
        summary_chi2 = summary.get("chi2")
        weight = summary.get("weight", 1.0)
        if summary_chi2 is not None:
            print(f"  {dataset.upper()} χ² = {summary_chi2:.6f} (weight={weight})")
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    """Handle the report command using the new standalone reporting system."""
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"[cosmos_cli] Error: Run directory {run_dir} does not exist")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    
    # Default to html when no formats specified (our system generates HTML)
    formats = set(args.format) if args.format else {"html"}
    
    print(f"[cosmos_cli] 🚀 Using STANDALONE REPORTING SYSTEM")
    print(f"[cosmos_cli] Run directory: {run_dir.name}")
    print(f"[cosmos_cli] Output directory: {output_dir}")
    print(f"[cosmos_cli] Format: {', '.join(formats)}")
    
    # Add the project root to the Python path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from reporting_system.report_cli import main as report_main
    except ImportError as e:
        print(f"[cosmos_cli] ❌ ERROR: Standalone reporting system not available: {e}")
        print("[cosmos_cli] 💡 Please ensure the reporting_system module is properly installed")
        return 1

    # Prepare arguments for the standalone reporting system
    sys.argv = [
        'report_cli.py',
        str(run_dir),
        '--output', str(output_dir / 'science_report.html')
    ]

    print("[cosmos_cli] 📊 Initializing Standalone Report Generator...")

    try:
        report_main()
    except Exception as e:
        print(f"[cosmos_cli] ❌ ERROR generating standalone report: {e}")
        print(f"[cosmos_cli] 💡 Check run directory and data integrity")
        return 1

    print(f"[cosmos_cli] ✅ STANDALONE REPORT GENERATED SUCCESSFULLY!")
    print(f"[cosmos_cli] 📁 Report location: {output_dir / 'science_report.html'}")

    # Get file size
    report_file = output_dir / 'science_report.html'
    file_size = report_file.stat().st_size if report_file.exists() else 0
    print(f"[cosmos_cli] 📊 File size: {file_size:,} bytes")

    print("[cosmos_cli] 🎨 Report sections:")
    print("  - 🔬 Hero Header (Run info, datasets)")
    print("  - 📊 Model Comparison (Performance analysis)")
    print("  - 🔬 Individual Models (LCDM, PBUF details)")
    print("  - 📈 Jackknife Analysis (Parameter stability)")
    print("  - 📋 Data Tables (Comprehensive data display)")
    print("  - 🎯 Conclusion (Recommendations)")
    print("  - ✅ Professional theme with exact example layout")

    return 0


def _parse_dataset_weights(raw: Sequence[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for token in raw:
        if "=" not in token:
            raise ValueError(f"--dataset-weight must be in the form key=value (got '{token}')")
        key, value = token.split("=", 1)
        if not key:
            raise ValueError(f"--dataset-weight missing dataset name in '{token}'")
        try:
            numeric = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid weight for dataset '{key}': {value}") from exc
        weights[key.strip().lower()] = numeric
    return weights


def _print_optimisation_summary(model_name: str, result: dict[str, Any]) -> None:
    print(f"Optimisation complete for {model_name.upper()}")
    models = result.get("models") or []
    if models:
        summary = models[0]
        best_chi2 = summary.get("weighted_chi2") or summary.get("best_chi2")
        if best_chi2 is not None:
            print(f"  Best χ²: {float(best_chi2):.6f}")
        best_params = summary.get("best_params") or summary.get("best_parameters")
        if best_params:
            print("  Best parameters:")
            for key in sorted(best_params):
                print(f"    {key}: {best_params[key]:.6g}")
        chi2_breakdown = summary.get("chi2_breakdown") or {}
        if chi2_breakdown:
            print("  Fit breakdown:")
            for fit_name, payload in chi2_breakdown.items():
                chi2_val = payload.get("chi2")
                weighted = payload.get("weighted_chi2", chi2_val)
                if chi2_val is not None and weighted is not None:
                    print(f"    {fit_name}: χ²={chi2_val:.4f} weighted={weighted:.4f}")
                else:
                    print(f"    {fit_name}: {payload}")


if __name__ == "__main__":
    raise SystemExit(main())
