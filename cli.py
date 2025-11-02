#!/usr/bin/env python3
"""
PBUF4 CLI - Main entry point for the PBUF cosmology framework.

This CLI provides subcommands for:
- dataset: download and convert cosmological datasets
- run: run individual fits (CMB, SN, BAO, etc.)
- fit: optimize cosmological parameters (joint, grid, comprehensive, etc.)
- report: generate scientific reports
- test: run validation tests

Usage:
    python cli.py dataset download --name planck2018_distance_priors
    python cli.py run cmb
    python cli.py fit joint --model lcdm --datasets cmb,sn_pantheon,bao_iso
    python cli.py report generate
    python cli.py fit grid --model pbuf
    python cli.py fit joint-comprehensive --model pbuf --datasets cmb,sn,bao_iso,bao_aniso,cc,rsd
    python cli.py test all
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Set


def create_model_from_params(params, model_type):
    """
    Create LCDM or PBUF model instance from parameter dictionary.

    Parameters
    ----------
    params : dict
        Parameter dictionary with keys like H0, Om0, Ok0, etc.
    model_type : str
        "lcdm" or "pbuf"

    Returns
    -------
    model : LCDM or PBUF instance
    """
    from cosmos.lcdm.model import LCDM
    from cosmos.pbuf.model import PBUF

    # Extract parameters with defaults
    H0 = params.get("H0", 67.36)
    Om0 = params.get("Om0", 0.3153)
    Ok0 = params.get("Ok0", 0.0)
    Or0 = params.get("Or0", 9.2e-5)
    Obh2 = params.get("Obh2", 0.02237)

    if model_type == "lcdm":
        Ol0 = params.get("Ol0", 0.6847)
        return LCDM(
            omega_m=Om0,
            omega_lambda=Ol0,
            h=H0/100.0,
            omega_k=Ok0,
            omega_r=Or0,
            omega_b=Obh2/(H0/100.0)**2,
            T_cmb=2.7255
        )
    elif model_type == "pbuf":
        alpha = params.get("alpha", 1e-3)
        Rmax = params.get("Rmax", 1.0e8)
        k_sat = params.get("k_sat", 1.0)
        eps0 = params.get("eps0", 0.7)
        n_alpha = params.get("n_alpha", 0.0)
        n_eps = params.get("n_eps", 0.0)
        n_R = params.get("n_R", 0.0)
        return PBUF(
            omega_m=Om0,
            h=H0/100.0,
            alpha=alpha,
            Rmax=Rmax,
            k_sat=k_sat,
            eps0=eps0,
            n_alpha=n_alpha,
            n_eps=n_eps,
            n_R=n_R,
            omega_k=Ok0,
            omega_r=Or0,
            omega_b=Obh2/(H0/100.0)**2,
            T_cmb=2.7255
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_default_params():
    """Get default parameters for LCDM and PBUF models."""
    return {
        "lcdm": {
            "H0": 67.36,
            "Om0": 0.3153,
            "Ok0": 0.0,
            "Ol0": 0.6847,
        "Or0": 9.2e-5,
        "Obh2": 0.02237,
    },
    "pbuf": {
        "H0": 67.36,
        "Om0": 0.3153,
        "Ok0": 0.0,
        "Ol0": 0.0,  # No cosmological constant
        "Or0": 9.2e-5,
        "Obh2": 0.02237,
        "alpha": 5e-4,
        "Rmax": 1.0e8,
        "k_sat": 1.0,
        "eps0": 0.7,
        "n_alpha": 0.0,
        "n_eps": 0.0,
        "n_R": 0.0,
    }
}


def merge_params(defaults, overrides):
    """Merge parameter overrides with defaults."""
    if overrides is None:
        return defaults
    result = defaults.copy()
    result.update(overrides)
    return result


# Import CLI modules (will implement these)
def dataset_download(args):
    """Handle dataset download command."""
    print(f"📡 Downloading dataset: {args.name}")
    from data.downloader import download_dataset
    metadata = download_dataset(args.name)
    print(f"✅ Downloaded {args.name} to data/raw/{args.name}/")
    print(f"   Source: {metadata.get('source_url', 'N/A')}")


def dataset_convert(args):
    """Handle dataset convert command."""
    print(f"🔄 Converting dataset: {args.source}")
    from data.converter import convert_dataset

    # Convert type string to proper format if provided
    dataset_type = None
    if hasattr(args, 'type') and args.type:
        if args.type == "auto":
            dataset_type = None  # Let auto-detection handle it
        else:
            type_mapping = {
                'sn_pantheon': 'SN',
                'sn_sh0es': 'SH0ES',
                'sn': 'SN',
                'bao': 'BAO',
                'cc': 'CC',
                'rsd': 'RSD',
                'cmb': 'CMB',
                'sh0es': 'SH0ES'
            }
            dataset_type = type_mapping.get(args.type.lower(), args.type.upper())

    data_dict = convert_dataset(args.source, args.output, dataset_type)
    print(f"✅ Converted to {args.output}")
    print(f"   Points: {data_dict.get('n_data', 'N/A')}")
    print(f"   z-range: {data_dict.get('z_min', 'N/A')} - {data_dict.get('z_max', 'N/A')}")


def run_fit(args):
    """Handle run <fit> command."""
    print(f"🏃 Running fit: {args.fit}")
    fit_name = args.fit

    if fit_name == "science":
        run_science(args)
        return

    # Handle model specification - support both new and legacy syntax
    overrides = {}

    # Legacy syntax: --lcdm and --pbuf
    if args.lcdm:
        overrides["lcdm"] = json.loads(args.lcdm)
    if args.pbuf:
        overrides["pbuf"] = json.loads(args.pbuf)

    # New syntax: --model and --parameters
    if hasattr(args, 'model') and args.model and hasattr(args, 'parameters') and args.parameters:
        if args.model == "both":
            # Apply parameters to both models
            params = json.loads(args.parameters)
            overrides["lcdm"] = params
            overrides["pbuf"] = params
        elif args.model in ["lcdm", "pbuf"]:
            # Apply parameters to specified model
            overrides[args.model] = json.loads(args.parameters)
        else:
            print(f"❌ Unknown model: {args.model}")
            return

    # Default to both models if no parameters specified
    if not overrides:
        overrides = {"lcdm": {}, "pbuf": {}}

    # Import the appropriate runner
    if fit_name == "cmb":
        from cosmos.fits.cmb.runner import run_fit as run_cmb_fit
    elif fit_name == "sn":
        from cosmos.fits.sn.runner import run_fit as run_sn_fit
    elif fit_name == "sn_pantheon":
        from cosmos.fits.sn.pantheon.runner import run_fit as run_sn_pantheon_fit
    elif fit_name == "sn_sh0es":
        from cosmos.fits.sn.sh0es.runner import run_fit as run_sn_sh0es_fit
    elif fit_name == "bao_iso":
        from cosmos.fits.bao.runner import run_fit as run_bao_iso_fit
    elif fit_name == "bao_aniso":
        from cosmos.fits.bao.aniso_runner import run_fit as run_bao_aniso_fit
    elif fit_name == "cc":
        from cosmos.fits.cc.runner import run_fit as run_cc_fit
    elif fit_name == "rsd":
        from cosmos.fits.rsd.runner import run_fit as run_rsd_fit
    else:
        print(f"❌ Unknown fit type: {fit_name}")
        return

    # Run the fit
    try:
        if fit_name == "cmb":
            run_cmb_fit(overrides)
        elif fit_name == "sn":
            run_sn_fit(overrides)
        elif fit_name == "sn_pantheon":
            run_sn_pantheon_fit(overrides)
        elif fit_name == "sn_sh0es":
            run_sn_sh0es_fit(overrides)
        elif fit_name == "bao_iso":
            run_bao_iso_fit(overrides)
        elif fit_name == "bao_aniso":
            run_bao_aniso_fit(overrides)
        elif fit_name == "cc":
            run_cc_fit(overrides)
        elif fit_name == "rsd":
            run_rsd_fit(overrides)
    except ImportError:
        print(f"❌ Runner not implemented yet for {fit_name}")


def _print_joint_summary(payload: Mapping[str, Any], models_to_report: List[str], *, show_phase6a: bool = False) -> None:
    artifact = payload.get("artifact_path")
    dataset_info = payload.get("run_meta", {}).get("datasets", {})
    dataset_names = list(dataset_info.keys())
    if artifact:
        print(f"💾 Joint capture saved to {artifact}")
    else:
        print("💾 Joint capture completed (no artifact path reported)")
    if dataset_names:
        print(f"   Datasets: {', '.join(dataset_names)}")
    comparison = payload.get("comparison")
    if comparison:
        chi_delta = comparison.get("chi2", {}).get("delta")
        if chi_delta is not None:
            print(f"   Δχ² (PBUF − LCDM): {chi_delta:.3f}")
        aic_delta = comparison.get("AIC", {}).get("delta")
        if aic_delta is not None:
            print(f"   ΔAIC: {aic_delta:.3f}")
        bic_delta = comparison.get("BIC", {}).get("delta")
        if bic_delta is not None:
            print(f"   ΔBIC: {bic_delta:.3f}")
    print()

    for model in models_to_report:
        info = payload.get("best_fit", {}).get(model)
        if not info:
            continue
        label = model.upper()
        chi2_total = info.get("chi2_total")
        dof = info.get("degrees_of_freedom", {}).get("dof", 0)
        chi2_red = chi2_total / max(dof, 1) if chi2_total is not None else None
        print(f"{label} best fit:")
        if chi2_total is not None:
            print(f"   χ²_total = {chi2_total:.3f}  (dof={dof}, χ²_red={chi2_red:.3f})")
        breakdown = info.get("breakdown", {})
        if breakdown:
            print("   Per-dataset χ²:")
            per_dataset = info.get("degrees_of_freedom", {}).get("per_dataset", {})
            for name, value in breakdown.items():
                entry = per_dataset.get(name, {})
                n_data = entry.get("n_data")
                extra = f" (n={n_data})" if isinstance(n_data, int) else ""
                print(f"      {name:10s}: {value:.3f}{extra}")
        if show_phase6a:
            phase6a = info.get("physics_flags", {}).get("phase6a")
            if isinstance(phase6a, dict):
                status = phase6a.get("status")
                note = phase6a.get("note")
                if status is None and note:
                    print(f"   Phase-6a: {note}")
                else:
                    status_txt = "pass" if status else "fail"
                    print(f"   Phase-6a: {status_txt}")
                    if note and status:
                        print(f"      {note}")
                    if status:
                        omega_min = phase6a.get("omega_sigma_min")
                        rho_max = phase6a.get("rho_el_over_H2_max")
                        knee_max = phase6a.get("knee_ratio_max")
                        details = []
                        if omega_min is not None:
                            details.append(f"min Ωσ={omega_min:.3e}")
                        if rho_max is not None:
                            details.append(f"max ρ_el/H²={rho_max:.3e}")
                        if knee_max is not None:
                            details.append(f"max knee={knee_max:.3e}")
                        if details:
                            print(f"      {'; '.join(details)}")
        print()

    print("✅ Joint capture complete.")


def run_joint(args):
    """Handle joint command using the comprehensive capture pipeline."""
    dataset_tokens = _dataset_arg_to_list(getattr(args, "datasets", None))
    normalized = _normalize_joint_dataset_names(dataset_tokens)
    if dataset_tokens and normalized is not None and len(normalized) == 0:
        print("❌ No recognized datasets from --datasets; nothing to run.")
        return

    output_path = Path(args.output).expanduser() if getattr(args, "output", None) else None
    quiet = getattr(args, "quiet", False)

    from cosmos.fits.joint import run_joint_capture

    payload = run_joint_capture(
        datasets=normalized,
        output_path=output_path,
        verbose=not quiet,
    )

    requested_model = getattr(args, "model", "both").lower()
    available_models = list(payload.get("best_fit", {}).keys())
    if requested_model == "both":
        models_to_report = available_models
    elif requested_model in available_models:
        models_to_report = [requested_model]
    else:
        print(f"⚠️  Unknown model '{requested_model}', showing all available results.")
        models_to_report = available_models

    _print_joint_summary(payload, models_to_report, show_phase6a=True)


def fit_run_model(args):
    """Handle fit run <model> command."""
    from cosmos.fits.joint.runner import run_individual_fits

    model_type = args.model.lower()
    if model_type not in ["lcdm", "pbuf"]:
        print(f"❌ Unknown model type: {model_type}")
        print("   Available models: lcdm, pbuf")
        return

    print(f"🏃 Running {model_type.upper()} fits across all available datasets...")
    print("=" * 60)

    # Load model overrides
    overrides = {}
    if hasattr(args, 'lcdm') and args.lcdm:
        overrides["lcdm"] = json.loads(args.lcdm)
    if hasattr(args, 'pbuf') and args.pbuf:
        overrides["pbuf"] = json.loads(args.pbuf)

    # Apply overrides only for the requested model
    model_overrides = {model_type: overrides.get(model_type, {})}

    # Run individual fits
    results = run_individual_fits(model_overrides, verbose=True)

    if not results:
        print("❌ No datasets available for fitting")
        return

    print(f"\n📊 {model_type.upper()} Fit Results Summary:")
    print("=" * 50)

    total_chi2 = 0
    total_dof = 0
    valid_fits = 0

    for dataset, model_results in results.items():
        if model_type in model_results and "error" not in model_results[model_type]:
            result = model_results[model_type]
            chi2 = result["chi2"]
            n_data = result["n_data"]
            n_params = result["n_params"]
            dof = max(n_data - n_params, 1)
            chi2_red = chi2 / dof

            total_chi2 += chi2
            total_dof += dof
            valid_fits += 1

            print(f"   {dataset:12} | χ² = {chi2:8.3f} | χ²_red = {chi2_red:6.3f} | dof = {dof:3d}")

    if valid_fits > 0:
        chi2_red_total = total_chi2 / total_dof
        print(f"\n   {'TOTAL':12} | χ² = {total_chi2:8.3f} | χ²_red = {chi2_red_total:6.3f} | dof = {total_dof:3d}")
        print(f"   Datasets: {valid_fits} | Model: {model_type.upper()}")
    else:
        print(f"❌ No valid {model_type.upper()} fits completed")

    # Save summary
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = results_dir / f"{model_type}_summary_{timestamp}.json"

    with open(summary_file, "w") as f:
        json.dump({
            "model": model_type,
            "timestamp": timestamp,
            "results": results,
            "summary": {
                "total_chi2": total_chi2,
                "total_dof": total_dof,
                "chi2_reduced_total": chi2_red_total,
                "valid_fits": valid_fits,
                "model_overrides": model_overrides
            }
        }, f, indent=2)

    print(f"\n💾 Saved summary to {summary_file}")


def _parse_csv_list(value: Optional[str], *, default: Optional[List[str]] = None) -> List[str]:
    if not value:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


def run_science(args) -> None:
    from cosmos.optim.science_runner import ScienceRunner, ScienceConfigError, load_config

    config_path = Path(args.config or "configs/science_run.json").expanduser()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ScienceConfigError, json.JSONDecodeError) as exc:
        print(f"❌ Failed to load science configuration: {exc}")
        return

    if args.science_root:
        config["output_root"] = str(Path(args.science_root).expanduser())

    resume_path: Optional[Path] = None
    if args.resume_dir:
        resume_path = Path(args.resume_dir).expanduser()

    runner = ScienceRunner(config)

    print(f"🧪 Launching science run '{config['run_id']}'")
    print(f"   Models: {', '.join(config['models'])}")
    print(f"   Scenarios: {', '.join(s['id'] for s in config['scenarios'])}")

    try:
        summary = runner.run(fresh=bool(args.fresh), resume_dir=resume_path)
    except Exception as exc:  # pragma: no cover
        print(f"❌ Science run failed: {exc}")
        return

    run_dir = runner.run_dir or Path(config.get("output_root", "data/science_runs"))
    print(f"✅ Science run completed in {run_dir}")
    runtime = summary.get("runtime_seconds")
    if runtime is not None:
        print(f"   Runtime: {float(runtime):.2f}s")
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        print(f"   Summary: {summary_path}")


def report_generate(args):
    """Handle report generate command."""
    from reports.report_pipeline import build_full_report

    formats = _parse_csv_list(args.formats, default=["html", "md", "pdf", "json"])
    models = _parse_csv_list(args.models, default=["lcdm", "pbuf"])
    science_root = args.science_root or "data/science_runs"
    output_dir = args.output or "reports/output"
    verbose = not getattr(args, "quiet", False)

    if verbose:
        print(f"📊 Generating reports in formats: {', '.join(formats)}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Science runs root: {science_root}")
        print(f"   Output directory: {output_dir}")

    try:
        result = build_full_report(
            models=models,
            output_dir=output_dir,
            formats=formats,
            science_run_root=science_root,
            verbose=verbose,
        )
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(f"❌ Missing dependency: {missing}. Install it (e.g. pip install {missing}) and retry.")
        return
    except Exception as exc:  # pragma: no cover
        print(f"❌ Report generation failed: {exc}")
        return

    if verbose:
        available = ", ".join(result.get("formats", []))
        runs = ", ".join(result.get("science_runs", [])) or "none"
        print(f"✅ Reports ready under {result.get('output')}")
        print(f"   Formats: {available}")
        print(f"   Science runs processed: {runs}")


def fit_joint_comprehensive(args):
    """Handle comprehensive joint capture with dataset selection."""
    dataset_tokens = _dataset_arg_to_list(getattr(args, "datasets", None))
    normalized = _normalize_joint_dataset_names(dataset_tokens)
    if dataset_tokens and normalized is not None and len(normalized) == 0:
        print("❌ No recognized datasets from --datasets; nothing to run.")
        return

    output_path = Path(args.output).expanduser() if getattr(args, "output", None) else None
    quiet = getattr(args, "quiet", False)

    from cosmos.fits.joint import run_joint_capture

    payload = run_joint_capture(
        datasets=normalized,
        output_path=output_path,
        verbose=not quiet,
    )

    requested_model = getattr(args, "model", "pbuf").lower()
    available_models = list(payload.get("best_fit", {}).keys())
    if requested_model == "both":
        models_to_report = available_models
    elif requested_model in available_models:
        models_to_report = [requested_model]
    else:
        print(f"⚠️  Unknown model '{requested_model}', showing all available results.")
        models_to_report = available_models

    _print_joint_summary(payload, models_to_report, show_phase6a=True)

    traces = payload.get("optimizer_trace", {})
    for model in models_to_report:
        trace = traces.get(model, {})
        counters = trace.get("counters")
        if counters:
            print(
                f"{model.upper()} optimizer evaluations: total={counters.get('total')}, "
                f"accepted={counters.get('accepted')}, rejected={counters.get('rejected')}"
            )
        top = trace.get("top_evaluations") or []
        if top:
            print(f"{model.upper()} top candidates:")
            for entry in top[:3]:
                chi2 = entry.get("chi2")
                params = entry.get("params") or {}
                if params:
                    summary = ", ".join(f"{k}={v:.3g}" for k, v in params.items())
                    print(f"   χ²={chi2:.3f} | {summary}")
                else:
                    print(f"   χ²={chi2:.3f}")
        print()

def test_all(args):
    """Handle test all command."""
    print("🧪 Running all tests")
    import subprocess
    import glob

    test_files = glob.glob("tests/test_*.py")
    if not test_files:
        print("❌ No test files found")
        return

    passed = 0
    failed = 0

    for test_file in sorted(test_files):
        print(f"\n🔬 Running {test_file}...")
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(f"✅ {test_file} PASSED")
                passed += 1
            else:
                print(f"❌ {test_file} FAILED")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                failed += 1

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"💥 {test_file} ERROR: {e}")
            failed += 1

    print("\n📋 Test Summary:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total:  {passed + failed}")

    if failed > 0:
        print(f"\n❌ {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")


def _dataset_arg_to_list(dataset_str):
    """Translate CLI dataset string into a normalized list."""
    if not dataset_str:
        return None
    dataset_str = dataset_str.strip().lower()
    if dataset_str in ("default", "base"):
        return None
    if dataset_str == "all":
        from cosmos.optim.grid_pipeline import BASE_DATASETS, BAO_DATASETS
        return list(BASE_DATASETS + BAO_DATASETS)
    return [chunk.strip().lower() for chunk in dataset_str.split(",") if chunk.strip()]


JOINT_DATASET_ALIASES: Dict[str, str] = {
    "cmb": "cmb",
    "sn": "pantheon",
    "sn_pantheon": "pantheon",
    "sn_pantheon_abs": "pantheon",
    "pantheon": "pantheon",
    "pantheonplus": "pantheon",
    "pantheon+": "pantheon",
    "pantheon_abs": "pantheon",
    "sn_sh0es": "sh0es",
    "sh0es": "sh0es",
    "bao": "iso",
    "bao_iso": "iso",
    "iso": "iso",
    "bao_aniso": "aniso",
    "aniso": "aniso",
    "cc": "cc",
    "chronometer": "cc",
    "rsd": "rsd",
    "fsigma8": "rsd",
}


def _normalize_joint_dataset_names(tokens: Optional[List[str]]) -> Optional[List[str]]:
    """Map CLI dataset tokens to joint-capture canonical keys."""
    if tokens is None:
        return None
    normalized: List[str] = []
    for raw in tokens:
        token = raw.strip().lower().replace("-", "_")
        canonical = JOINT_DATASET_ALIASES.get(token)
        if canonical is None:
            print(f"⚠️  Unknown dataset '{raw}', skipping.")
            continue
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _load_grid_config(path: str):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Grid configuration file must contain a JSON object.")
    return data


def _print_grid_summary(result):
    model_label = result.get("model_type", "unknown").upper()
    print(f"\n🎯 {model_label} grid evaluation complete")
    total = result.get('num_evaluations', 0)
    valid = result.get('num_valid', 0)
    invalid = result.get('num_invalid', total - valid)
    print(f"   Evaluations: {total}  (valid: {valid}, invalid: {invalid})")
    refined = result.get("refined_evaluations", 0)
    if refined:
        print(f"   Refinement evaluations: {refined}")
    best = result.get("best")
    if best:
        chi2_total = best.get("chi2_total", float("nan"))
        status = best.get("status", "unknown")
        print(f"   Best χ²_total: {chi2_total:.3f} (status: {status})")
        breakdown = best.get("chi2_breakdown") or {}
        if breakdown:
            print("   Dataset breakdown:")
            for name, value in breakdown.items():
                print(f"      {name:8s}: {value:.3f}")
        params = best.get("params") or {}
        if params:
            print("   Parameters:")
            for key, value in params.items():
                print(f"      {key} = {value:.6g}")
        validation = best.get("validation") or {}
        reasons = validation.get("reasons") or []
        if reasons:
            print("   Validation notes:")
            for reason in reasons:
                print(f"      - {reason}")
        errors = best.get("dataset_errors") or {}
        if errors:
            print("   Dataset warnings:")
            for name, msg in errors.items():
                print(f"      {name}: {msg}")
    else:
        print("   No valid cosmologies found on this grid.")
    results_file = result.get("results_file")
    if results_file:
        print(f"   Results file: {results_file}")


def _format_parameter_value(value: float) -> str:
    """Format parameter values to align with parameter_defaults style."""
    formatted = f"{value:.8g}"
    if "e" not in formatted and "E" not in formatted and "." not in formatted:
        formatted += ".0"
    return formatted


def _update_parameter_defaults(model: str, fiducial: Mapping[str, float], *, dry_run: bool = False) -> Set[str]:
    """
    Persist fiducial parameters into cosmos/optim/parameter_defaults.py.

    Returns the set of keys that were changed. When dry_run is True, no file
    modification is performed and the function simply reports the intended updates.
    """
    from cosmos.optim import parameter_defaults as defaults

    target_name = "PBUF_PARAMETER_DEFAULTS" if model == "pbuf" else "LCDM_PARAMETER_DEFAULTS"
    current_defaults = getattr(defaults, target_name, {})
    if not isinstance(current_defaults, Mapping):
        return set()

    updates: Dict[str, float] = {}
    for key, value in current_defaults.items():
        if key not in fiducial:
            continue
        new_value = float(fiducial[key])
        try:
            current_value = float(value)
        except (TypeError, ValueError):
            current_value = new_value
        if not math.isclose(new_value, current_value, rel_tol=1e-12, abs_tol=1e-12):
            updates[key] = new_value

    if not updates:
        return set()

    if dry_run:
        print("\n🛈 Dry run: skipping parameter_defaults.py update.")
        print("   Proposed updates:")
        for key in sorted(updates):
            print(f"   - {key:6s} = {_format_parameter_value(updates[key])}")
        return set()

    defaults_path = Path("cosmos/optim/parameter_defaults.py")
    original_text = defaults_path.read_text(encoding="utf-8")
    lines = original_text.splitlines()

    in_triple_quote = False
    active_block = None
    brace_depth = 0
    changed_keys: Set[str] = set()

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if not in_triple_quote:
            if active_block is None:
                if stripped.startswith(f"{target_name} ="):
                    active_block = target_name
                    brace_depth = line.count("{") - line.count("}")
            else:
                stripped_line = line.lstrip()
                for key, new_value in updates.items():
                    if stripped_line.startswith(f'"{key}"'):
                        indent_length = len(line) - len(stripped_line)
                        indent = line[:indent_length]
                        left, right = stripped_line.split(":", 1)
                        comma_idx = right.find(",")
                        if comma_idx == -1:
                            continue
                        after_comma = right[comma_idx:]
                        formatted_value = _format_parameter_value(new_value)
                        replacement = f"{indent}{left}: {formatted_value}{after_comma}"
                        lines[idx] = replacement
                        line = replacement
                        changed_keys.add(key)
                        break
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    active_block = None

        quote_count = line.count('"""')
        if quote_count % 2 == 1:
            in_triple_quote = not in_triple_quote

    if not changed_keys:
        return set()

    updated_text = "\n".join(lines)
    if original_text.endswith("\n"):
        updated_text += "\n"
    defaults_path.write_text(updated_text, encoding="utf-8")
    return changed_keys


def fit_grid_pipeline(args):
    """Run the deterministic grid evaluator through the CLI."""
    from cosmos.optim.grid_pipeline import run_dual_grid_search, run_grid_search

    dataset_list = _dataset_arg_to_list(args.datasets)
    include_bao = args.include_bao
    grid_config = _load_grid_config(args.grid_config) if args.grid_config else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    tag = args.tag
    refine_top = max(0, args.refine_top)
    refine_fraction = max(0.0, args.refine_fraction)
    refine_points = max(1, args.refine_points)

    def _grid_for(model_name: str):
        if grid_config is None:
            return None
        if "lcdm" in grid_config or "pbuf" in grid_config:
            return grid_config.get(model_name)
        return grid_config

    if args.model == "both":
        print("🏁 Running grid evaluation for LCDM and PBUF")
        combined = run_dual_grid_search(
            datasets=dataset_list,
            include_bao=include_bao,
            grid_lcdm=_grid_for("lcdm"),
            grid_pbuf=_grid_for("pbuf"),
            workers=args.workers,
            output_dir=output_dir,
            tag=tag,
            refine_top=refine_top,
            refine_fraction=refine_fraction,
            refine_points=refine_points,
        )
        _print_grid_summary(combined["lcdm"])
        _print_grid_summary(combined["pbuf"])
        delta = combined.get("delta_chi2")
        if delta is not None:
            print(f"\nΔχ² (PBUF − LCDM): {delta:.3f}")
    else:
        result = run_grid_search(
            args.model,
            datasets=dataset_list,
            include_bao=include_bao,
            grid=_grid_for(args.model),
            workers=args.workers,
            output_dir=output_dir,
            tag=tag,
            refine_top=refine_top,
            refine_fraction=refine_fraction,
            refine_points=refine_points,
        )
        _print_grid_summary(result)


def _load_seed_parameters(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Seed JSON must contain a JSON object.")
    normalized: Dict[str, float] = {}
    for key, value in data.items():
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Seed parameter '{key}' must be numeric (got {value!r}).") from exc
    return normalized


def _default_coordinate_datasets(model: str, include_bao: bool) -> List[str]:
    base = ["cmb", "sn_pantheon", "bao_iso"]
    if model == "lcdm":
        # For LCDM we often compare directly to SH0ES; keep defaults modest for now.
        base = ["cmb", "sn_pantheon", "bao_iso"]
    datasets = list(base)
    if include_bao:
        for name in ("bao_iso", "bao_aniso"):
            if name not in datasets:
                datasets.append(name)
    return datasets


def fit_coordinate_optimizer(args):
    """Drive the coordinate descent basin walker via the CLI."""
    from cosmos.optim.coord_optimizer import (
        CoordinateBasinWalker,
        DEFAULT_REFERENCES,
    )
    from cosmos.optim.coord_optimizer.observers import RecordingObserver
    from cosmos.optim.chi2_targets import load_chi2_targets
    from reports.basin_plotter import generate_basin_plots

    model = args.model.lower()

    dataset_list = _dataset_arg_to_list(args.datasets)
    if dataset_list is None:
        dataset_list = _default_coordinate_datasets(model, args.include_bao)
    else:
        dedup: List[str] = []
        for name in dataset_list:
            if name not in dedup:
                dedup.append(name)
        dataset_list = dedup
        if args.include_bao:
            for name in ("bao_iso", "bao_aniso"):
                if name not in dataset_list:
                    dataset_list.append(name)

    if not dataset_list:
        print("❌ No datasets specified for coordinate optimization.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference = dict(DEFAULT_REFERENCES[model])
    if args.seed_json:
        seed_path = Path(args.seed_json)
        if not seed_path.exists():
            print(f"❌ Seed JSON not found: {seed_path}")
            return
        try:
            overrides = _load_seed_parameters(seed_path)
        except Exception as exc:  # noqa: BLE001 - surface parsing errors to CLI
            print(f"❌ Failed to load seed parameters: {exc}")
            return
        reference.update(overrides)
    if args.eps0 is not None:
        reference["eps0"] = float(args.eps0)

    second_pass_params = tuple() if args.skip_second_pass else None

    progress_enabled = not args.no_progress
    chi2_registry = None
    targets_path: Optional[Path] = None
    if args.chi2_targets:
        targets_path = Path(args.chi2_targets)
    else:
        candidate = Path("config/chi2_targets.json")
        if candidate.exists():
            targets_path = candidate
    if targets_path and targets_path.exists():
        try:
            chi2_registry = load_chi2_targets(targets_path, model)
        except ValueError as exc:
            print(f"⚠️  Failed to load χ² targets from {targets_path}: {exc}")
            chi2_registry = None

    record_dir = args.basin_record_dir or args.basin_plot_dir
    observers: List[Any] = []
    record_observer: Optional[RecordingObserver] = None
    if record_dir:
        record_observer = RecordingObserver(record_dir, auto_run_subdir=True)
        observers.append(record_observer)

    walker = CoordinateBasinWalker(
        model_type=model,
        datasets=dataset_list,
        enforce_phase6a=args.phase6a or (model == "pbuf"),
        delta_chi2=float(args.delta_chi2),
        reference_params=reference,
        second_pass_params=second_pass_params,
        verbose=not args.quiet,
        progress=progress_enabled,
        max_workers=args.workers,
        improvement_tol=float(args.improvement_tol),
        max_cycles=int(args.max_cycles),
        chi2_targets=chi2_registry,
        observers=observers or None,
    )

    if progress_enabled and not walker.progress:
        print("⚠️  Requested progress bars but tqdm is not installed; continuing without them.")

    if chi2_registry is not None and not chi2_registry.is_empty():
        resolved = targets_path.resolve() if targets_path is not None else Path("config/chi2_targets.json").resolve()
        print(f"🎯 Using χ² targets from {resolved}")

    if args.converge:
        if not args.quiet:
            print(
                f"⚙️  Convergence mode enabled (max cycles={args.max_cycles}, "
                f"Δχ² tol={float(args.improvement_tol):g}, workers={walker.max_workers})"
            )
        result = walker.run_with_convergence()
    else:
        result = walker.run()

    island_result = None
    if args.island_samples and args.island_samples > 0:
        try:
            island_result = walker.find_island_center(
                result,
                num_samples=int(args.island_samples),
                chi2_delta=float(args.island_delta),
                seed=args.island_seed,
            )
            result["island_center"] = island_result
        except ValueError as exc:
            print(f"⚠️  Island center search skipped: {exc}")

    output_path.write_text(json.dumps(result, indent=2))

    trace_path: Optional[Path] = None
    generated_plots: Dict[str, Path] = {}
    basin_plot_error: Optional[str] = None

    if record_observer:
        if record_observer.last_trace_path is not None:
            trace_path = Path(record_observer.last_trace_path)
        elif record_observer.last_run_dir is not None:
            trace_path = Path(record_observer.last_run_dir) / record_observer.filename

    if record_observer and args.basin_plot_dir:
        plot_dir = Path(args.basin_plot_dir)
        if trace_path and trace_path.exists():
            try:
                generated = generate_basin_plots(trace_path, plot_dir)
                generated_plots = {key: Path(value) for key, value in generated.items()}
            except Exception as exc:
                basin_plot_error = str(exc)
        else:
            basin_plot_error = "trace file unavailable"

    print("✅ Coordinate optimizer completed")
    print(f"   Model: {model.upper()}")
    print(f"   Datasets: {', '.join(dataset_list)}")
    print(f"   Output: {output_path.resolve()}")
    print(f"   Phase 6a: {'enforced' if result.get('phase6a_enforced') else 'skipped'}")
    print(f"   Δχ² tolerance: {args.delta_chi2:g}")
    print(f"   Second pass: {'enabled' if walker.second_pass_params else 'disabled'}")
    print(f"   Workers: {walker.max_workers}")
    if args.converge:
        convergence = result.get("convergence") or {}
        converged_txt = "yes" if convergence.get("converged") else "no"
        print(
            f"   Convergence: {converged_txt} "
            f"(cycles {convergence.get('cycles_completed', 'n/a')}/{walker.max_cycles})"
        )
    if args.seed_json:
        print(f"   Seed: {Path(args.seed_json).resolve()}")
    else:
        print("   Seed: default reference parameters")

    fiducial = result.get("fiducial_params") or {}
    if fiducial:
        print("\n📌 Fiducial parameters:")
        for key, value in fiducial.items():
            print(f"   {key:6s} = {value:.6g}")

    fiducial_chi2 = result.get("fiducial_chi2")
    if isinstance(fiducial_chi2, (int, float)):
        print(f"\n   Fiducial χ²_total: {fiducial_chi2:.3f}")

    if chi2_registry is not None and not chi2_registry.is_empty():
        print("\n🎯 Target expectations:")
        for dataset, info in chi2_registry.describe().items():
            print(
                f"   {dataset:12s} target={info['target']:.6g} ± {info['tolerance']:.6g}"
            )

    updated_keys: Set[str] = set()
    if fiducial:
        updated_keys = _update_parameter_defaults(model, fiducial, dry_run=args.dry_run)
        if not args.dry_run:
            if updated_keys:
                print("\n📝 Updated cosmos/optim/parameter_defaults.py:")
                for key in sorted(updated_keys):
                    print(f"   {key:6s} = {_format_parameter_value(fiducial[key])}")
                for key in updated_keys:
                    DEFAULT_REFERENCES[model][key] = float(fiducial[key])
            else:
                print("\nℹ️ Parameter defaults already reflect these values.")

    island_payload = result.get("island_center")
    if island_payload:
        center_params = island_payload.get("center_params") or {}
        print("\n🏝️ Island center (interior of viable basin):")
        print(
            f"   χ² = {island_payload['center_chi2']:.3f} "
            f"(threshold ≤ {island_payload['chi2_threshold']:.3f})"
        )
        print(
            f"   Core density: {island_payload['num_core']} / {island_payload['num_viable']} "
            f"viable (samples={island_payload['num_samples']})"
        )
        for key in center_params:
            if key in fiducial:
                print(f"   {key:6s} = {center_params[key]:.6g}")

    axis_scans = result.get("axis_scans") or []
    if axis_scans:
        print("\n📊 Basin edges:")
        for entry in axis_scans:
            label = f"{entry.get('param')} (pass {entry.get('pass')})"
            best = entry.get("best")
            left = entry.get("left_edge")
            right = entry.get("right_edge")
            curve = entry.get("curve") or []
            valid_points = sum(1 for point in curve if point.get("valid"))
            total_points = len(curve)
            phase6a_points = sum(1 for point in curve if point.get("passes_phase6a"))
            summary_tail = f"[valid {valid_points}/{total_points}, phase6a {phase6a_points}/{total_points}]"
            if best is None:
                print(f"   {label:18s} no valid points {summary_tail}")
            elif left is None or right is None:
                print(f"   {label:18s} best={best:.6g} | edges not established {summary_tail}")
            else:
                print(f"   {label:18s} best={best:.6g} | edges [{left:.6g}, {right:.6g}] {summary_tail}")

    if trace_path and trace_path.exists():
        print(f"\n🧾 Basin trace: {trace_path.resolve()}")
    if generated_plots:
        print("\n🖼️ Basin plots:")
        for name, path in generated_plots.items():
            print(f"   {name}: {path.resolve()}")
    elif basin_plot_error:
        print(f"\n⚠️  Basin plot generation skipped: {basin_plot_error}")


def main():
    """Main CLI entry point."""
    # Print version header
    print("Cosmos Engine Version 1.0")
    print("By Fabian Olesen\n")
    
    # Set up the main parser
    parser = argparse.ArgumentParser(
        description="PBUF4 Cosmology Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
  python cli.py dataset download --name planck2018_distance_priors
{{ ... }}

  # Run fits with different syntax options:
  python cli.py run sn_pantheon                          # Both LCDM and PBUF (default)
  python cli.py run sn_pantheon --model pbuf             # Only PBUF model
  python cli.py run sn_pantheon --model both --parameters '{"H0":70.0,"Om0":0.31}'
  python cli.py run cc --model pbuf --parameters '{"H0":68.0,"alpha":1e-3}'

  # Legacy syntax (still supported):
  python cli.py run sn_pantheon --pbuf '{"alpha":5e-4,"Rmax":1e9,"k_sat":0.8}'
  python cli.py run sn_sh0es --lcdm '{"H0":69.1,"Om0":0.32}'

  python cli.py fit joint --model lcdm --datasets cmb,sn_pantheon,bao_iso
  python cli.py fit run --model pbuf
  python cli.py fit coord --output data/results/basin_scan.json
  python cli.py fit grid --model pbuf
  python cli.py fit joint-comprehensive --model pbuf --datasets cmb,sn_pantheon,sn_sh0es,bao_iso,bao_aniso,cc,rsd
  python cli.py test all
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    # dataset download
    download_parser = dataset_subparsers.add_parser("download", help="Download raw datasets")
    download_parser.add_argument("--name", required=True,
                                choices=["planck2018_distance_priors", "pantheon_sn",
                                        "bao_boss_dr12", "bao_eBOSS", "cc_cosmic_chronometers_compilation",
                                        "rsd_fsigma8_compilation"],
                                help="Dataset name to download")
    download_parser.set_defaults(func=dataset_download)

    # dataset convert
    convert_parser = dataset_subparsers.add_parser("convert", help="Convert raw to standardized format")
    convert_parser.add_argument("--source", required=True, help="Source dataset name (directory in data/raw/)")
    convert_parser.add_argument("--output", required=True, help="Output .npz file path")
    convert_parser.add_argument("--type",
                               choices=["sn_pantheon", "sn_sh0es", "sn", "bao_iso", "bao_aniso", "bao", "cc", "rsd", "cmb", "sh0es", "auto"],
                               help="Explicit dataset type (auto-detect if not specified)")
    convert_parser.set_defaults(func=dataset_convert)

    # Run commands
    run_parser = subparsers.add_parser("run", help="Run individual fits")
    run_parser.add_argument(
        "fit",
        choices=[
            "cmb",
            "sn",
            "sn_pantheon",
            "sn_sh0es",
            "bao_iso",
            "bao_aniso",
            "cc",
            "rsd",
            "science",
        ],
        help="Dataset to fit (sn_pantheon: Pantheon+, sn_sh0es: SH0ES). Use 'science' for the unified science runner.",
    )
    run_parser.add_argument("--model",
                           choices=["lcdm", "pbuf", "both"],
                           help="Model(s) to fit: 'lcdm', 'pbuf', or 'both' (default: both)")
    run_parser.add_argument("--parameters", help="Parameter overrides as JSON (applies to specified model)")
    run_parser.add_argument("--lcdm", help="LCDM parameter overrides as JSON (legacy)")
    run_parser.add_argument("--pbuf", help="PBUF parameter overrides as JSON (legacy)")
    run_parser.add_argument("--config", help="Path to science run configuration (for 'science' fit)")
    run_parser.add_argument("--science-root", help="Override science run output root (for 'science' fit)")
    run_parser.add_argument("--resume-dir", help="Resume science run from existing directory")
    run_parser.add_argument("--fresh", action="store_true", help="Start a fresh science run, ignoring checkpoints")
    run_parser.set_defaults(func=run_fit)

    # Fit commands
    fit_parser = subparsers.add_parser("fit", help="Parameter optimization")
    fit_subparsers = fit_parser.add_subparsers(dest="fit_command")

    # fit joint (legacy joint fit)
    fit_joint_parser = fit_subparsers.add_parser(
        "joint",
        help="Run the joint capture pipeline and persist diagnostics.",
    )
    fit_joint_parser.add_argument(
        "--model",
        choices=["lcdm", "pbuf", "both"],
        default="both",
        help="Models to summarise (default: both).",
    )
    fit_joint_parser.add_argument(
        "--datasets",
        help="Comma-separated dataset list or 'all' (default: full bundle).",
    )
    fit_joint_parser.add_argument(
        "--output",
        help="Optional output path for the JSON capture (default: data/results/joint_capture_<timestamp>.json).",
    )
    fit_joint_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress solver progress logging.",
    )
    fit_joint_parser.set_defaults(func=run_joint)

    # fit joint comprehensive
    joint_comprehensive_parser = fit_subparsers.add_parser(
        "joint-comprehensive",
        help="Run joint capture with optimizer traces and physics validation summary.",
    )
    joint_comprehensive_parser.add_argument(
        "--model",
        default="pbuf",
        choices=["lcdm", "pbuf", "both"],
        help="Models to highlight (default: pbuf).",
    )
    joint_comprehensive_parser.add_argument(
        "--datasets",
        help="Comma-separated datasets (e.g., 'cmb,pantheon,bao_iso') or 'all'.",
    )
    joint_comprehensive_parser.add_argument(
        "--output",
        help="Optional output path for the JSON capture (default: data/results/joint_capture_<timestamp>.json).",
    )
    joint_comprehensive_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress solver progress logging.",
    )
    joint_comprehensive_parser.set_defaults(func=fit_joint_comprehensive)

    # fit run model
    run_model_parser = fit_subparsers.add_parser("run", help="Run fits for a specific model across all available datasets")
    run_model_parser.add_argument("--model", required=True,
                                 choices=["lcdm", "pbuf"],
                                 help="Model to fit (required)")
    run_model_parser.add_argument("--lcdm", help="LCDM parameter overrides as JSON")
    run_model_parser.add_argument("--pbuf", help="PBUF parameter overrides as JSON")
    run_model_parser.set_defaults(func=fit_run_model)

    # fit deterministic grid pipeline
    grid_parser = fit_subparsers.add_parser(
        "grid",
        help="Run deterministic grid-based scoring across datasets",
    )
    grid_parser.add_argument(
        "--model",
        choices=["lcdm", "pbuf", "both"],
        default="both",
        help="Model(s) to evaluate (default: both)",
    )
    grid_parser.add_argument(
        "--datasets",
        help="Comma-separated dataset list or 'all' (default: base set including sn_pantheon, sn_sh0es)",
    )
    grid_parser.add_argument(
        "--include-bao",
        action="store_true",
        help="Append BAO datasets to the default list",
    )
    grid_parser.add_argument(
        "--grid-config",
        help="Path to JSON file defining the parameter grid",
    )
    grid_parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Directory for JSON score tables (default: data/results)",
    )
    grid_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker count (default: 1 => serial)",
    )
    grid_parser.add_argument(
        "--tag",
        help="Optional run tag included in metadata/output filename",
    )
    grid_parser.add_argument(
        "--refine-top",
        type=int,
        default=0,
        help="Number of top-ranked cosmologies to locally refine (0 disables)",
    )
    grid_parser.add_argument(
        "--refine-fraction",
        type=float,
        default=0.05,
        help="Fractional +/- range for each parameter during refinement (default 5%)",
    )
    grid_parser.add_argument(
        "--refine-points",
        type=int,
        default=3,
        help="Number of samples per axis in each local refinement grid (default 3)",
    )
    grid_parser.set_defaults(func=fit_grid_pipeline)

    coord_parser = fit_subparsers.add_parser(
        "coord",
        help="Coordinate descent basin walker for LCDM or PBUF.",
    )
    coord_parser.add_argument(
        "--model",
        default="pbuf",
        choices=["pbuf", "lcdm"],
        help="Model to optimize (default: pbuf).",
    )
    coord_parser.add_argument(
        "--datasets",
        help="Comma-separated dataset list or 'all' (default: cmb,sn_pantheon,bao_iso).",
    )
    coord_parser.add_argument(
        "--include-bao",
        action="store_true",
        help="Append BAO datasets (iso, aniso) to the bundle.",
    )
    coord_parser.add_argument(
        "--phase6a",
        action="store_true",
        help="Require Phase 6a validation during scans.",
    )
    coord_parser.add_argument(
        "--delta-chi2",
        type=float,
        default=20.0,
        help="Δχ² tolerance used to define basin edges (default: 20).",
    )
    coord_parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for optimizer results.",
    )
    coord_parser.add_argument(
        "--seed-json",
        help="Optional JSON file providing a starting parameter set.",
    )
    coord_parser.add_argument(
        "--eps0",
        type=float,
        help="Override elastic stiffness baseline (default: 0.7).",
    )
    coord_parser.add_argument(
        "--skip-second-pass",
        action="store_true",
        help="Disable the tightening pass on (H0, Om0).",
    )
    coord_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging from the basin walker.",
    )
    coord_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during scanning.",
    )
    coord_parser.add_argument(
        "--converge",
        "--auto-converge",
        dest="converge",
        action="store_true",
        help="Iterate run_with_convergence() so scans continue until Δχ² drops below the improvement tolerance.",
    )
    coord_parser.add_argument(
        "--max-cycles",
        type=int,
        default=6,
        help="Maximum convergence cycles (default: 6).",
    )
    coord_parser.add_argument(
        "--workers",
        type=int,
        help="Parallel worker count for axis scans (default: auto).",
    )
    coord_parser.add_argument(
        "--improvement-tol",
        type=float,
        default=1.0e-2,
        help="χ² improvement threshold for convergence (default: 1e-2).",
    )
    coord_parser.add_argument(
        "--island-samples",
        type=int,
        default=0,
        help="Number of random samples to locate the island center (0 disables).",
    )
    coord_parser.add_argument(
        "--island-delta",
        type=float,
        default=20.0,
        help="Δχ² threshold for island core membership (default: 20).",
    )
    coord_parser.add_argument(
        "--island-seed",
        type=int,
        help="Optional RNG seed for the island sampling stage.",
    )
    coord_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not persist the updated fiducial parameters into parameter_defaults.py.",
    )
    coord_parser.add_argument(
        "--chi2-targets",
        help="Optional path to a χ² target configuration file (default: config/chi2_targets.json).",
    )
    coord_parser.add_argument(
        "--basin-record-dir",
        help="Directory to persist per-scan basin traces for plotting (JSON + CSV).",
    )
    coord_parser.add_argument(
        "--basin-plot-dir",
        help="Directory to write basin diagnostic plots (requires recording; defaults to record dir).",
    )
    coord_parser.set_defaults(func=fit_coordinate_optimizer)

    # Report commands
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_subparsers = report_parser.add_subparsers(dest="report_command")

    # report generate
    generate_parser = report_subparsers.add_parser("generate", help="Generate scientific reports")
    generate_parser.add_argument("--formats", help="Comma-separated output formats (default: html,md,pdf,json)")
    generate_parser.add_argument("--models", help="Comma-separated model list (default: lcdm,pbuf)")
    generate_parser.add_argument("--output", help="Destination directory for generated reports")
    generate_parser.add_argument("--science-root", help="Path to science run root directory")
    generate_parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    generate_parser.set_defaults(func=report_generate)

    # Test commands
    test_parser = subparsers.add_parser("test", help="Run validation tests")
    test_subparsers = test_parser.add_subparsers(dest="test_command")

    # test all
    test_all_parser = test_subparsers.add_parser("all", help="Run all tests")
    test_all_parser.set_defaults(func=test_all)

    # Parse arguments
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    # Run the selected command
    try:
        args.func(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
