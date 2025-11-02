"""
Joint Fit Runner - Run joint cosmological fits across all datasets.

This module orchestrates joint cosmological fitting across all available
datasets (CMB, SN, BAO isotropic, BAO anisotropic, CC, RSD) for both
LCDM and PBUF models. It computes joint χ², AIC, BIC, and model comparison
statistics.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Import individual fit runners
from cosmos.fits.cmb.runner import run_fit as run_cmb_fit
from cosmos.fits.sn.runner import run_fit as run_sn_fit
from cosmos.fits.sn.pantheon.runner import run_fit as run_sn_pantheon_fit
from cosmos.fits.sn.sh0es.runner import run_fit as run_sn_sh0es_fit
from cosmos.fits.bao.runner import run_fit as run_bao_iso_fit
from cosmos.fits.bao.aniso_runner import run_fit as run_bao_aniso_fit
from cosmos.fits.cc.runner import run_fit as run_cc_fit
from cosmos.fits.rsd.runner import run_fit as run_rsd_fit

# Import summary builder for joint statistics
from reports.summary_builder import compute_model_stats


def get_available_datasets():
    """Get list of available datasets from both standardized data and module availability."""
    datasets = []

    # Check standardized files
    std_dir = Path("data/standardized")
    if std_dir.exists():
        for npz_file in std_dir.glob("*.npz"):
            name = npz_file.stem
            if name.startswith("cmb"):
                datasets.append("cmb")
            elif name.startswith("sn") or name.startswith("pantheon"):
                datasets.append("sn")
            elif name.startswith("sh0es"):
                datasets.append("sn")  # Group with SN for now
            elif "bao" in name.lower():
                if "aniso" in name.lower() or "anisotropic" in name.lower():
                    datasets.append("bao_aniso")
                else:
                    datasets.append("bao_iso")
            elif name.startswith("cc") or "chronometer" in name.lower():
                datasets.append("cc")
            elif name.startswith("rsd") or "fsigma8" in name.lower():
                datasets.append("rsd")

    # Check for new module availability
    try:
        # Try to import new modules to see if they're available
        import importlib
        if importlib.util.find_spec("cosmos.fits.sn.pantheon"):
            datasets.append("sn_pantheon")
        if importlib.util.find_spec("cosmos.fits.sn.sh0es"):
            datasets.append("sn_sh0es")
    except:
        pass

    return list(set(datasets))  # Remove duplicates


def run_individual_fits(model_overrides: Optional[Dict[str, Any]] = None, verbose: bool = True):
    """
    Run individual fits for all available datasets.

    Parameters
    ----------
    model_overrides : dict or None
        Parameter overrides for models
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        Nested results structure: results[dataset][model] = fit_info
    """
    if verbose:
        print("🔬 Running individual fits for all datasets...")
        print("=" * 50)

    available_datasets = get_available_datasets()
    if not available_datasets:
        print("⚠️ No standardized datasets found in data/standardized/")
        print("   Run 'python cli.py dataset convert' commands first")
        return {}

    if verbose:
        print(f"📊 Available datasets: {available_datasets}")
        print()

    results = {}

    # Run fits for each dataset
    for dataset in available_datasets:
        if verbose:
            print(f"🏃 Running {dataset} fits...")

        try:
            if dataset == "cmb":
                dataset_results = run_cmb_fit(model_overrides, verbose=False)
            elif dataset == "sn":
                dataset_results = run_sn_fit(model_overrides, verbose=False)
            elif dataset == "sn_pantheon":
                dataset_results = run_sn_pantheon_fit(model_overrides, verbose=False)
            elif dataset == "sn_sh0es":
                dataset_results = run_sn_sh0es_fit(model_overrides, verbose=False)
            elif dataset == "bao_iso":
                dataset_results = run_bao_iso_fit(model_overrides, verbose=False)
            elif dataset == "bao_aniso":
                dataset_results = run_bao_aniso_fit(model_overrides, verbose=False)
            elif dataset == "cc":
                dataset_results = run_cc_fit(model_overrides, verbose=False)
            elif dataset == "rsd":
                dataset_results = run_rsd_fit(model_overrides, verbose=False)
            else:
                print(f"⚠️ Unknown dataset type: {dataset}")
                continue

            results[dataset] = dataset_results

            if verbose:
                print(f"   ✅ {dataset} completed")

        except Exception as e:
            print(f"❌ Error running {dataset} fit: {e}")
            results[dataset] = {"error": str(e)}

    if verbose:
        print()

    return results


def run_joint_fit(overrides: Optional[Dict[str, Any]] = None, verbose: bool = True):
    """
    Run joint fit across all datasets and compute comparison statistics.

    If datasets are specified in overrides, performs a true joint fit.
    Otherwise, runs individual fits and computes aggregated statistics.

    Parameters
    ----------
    overrides : dict or None
Contains model_type, datasets, and model parameter overrides
    verbose : bool
    Print detailed output

    Returns
    -------
    dict
    Complete statistics including per-dataset and joint results
"""
    # Extract parameters
    model_type = overrides.get("model_type", "pbuf") if overrides else "pbuf"
    datasets = overrides.get("datasets") if overrides else None
    model_overrides = {k: v for k, v in (overrides or {}).items() if k in ["lcdm", "pbuf"]}

    if datasets is not None:
        # True joint fit across specified datasets
        if verbose:
            print(f"🔬 Performing true joint fit for {model_type.upper()} across {len(datasets)} datasets")
        from cosmos.fits.joint.optimizer import fit_joint
        initial_params = model_overrides.get(model_type)
        result = fit_joint(
            model_type=model_type,
            datasets=datasets,
            initial_params=initial_params,
            verbose=verbose,
        )
        return result

    # Original behavior: individual fits
    if verbose:
        print(f"🔬 Running individual fits for {model_type.upper()} across all datasets")
    individual_results = run_individual_fits(model_overrides, verbose)

    if not individual_results:
        print("❌ No datasets available for joint fitting")
        return {}

    # Save individual results
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset, model_results in individual_results.items():
        for model_type, result in model_results.items():
            if "error" not in result:
                output_file = results_dir / f"{dataset}_{model_type}.json"
                result["dataset"] = dataset
                result["timestamp"] = timestamp

                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)

    if verbose:
        print("💾 Saved individual fit results to data/results/")
        print()

    # Compute joint statistics using summary_builder logic
    if verbose:
        print("📊 Computing joint statistics...")
        print("=" * 40)

    # Convert to format expected by summary_builder
    sb_results = {}
    for dataset, model_results in individual_results.items():
        sb_results[dataset] = {}
        for model_type, result in model_results.items():
            if "error" not in result and np.isfinite(result.get("chi2", np.nan)):
                sb_results[dataset][model_type] = {
                    "chi2": result["chi2"],
                    "n_data": result["n_data"],
                    "n_params": result["n_params"],
                    "parameters": result.get("parameters", {}),
                    "metadata": result.get("metadata", {})
                }

    # Compute model statistics
    models = ["lcdm", "pbuf"]
    stats = compute_model_stats(sb_results, models)

    if verbose:
        print("📈 Joint Fit Results:")
        print(f"   Datasets included: {list(sb_results.keys())}")

        # Per-dataset summary
        print("\n   Per-dataset χ²:")
        for dataset in sb_results.keys():
            if "lcdm" in sb_results[dataset] and "pbuf" in sb_results[dataset]:
                chi2_lcdm = sb_results[dataset]["lcdm"]["chi2"]
                chi2_pbuf = sb_results[dataset]["pbuf"]["chi2"]
                print(f"     {dataset:15} chi2_LCDM={chi2_lcdm:8.4f} chi2_PBUF={chi2_pbuf:8.4f} delta_chi2={chi2_pbuf-chi2_lcdm:8.4f}")

        # Model totals
        print("\n   Model totals:")
        for model in models:
            if model in stats["models"]:
                total = stats["models"][model]
                print(f"     {model.upper():15} chi2_total={total['chi2_total']:8.4f} chi2_red={total['chi2_reduced_total']:8.4f} n_data={total['n_data_total']:3d}")

        # Global comparison
        if "global" in stats and "comparison" in stats["global"]:
            comp = stats["global"]["comparison"]
            delta_aic = comp.get("ΔAIC (PBUF-LCDM)", 0)
            delta_bic = comp.get("ΔBIC (PBUF-LCDM)", 0)
            print("\n   Model comparison:")
            print(f"     ΔAIC (PBUF-LCDM) = {delta_aic:8.4f} ({'PBUF better' if delta_aic < 0 else 'LCDM better'})")
            print(f"     ΔBIC (PBUF-LCDM) = {delta_bic:8.4f} ({'PBUF better' if delta_bic < 0 else 'LCDM better'})")

        print()

    # Save joint statistics
    joint_stats_file = results_dir / f"joint_fit_{timestamp}.json"
    stats["timestamp"] = timestamp
    stats["model_overrides"] = model_overrides or {}

    with open(joint_stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"💾 Saved joint statistics to {joint_stats_file}")

    return stats


if __name__ == "__main__":
    # Test joint fit
    print("🧪 Testing joint fit runner...")
    stats = run_joint_fit(verbose=True)
    print("✅ Joint fit test completed successfully!")
