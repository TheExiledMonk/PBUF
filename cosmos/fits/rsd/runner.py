"""
RSD Fit Runner - Execute redshift space distortion fσ8(z) fits.

This module runs cosmological fits against redshift space distortion measurements
for both LCDM and PBUF models. It loads standardized data, computes χ² values,
and reports the comparison between models.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.fits.rsd.chi2 import chi_squared_rsd
from cosmos.optim.parameter_defaults import (
    LCDM_PARAMETER_DEFAULTS,
    PBUF_PARAMETER_DEFAULTS,
    SIGMA8_PLANCK,
)
from cosmos.fits._dataset_loader import load_rsd_dataset


def get_default_params():
    """Get default parameters for LCDM and PBUF models."""
    lcdm_defaults = dict(LCDM_PARAMETER_DEFAULTS)
    pbuf_defaults = dict(PBUF_PARAMETER_DEFAULTS)

    def _inferred_ol0(defaults: Dict[str, Any]) -> float:
        return max(
            0.0,
            1.0
            - float(defaults.get("Om0", 0.0))
            - float(defaults.get("Ok0", 0.0))
            - float(defaults.get("Or0", 0.0)),
        )

    return {
        "lcdm": {
            "H0": float(lcdm_defaults.get("H0", 67.36)),
            "Om0": float(lcdm_defaults.get("Om0", 0.3153)),
            "Ok0": float(lcdm_defaults.get("Ok0", 0.0)),
            "Ol0": float(lcdm_defaults.get("Ol0", _inferred_ol0(lcdm_defaults))),
            "Or0": float(lcdm_defaults.get("Or0", 9.2e-5)),
            "Obh2": 0.02237,
        },
        "pbuf": {
            "H0": float(pbuf_defaults.get("H0", 67.36)),
            "Om0": float(pbuf_defaults.get("Om0", 0.3153)),
            "Ok0": float(pbuf_defaults.get("Ok0", 0.0)),
            "Ol0": 0.0,  # No cosmological constant
            "Or0": float(pbuf_defaults.get("Or0", 9.2e-5)),
            "Obh2": 0.02237,
            "alpha": float(pbuf_defaults.get("alpha", 1.0e-3)),
            "Rmax": float(pbuf_defaults.get("Rmax", 1.0e8)),
            "k_sat": float(pbuf_defaults.get("k_sat", 1.0)),
            "eps0": float(pbuf_defaults.get("eps0", 0.7)),
            "n_alpha": float(pbuf_defaults.get("n_alpha", 0.0)),
            "n_eps": float(pbuf_defaults.get("n_eps", 0.0)),
            "n_R": float(pbuf_defaults.get("n_R", 0.0)),
        }
    }


def create_model(params: dict, model_type: str):
    """Create model instance from parameter dictionary."""
    if model_type == "lcdm":
        return LCDM(
            omega_m=params["Om0"],
            omega_lambda=params["Ol0"],
            h=params["H0"]/100.0,
            omega_k=params["Ok0"],
            omega_r=params["Or0"],
            omega_b=params["Obh2"]/(params["H0"]/100.0)**2,
            T_cmb=2.7255
        )
    elif model_type == "pbuf":
        return PBUF(
            omega_m=params["Om0"],
            h=params["H0"]/100.0,
            alpha=params["alpha"],
            Rmax=params["Rmax"],
            k_sat=params["k_sat"],
            eps0=params.get("eps0", 0.7),
            n_alpha=params.get("n_alpha", 0.0),
            n_eps=params.get("n_eps", 0.0),
            n_R=params.get("n_R", 0.0),
            omega_k=params["Ok0"],
            omega_r=params["Or0"],
            omega_b=params["Obh2"]/(params["H0"]/100.0)**2,
            T_cmb=2.7255
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_rsd_data():
    """Load standardized RSD data."""
    try:
        return load_rsd_dataset()
    except FileNotFoundError as exc:
        print(f"⚠️ Could not load RSD data: {exc}")
        return None


def run_fit(model_overrides: Optional[Dict[str, Any]] = None, verbose: bool = True):
    """
    Run RSD fit for both LCDM and PBUF models.

    Parameters
    ----------
    model_overrides : dict or None
        Parameter overrides like {"lcdm": {"H0": 69.1}, "pbuf": {"alpha": 5e-4}}
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        Fit results for both models
    """
    if verbose:
        print("🏃 Running RSD fit...")
        print("=" * 40)

    # Get default parameters
    defaults = get_default_params()

    # Apply overrides
    params = {}
    for model_type in ["lcdm", "pbuf"]:
        params[model_type] = defaults[model_type].copy()
        if model_overrides and model_type in model_overrides:
            params[model_type].update(model_overrides[model_type])

    if verbose:
        if model_overrides:
            print("📋 Applied parameter overrides:")
            for model_type, overrides in model_overrides.items():
                print(f"   {model_type}: {overrides}")
        print()

    # Load RSD data
    data = load_rsd_data()
    if data is None:
        raise FileNotFoundError("Could not load RSD data. Run 'python cli.py dataset convert --source rsd_fsigma8_compilation --output data/standardized/rsd.npz' first.")

    if verbose:
        print("📊 RSD Dataset:")
        print(f"   Name: {data.get('name', 'RSD Compilation')}")
        print(f"   Points: {len(data.get('obs', []))}")
        print(f"   z-range: {np.min(data['z']):.3f} {np.max(data['z']):.3f}")
        print()

    # Create models and compute chi2
    results = {}

    for model_type in ["lcdm", "pbuf"]:
        if verbose:
            print(f"🔬 Computing {model_type.upper()} χ²...")

        try:
            # Create model
            model = create_model(params[model_type], model_type)

            sigma8_override = params[model_type].get("sigma8_0")
            if sigma8_override is None and model_type == "lcdm":
                sigma8_override = SIGMA8_PLANCK

            if sigma8_override is not None:
                chi2 = chi_squared_rsd(model, data=data, sigma8_0=sigma8_override)
            else:
                chi2 = chi_squared_rsd(model, data=data)

            # Store results
            results[model_type] = {
                "chi2": chi2,
                "n_data": len(data["obs"]),
                "n_params": len(params[model_type]),
                "parameters": params[model_type],
                "model_type": model_type
            }

            if verbose:
                print(f"   χ² = {chi2:.6f}")
                print(f"   Parameters: {params[model_type]}")
                print()

        except Exception as e:
            print(f"❌ Error computing {model_type} χ²: {e}")
            results[model_type] = {
                "chi2": np.nan,
                "error": str(e),
                "parameters": params[model_type],
                "model_type": model_type
            }

    # Compute comparison statistics
    if "lcdm" in results and "pbuf" in results:
        chi2_lcdm = results["lcdm"]["chi2"]
        chi2_pbuf = results["pbuf"]["chi2"]

        if np.isfinite(chi2_lcdm) and np.isfinite(chi2_pbuf):
            delta_chi2 = chi2_pbuf - chi2_lcdm

            # Reduced chi2
            n_data = len(data["obs"])
            dof_lcdm = max(n_data - results["lcdm"]["n_params"], 1)
            dof_pbuf = max(n_data - results["pbuf"]["n_params"], 1)

            chi2_red_lcdm = chi2_lcdm / dof_lcdm
            chi2_red_pbuf = chi2_pbuf / dof_pbuf

            # AIC and BIC
            aic_lcdm = chi2_lcdm + 2 * results["lcdm"]["n_params"]
            aic_pbuf = chi2_pbuf + 2 * results["pbuf"]["n_params"]

            bic_lcdm = chi2_lcdm + results["lcdm"]["n_params"] * np.log(n_data)
            bic_pbuf = chi2_pbuf + results["pbuf"]["n_params"] * np.log(n_data)

            delta_aic = aic_pbuf - aic_lcdm
            delta_bic = bic_pbuf - bic_lcdm

            # Summary
            if verbose:
                print("📈 Model Comparison Summary:")
                print(f"   LCDM: χ²={chi2_lcdm:.6f} χ²_red={chi2_red_lcdm:.6f} AIC={aic_lcdm:.3f} BIC={bic_lcdm:.3f}")
                print(f"   PBUF: χ²={chi2_pbuf:.6f} χ²_red={chi2_red_pbuf:.6f} AIC={aic_pbuf:.3f} BIC={bic_pbuf:.3f}")
                print(f"   Δχ² (PBUF-LCDM) = {delta_chi2:.6f}")
                print(f"   ΔAIC (PBUF-LCDM) = {delta_aic:.3f}")
                print(f"   ΔBIC (PBUF-LCDM) = {delta_bic:.3f}")

                if delta_chi2 < 0:
                    print(f"   ✅ PBUF fits RSD better than LCDM")
                else:
                    print(f"   ℹ️ LCDM fits RSD better (Δχ² = {delta_chi2:.3f}")

                print()

    # Save results
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)

    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_type, result in results.items():
        if "error" not in result:
            output_file = results_dir / f"rsd_{model_type}.json"
            result["dataset"] = "rsd"
            result["timestamp"] = timestamp
            result["data_file"] = str(data.get("name", "unknown"))

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

    if verbose:
        print("💾 Saved individual fit results to data/results/")

    return results


if __name__ == "__main__":
    # Test run
    print("🧪 Testing RSD fit runner...")
    results = run_fit(verbose=True)
    print("✅ Test completed successfully!")
