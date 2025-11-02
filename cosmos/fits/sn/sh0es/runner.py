"""
SH0ES SN Fit Runner - Execute SH0ES supernovae calibration fits.

This module runs cosmological fits against SH0ES supernovae calibration data
for both LCDM and PBUF models. It uses the SH0ES H0 measurement as a Gaussian
prior constraint, enabling absolute scale calibration for cosmological models.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.fits.sn.sh0es.chi2 import chi2_sn_sh0es
from cosmos.fits.sn.sh0es.observables import compute_sh0es_mu_model, extract_model_h0
from cosmos.optim.parameter_defaults import (
    LCDM_PARAMETER_DEFAULTS,
    PBUF_PARAMETER_DEFAULTS,
)


def get_default_params():
    """Get default parameters for LCDM and PBUF models."""
    lcdm_defaults = dict(LCDM_PARAMETER_DEFAULTS)
    lcdm_defaults.update({
        "Ol0": 0.6847,
        "Obh2": 0.02237,
    })
    pbuf_defaults = dict(PBUF_PARAMETER_DEFAULTS)
    pbuf_defaults.update({
        "Ol0": 0.0,
        "Obh2": 0.02237,
    })
    return {
        "lcdm": lcdm_defaults,
        "pbuf": pbuf_defaults,
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


def run_fit(model_overrides: Optional[Dict[str, Any]] = None, verbose: bool = True):
    """
    Run SH0ES SN fit for both LCDM and PBUF models.

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
        print("🏃 Running SH0ES SN fit...")
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

    # SH0ES constraint details
    if verbose:
        print("🎯 SH0ES Constraint:")
        print(f"   H0_obs = 73.04 ± 1.04 km/s/Mpc")
        print(f"   Mode: Gaussian prior on H0")
        print()

    # Create models and compute chi2
    results = {}

    for model_type in ["lcdm", "pbuf"]:
        if verbose:
            print(f"🔬 Computing {model_type.upper()} χ²...")

        try:
            # Create model function for chi2 calculation
            def model_func(p):
                return create_model(p, model_type)

            # Compute chi2 using new sh0es module
            result = chi2_sn_sh0es(model_func, params[model_type])

            # Extract model H0 for comparison
            model = create_model(params[model_type], model_type)
            model_h0 = extract_model_h0(model, params[model_type])

            # Store results
            results[model_type] = {
                "chi2": result["chi2"],
                "n_data": result["n_data"],
                "n_params": len(params[model_type]),
                "parameters": params[model_type],
                "model_type": model_type,
                "dataset": result["dataset"],
                "status": result["status"],
                "model_h0": model_h0,
                "constraint_h0": 73.04,
                "constraint_error": 1.04
            }

            if verbose:
                print(f"   χ² = {result['chi2']:.6f}")
                print(f"   Status: {result['status']}")
                print(f"   Data points: {result['n_data']}")
                print(f"   Model H0: {model_h0:.3f} km/s/Mpc")
                print(f"   Constraint H0: 73.04 ± 1.04 km/s/Mpc")
                print(f"   Parameters: {params[model_type]}")
                print()

        except Exception as e:
            print(f"❌ Error computing {model_type} χ²: {e}")
            results[model_type] = {
                "chi2": np.nan,
                "error": str(e),
                "parameters": params[model_type],
                "model_type": model_type,
                "status": "error"
            }

    # Compute comparison statistics
    if "lcdm" in results and "pbuf" in results:
        chi2_lcdm = results["lcdm"]["chi2"]
        chi2_pbuf = results["pbuf"]["chi2"]

        if np.isfinite(chi2_lcdm) and np.isfinite(chi2_pbuf):
            delta_chi2 = chi2_pbuf - chi2_lcdm

            # H0 tension analysis
            h0_lcdm = results["lcdm"]["model_h0"]
            h0_pbuf = results["pbuf"]["model_h0"]
            h0_constraint = 73.04
            h0_error = 1.04

            tension_lcdm = abs(h0_lcdm - h0_constraint) / h0_error
            tension_pbuf = abs(h0_pbuf - h0_constraint) / h0_error

            # Summary
            if verbose:
                print("📈 Model Comparison Summary:")
                print(f"   LCDM: χ²={chi2_lcdm:.6f} H0={h0_lcdm:.3f} tension={tension_lcdm:.2f}σ")
                print(f"   PBUF: χ²={chi2_pbuf:.6f} H0={h0_pbuf:.3f} tension={tension_pbuf:.2f}σ")
                print(f"   Δχ² (PBUF-LCDM) = {delta_chi2:.6f}")

                if tension_lcdm < tension_pbuf:
                    print(f"   ✅ LCDM has lower H0 tension with SH0ES")
                else:
                    print(f"   ✅ PBUF has lower H0 tension with SH0ES")

                print()

    # Save results
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)

    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_type, result in results.items():
        if "error" not in result:
            output_file = results_dir / f"sn_sh0es_{model_type}.json"
            result["model"] = model_type  # Added for compatibility with summary_builder
            result["dataset"] = "sn_sh0es"  # Updated for new naming
            result["timestamp"] = timestamp

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

    if verbose:
        print("💾 Saved individual fit results to data/results/")

    return results


if __name__ == "__main__":
    # Test run
    print("🧪 Testing SH0ES SN fit runner...")
    results = run_fit(verbose=True)
    print("✅ Test completed successfully!")
