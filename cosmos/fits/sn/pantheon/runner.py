"""
Pantheon SN Fit Runner - Execute Pantheon supernovae distance modulus fits.

This module runs cosmological fits against Pantheon supernovae distance modulus data
for both LCDM and PBUF models. It loads standardized data, computes χ² values,
and reports the comparison between models using the new modular SN implementation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Local imports
from .loader import load_pantheon_data
from .chi2 import chi2_sn_pantheon_abs, chi2_sn_pantheon
from .observables import compute_pantheon_mu_model

# Import model classes directly
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF

# Default parameter values
LCDM_PARAMETER_DEFAULTS = {
    'H0': 67.8,      # Hubble constant [km/s/Mpc]
    'Om0': 0.31,     # Matter density parameter
    'Or0': 9.2e-5,   # Radiation density parameter
    'Ok0': 0.0,      # Curvature density parameter
    'Ol0': 0.69,     # Dark energy density parameter
    'Obh2': 0.022,   # Baryon density parameter * h^2
}

PBUF_PARAMETER_DEFAULTS = {
    'H0': 67.8,        # Hubble constant [km/s/Mpc]
    'Om0': 0.31,       # Matter density parameter
    'Or0': 9.2e-5,     # Radiation density parameter
    'Ok0': 0.0,        # Curvature density parameter
    'Ol0': 0.0,        # Dark energy density parameter (not used in PBUF)
    'Obh2': 0.022,     # Baryon density parameter * h^2
    'alpha': 0.1,      # PBUF parameter
    'Rmax': 1e6,       # PBUF parameter
    'k_sat': 0.1,      # PBUF parameter
    'eps0': 0.1,       # PBUF parameter
    'n_alpha': 0.5,    # PBUF parameter
    'n_eps': -0.5,     # PBUF parameter
    'n_R': 0.0,        # PBUF parameter
}


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
    Run Pantheon SN fit for both LCDM and PBUF models.

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
        print("🏃 Running Pantheon SN fit...")
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

    # Create models and compute chi2
    results = {}

    for model_type in ["lcdm", "pbuf"]:

        try:
            # Create model function for chi2 calculation
            def model_func(p):
                return create_model(p, model_type)

            # Use absolute magnitude chi2 for Pantheon+SH0ES data
            result = chi2_sn_pantheon_abs(model_func, params[model_type])

            # Store results with debug info
            n_data = int(result.get("n_data", 1))  # Ensure n_data is an integer, default to 1

            
            results[model_type] = {
                "chi2": float(result["chi2"]),
                "n_data": n_data,
                "n_params": len(params[model_type]),
                "parameters": params[model_type],
                "model_type": model_type,
                "dataset": result.get("dataset", "SN_PANTHEON_ABS"),
                "status": str(result.get("status", "unknown"))
            }


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

            # Get n_data, ensuring it's at least 1 to avoid division by zero
            n_data = max(int(results["lcdm"].get("n_data", 1)), 1)

            
            # Calculate degrees of freedom, ensuring at least 1
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
                    print(f"   ✅ PBUF fits Pantheon SN better than LCDM")
                else:
                    print(f"   ℹ️ LCDM fits Pantheon SN better (Δχ² = {delta_chi2:.3f}")

                print()

    # Save results
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)

    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_type, result in results.items():
        if "error" not in result:
            output_file = results_dir / f"sn_pantheon_{model_type}.json"
            result["model"] = model_type  # Added for compatibility with summary_builder
            result["dataset"] = "sn_pantheon"  # Updated for new naming
            result["timestamp"] = timestamp

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

    if verbose:
        print("💾 Saved individual fit results to data/results/")

    return results


if __name__ == "__main__":
    # Test run
    print("🧪 Testing Pantheon SN fit runner...")
    results = run_fit(verbose=True)
    print("✅ Test completed successfully!")
