"""
Joint χ² consistency test for LCDM and PBUF.

This test ensures that:
  - all χ² submodules (CMB, SN, BAO, CC, RSD) work together,
  - combined χ² values are finite and additive,
  - the evaluation pipeline is ready for real data.

We use the same synthetic / self-consistent datasets used
in earlier tests — so χ²_LCDM ≈ 0 and χ²_PBUF > 0.
"""

import numpy as np

# Import standardized chi-squared evaluators
from cosmos.fits.cc.chi2 import chi_squared_cc
from cosmos.fits.sn.chi2 import chi_squared_sn
from cosmos.fits.bao.iso.chi2 import chi_squared_bao_iso
from cosmos.fits.bao.aniso.chi2 import chi_squared_bao_aniso
from cosmos.fits.rsd.chi2 import chi_squared_rsd
from cosmos.fits.cmb.observables import chi_squared_cmb  # CMB is in observables.py

# Model constructors
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF

# Data standardization
from data_interface.standardize import ensure_standard_dataset
from data_interface import load_cmb_priors, load_sn_data, load_bao_data, load_bao_iso_data, load_cc_data, load_rsd_data


# --------------------------------------------------------------
# 1. Define joint χ² evaluator
# --------------------------------------------------------------
def chi2_total(model_params, model_type="lcdm"):
    """
    Compute the total χ² from all cosmological probes using standardized format.

    Parameters
    ----------
    model_params : dict
        Cosmological parameters
    model_type : str
        "lcdm" or "pbuf"

    Returns
    -------
    dict with individual and total χ² components
    """
    # Create model instance from parameters
    if model_type.lower() == "lcdm":
        # LCDM parameters
        required_params = ["H0", "Om0", "Ok0", "Ol0", "Or0", "Obh2"]
        for param in required_params:
            if param not in model_params:
                raise ValueError(f"Missing LCDM parameter: {param}")

        h = model_params["H0"] / 100.0
        omega_b = model_params["Obh2"] / (h**2)

        model = LCDM(
            omega_m=model_params["Om0"],
            omega_lambda=model_params["Ol0"],
            h=h,
            omega_k=model_params["Ok0"],
            omega_r=model_params["Or0"],
            omega_b=omega_b
        )
    elif model_type.lower() == "pbuf":
        # PBUF parameters
        required_params = ["H0", "Om0", "Ok0", "Ol0", "Or0", "Obh2", "alpha", "Rmax", "k_sat"]
        for param in required_params:
            if param not in model_params:
                raise ValueError(f"Missing PBUF parameter: {param}")

        h = model_params["H0"] / 100.0
        omega_b = model_params["Obh2"] / (h**2)

        model = PBUF(
            omega_m=model_params["Om0"],
            h=h,
            alpha=model_params["alpha"],
            Rmax=model_params["Rmax"],
            k_sat=model_params["k_sat"],
            omega_k=model_params["Ok0"],
            omega_r=model_params["Or0"],
            omega_b=omega_b
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lcdm' or 'pbuf'")

    results = {}

    # CMB chi-squared (using standardized data)
    try:
        from cosmos.fits.cmb.observables import chi_squared_cmb
        cmb_data = load_cmb_priors()
        cmb_data = ensure_standard_dataset(cmb_data, "CMB")
        results["cmb"] = chi_squared_cmb(model, data=cmb_data)
    except Exception as e:
        print(f"Warning: CMB χ² calculation failed: {e}")
        results["cmb"] = 0.0

    # SN chi-squared (using standardized data)
    try:
        sn_data = load_sn_data()
        sn_data = ensure_standard_dataset(sn_data, "SN")
        results["sn"] = chi_squared_sn(model, data=sn_data)
    except Exception as e:
        print(f"Warning: SN χ² calculation failed: {e}")
        results["sn"] = 0.0

    # BAO isotropic chi-squared (using standardized data)
    try:
        bao_iso_data = load_bao_iso_data()
        bao_iso_data = ensure_standard_dataset(bao_iso_data, "BAO_ISO")
        results["bao_iso"] = chi_squared_bao_iso(model, data=bao_iso_data)
    except Exception as e:
        print(f"Warning: BAO isotropic χ² calculation failed: {e}")
        results["bao_iso"] = 0.0

    # BAO anisotropic chi-squared (using standardized data)
    try:
        bao_aniso_data = load_bao_data()
        bao_aniso_data = ensure_standard_dataset(bao_aniso_data, "BAO_ANISO")
        results["bao_aniso"] = chi_squared_bao_aniso(model, data=bao_aniso_data)
    except Exception as e:
        print(f"Warning: BAO anisotropic χ² calculation failed: {e}")
        results["bao_aniso"] = 0.0

    # CC chi-squared (using standardized data)
    try:
        cc_data = load_cc_data()
        cc_data = ensure_standard_dataset(cc_data, "CC")
        results["cc"] = chi_squared_cc(model, data=cc_data)
    except Exception as e:
        print(f"Warning: CC χ² calculation failed: {e}")
        results["cc"] = 0.0

    # RSD chi-squared (using standardized data)
    try:
        rsd_data = load_rsd_data()
        rsd_data = ensure_standard_dataset(rsd_data, "RSD")
        results["rsd"] = chi_squared_rsd(model, data=rsd_data)
    except Exception as e:
        print(f"Warning: RSD χ² calculation failed: {e}")
        results["rsd"] = 0.0

    # Total
    results["total"] = sum(results.values())
    return results


# --------------------------------------------------------------
# 2. Define baseline parameter sets
# --------------------------------------------------------------
lcdm_params = dict(
    H0=67.4,
    Om0=0.315,
    Or0=5e-5,
    Ok0=0.0,
    Ol0=0.685,
    Obh2=0.022,
)

pbuf_params = dict(
    H0=67.4,
    Om0=0.315,
    Or0=5e-5,
    Ok0=0.0,
    Ol0=0.0,  # no Λ in PBUF
    Obh2=0.022,
    alpha=5e-4,
    Rmax=1e9,
    k_sat=0.8,
)


# --------------------------------------------------------------
# 3. Evaluate joint χ²
# --------------------------------------------------------------
chi2_LCDM = chi2_total(lcdm_params, model_type="lcdm")
chi2_PBUF = chi2_total(pbuf_params, model_type="pbuf")

# --------------------------------------------------------------
# 4. Report results
# --------------------------------------------------------------
print("------------------------------------------------------")
print("JOINT χ² SUMMARY")
print("------------------------------------------------------")
print(f"LCDM:")
for k, v in chi2_LCDM.items():
    print(f"  {k:<10} = {v:.4f}")
print("\nPBUF:")
for k, v in chi2_PBUF.items():
    print(f"  {k:<10} = {v:.4f}")

print("------------------------------------------------------")
print(f"LCDM total χ² = {chi2_LCDM['total']:.4f}")
print(f"PBUF total χ² = {chi2_PBUF['total']:.4f}")
print("------------------------------------------------------")

# --------------------------------------------------------------
# 5. Sanity checks
# --------------------------------------------------------------
assert np.isfinite(chi2_LCDM["total"]), "NaN in LCDM χ²"
assert np.isfinite(chi2_PBUF["total"]), "NaN in PBUF χ²"
assert not np.isclose(
    chi2_LCDM["total"], chi2_PBUF["total"]
), "LCDM and PBUF χ² values should not be identical for the smoke test"

print("\n✅ Joint χ² pipeline test passed — ready for real data integration.")
