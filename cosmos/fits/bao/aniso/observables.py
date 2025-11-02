"""
Compute anisotropic BAO observables for a cosmological model.

We return, for each z:
    X1 = D_M(z) / r_d
    X2 = D_H(z) / r_d = c / [H(z) * r_d]
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset

from cosmos.helper.constants import C_LIGHT
from cosmos.helper.distances import (
    transverse_comoving_distance,
    sound_horizon,
)
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.fits._dataset_loader import load_bao_aniso_dataset

def compute_bao_anisotropic_observables(model, z_values):
    """
    Compute anisotropic BAO observables for a set of redshifts.

    Parameters
    ----------
    model : LCDM or PBUF instance
    z_values : array-like
        Redshifts

    Returns
    -------
    dict
        {
            "DM_over_rd": array shape (N,),
            "DH_over_rd": array shape (N,)
        }
    """
    z = np.asarray(z_values, dtype=float)
    rd = sound_horizon(model, z_drag=None)  # r_d (a.k.a. r_drag), in Mpc
    c_kms = C_LIGHT / 1000.0  # speed of light in km/s for consistency with H(z)

    DM_over_rd = []
    DH_over_rd = []

    for zi in z:
        D_M = transverse_comoving_distance(zi, model)   # [Mpc]
        H_z = model.H(zi)                                # [km/s/Mpc]

        DM_over_rd.append(D_M / rd)

        # D_H(z) / r_d = c / [H(z) * r_d]  (dimensionless)
        DH_over_rd.append(c_kms / (H_z * rd))

    return {
        "DM_over_rd": np.array(DM_over_rd, dtype=float),
        "DH_over_rd": np.array(DH_over_rd, dtype=float),
    }


def chi2_bao_aniso(model_params, model_type="lcdm"):
    """
    Compute χ² between model predictions and anisotropic BAO data.

    Parameters
    ----------
    model_params : dict
        Cosmological parameters
    model_type : str
        "lcdm" or "pbuf"

    Returns
    -------
    float
        χ² value ≥ 0

    Notes
    -----
    This function creates a model instance and calls the standardized
    chi_squared_bao_aniso function using the PBUF data format.
    """
    from .chi2 import chi_squared_bao_aniso

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
            omega_b=omega_b,
        )
    elif model_type.lower() == "pbuf":
        # PBUF parameters
        required_params = ["H0", "Om0", "Ok0", "Or0", "Obh2", "alpha", "Rmax", "k_sat"]
        for param in required_params:
            if param not in model_params:
                raise ValueError(f"Missing PBUF parameter: {param}")

        h = model_params["H0"] / 100.0
        omega_b = model_params["Obh2"] / (h**2)

        elastic_kwargs = {
            "eps0": model_params.get("eps0", 0.7),
            "n_alpha": model_params.get("n_alpha", 0.0),
            "n_eps": model_params.get("n_eps", 0.0),
            "n_R": model_params.get("n_R", 0.0),
        }

        model = PBUF(
            omega_m=model_params["Om0"],
            h=h,
            alpha=model_params["alpha"],
            Rmax=model_params["Rmax"],
            k_sat=model_params.get("k_sat", 1.0),
            eps0=elastic_kwargs["eps0"],
            n_alpha=elastic_kwargs["n_alpha"],
            n_eps=elastic_kwargs["n_eps"],
            n_R=elastic_kwargs["n_R"],
            omega_k=model_params["Ok0"],
            omega_r=model_params["Or0"],
            omega_b=omega_b,
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lcdm' or 'pbuf'")

    # Load standardized data and compute χ²
    data = load_bao_aniso_dataset()
    data = ensure_standard_dataset(data, "BAO_ANISO")
    return chi_squared_bao_aniso(model, data)
