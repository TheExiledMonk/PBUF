"""
Compute supernova distance modulus predictions from cosmological models.

This module provides functions to calculate theoretical distance modulus
predictions μ(z) for cosmological models, which can be compared against
observed supernova data.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset

from cosmos.helper.distances import luminosity_distance
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.fits._dataset_loader import load_sn_dataset

def compute_sn_mu_model(model, z, M=0.0):
    """
    Compute predicted SN distance modulus μ_th(z) for a given cosmological model.

    The distance modulus is defined as:
    μ(z) = 5 * log10(D_L(z) / 10 pc) = 5 * log10(D_L(z) * 1e6) - 5

    where D_L(z) is the luminosity distance in Mpc.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with H(z) method
    z : array-like
        Redshift(s) at which to evaluate the model
    M : float, optional
        Absolute magnitude offset (nuisance parameter, degenerate with H0)

    Returns
    -------
    np.ndarray
        Model-predicted distance modulus μ(z) values

    Notes
    -----
    The absolute magnitude offset M accounts for calibration uncertainties
    and is degenerate with the Hubble constant H0. It is typically treated
    as a nuisance parameter in supernova fitting.
    """
    # Compute luminosity distance in Mpc
    D_L = luminosity_distance(z, model)

    # Convert to distance modulus
    # D_L is in Mpc, we need D_L in units where 10 pc = 1
    # 1 Mpc = 1e6 pc, so D_L_Mpc * 1e6 gives distance in units of 10 pc
    mu_th = 5.0 * np.log10(D_L * 1e6) - 5.0 + M

    return mu_th


def chi2_sn(model_params, model_type="lcdm"):
    """
    Compute χ² between model predictions and supernova data.

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
    chi_squared_sn function using the PBUF data format.
    """
    from .chi2 import chi_squared_sn

    # Create model instance from parameters
    if model_type.lower() == "lcdm":
        # LCDM parameters
        required_params = ["H0", "Om0", "Ok0", "Ol0", "Or0", "Obh2"]
        for param in required_params:
            if param not in model_params:
                raise ValueError(f"Missing LCDM parameter: {param}")

        model = LCDM(
            omega_m=model_params["Om0"],
            omega_lambda=model_params["Ol0"],
            h=model_params["H0"] / 100.0,
            omega_k=model_params["Ok0"],
            omega_r=model_params["Or0"],
            omega_b=model_params["Obh2"],
        )
    elif model_type.lower() == "pbuf":
        # PBUF parameters
        required_params = ["H0", "Om0", "Ok0", "Or0", "Obh2", "alpha", "Rmax", "k_sat"]
        for param in required_params:
            if param not in model_params:
                raise ValueError(f"Missing PBUF parameter: {param}")

        elastic_kwargs = {
            "eps0": model_params.get("eps0", 0.7),
            "n_alpha": model_params.get("n_alpha", 0.0),
            "n_eps": model_params.get("n_eps", 0.0),
            "n_R": model_params.get("n_R", 0.0),
        }

        model = PBUF(
            omega_m=model_params["Om0"],
            h=model_params["H0"] / 100.0,
            alpha=model_params["alpha"],
            Rmax=model_params["Rmax"],
            k_sat=model_params.get("k_sat", 1.0),
            eps0=elastic_kwargs["eps0"],
            n_alpha=elastic_kwargs["n_alpha"],
            n_eps=elastic_kwargs["n_eps"],
            n_R=elastic_kwargs["n_R"],
            omega_k=model_params["Ok0"],
            omega_r=model_params["Or0"],
            omega_b=model_params["Obh2"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lcdm' or 'pbuf'")

    # Load standardized data and compute χ²
    data = load_sn_dataset()
    data = ensure_standard_dataset(data, "SN")
    return chi_squared_sn(model, data=data)
