"""
Compute predicted H(z) for cosmic chronometer comparison.

This module provides functions to calculate theoretical H(z)
predictions from cosmological models for comparison with
cosmic chronometer measurements.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.fits._dataset_loader import load_cc_dataset

def compute_cc_hubble_model(model, z_values):
    """
    Compute the theoretical H(z) for an array of redshifts,
    in km/s/Mpc, to compare with cosmic chronometer data.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with H(z) method
    z_values : array-like
        Redshift values

    Returns
    -------
    np.ndarray
        H_th(z) [km/s/Mpc] for each given z

    Notes
    -----
    Cosmic chronometers provide direct measurements of H(z) through
    differential aging of passively evolving galaxies. This function
    computes the theoretical prediction for the same quantity.
    """
    z = np.asarray(z_values, dtype=float)
    Hz = np.array([model.H(zi) for zi in z], dtype=float)
    return Hz


def chi2_cc(model_params, model_type="lcdm"):
    """
    Compute χ² between model predictions and cosmic chronometer data.

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
    chi_squared_cc function using the PBUF data format.
    """
    from .chi2 import chi_squared_cc

    # Create model instance from parameters
    model_type_lower = model_type.lower()
    h = model_params["H0"] / 100.0

    if model_type_lower == "lcdm":
        # LCDM parameters
        required_params = ["H0", "Om0", "Ok0", "Ol0", "Or0", "Obh2"]
        for param in required_params:
            if param not in model_params:
                raise ValueError(f"Missing LCDM parameter: {param}")

        omega_b = model_params["Obh2"] / (h**2)

        model = LCDM(
            omega_m=model_params["Om0"],
            omega_lambda=model_params["Ol0"],
            h=h,
            omega_k=model_params["Ok0"],
            omega_r=model_params["Or0"],
            omega_b=omega_b,
        )
    elif model_type_lower == "pbuf":
        # PBUF parameters
        required_params = ["H0", "Om0", "Ok0", "Or0", "Obh2", "alpha", "Rmax", "k_sat"]
        for param in required_params:
            if param not in model_params:
                raise ValueError(f"Missing PBUF parameter: {param}")

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
    data = load_cc_dataset()
    data = ensure_standard_dataset(data, "CC")
    return chi_squared_cc(model, data)
