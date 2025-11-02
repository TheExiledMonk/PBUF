"""
Compute theoretical RSD observable fσ₈(z) using the full growth solver.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset

from cosmos.helper.growth import fsigma8
from cosmos.lcdm.model import LCDM
from cosmos.optim.parameter_defaults import SIGMA8_PLANCK
from cosmos.pbuf.model import PBUF
from cosmos.fits._dataset_loader import load_rsd_dataset


def compute_rsd_observable(model, z_values, sigma8_0=SIGMA8_PLANCK):
    """
    Compute fσ8(z) = f(z) * σ8(z) for given redshifts via numerical growth.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Must provide H(z) and density_parameters_at_z(z).
    z_values : array-like
        Redshift array
    sigma8_0 : float
        σ8 normalization at z=0. Can be supplied per model.

    Returns
    -------
    np.ndarray
        fσ8(z) values with the same shape as z_values.
    """
    z = np.asarray(z_values, dtype=float)
    fs8 = np.asarray(fsigma8(z, model, sigma8_0=sigma8_0), dtype=float)
    return fs8.reshape(z.shape)


def chi2_rsd(model_params, model_type="lcdm", sigma8_0=None):
    """
    Compute χ² between model predictions and RSD fσ8 data.

    Parameters
    ----------
    model_params : dict
        Cosmological parameters (may include optional 'sigma8_0').
    model_type : str
        "lcdm" or "pbuf"
    sigma8_0 : float or None
        Override for σ8 normalization. If None, pulled from model_params
        when present, otherwise defaults to 0.811.

    Returns
    -------
    float
        χ² value ≥ 0
    """
    from .chi2 import chi_squared_rsd

    params = dict(model_params)
    sigma8 = params.pop("sigma8_0", None)
    if sigma8_0 is not None:
        sigma8 = sigma8_0

    model_type_lower = model_type.lower()
    if sigma8 is None and model_type_lower != "pbuf":
        sigma8 = SIGMA8_PLANCK

    # Create model instance from parameters
    if model_type_lower == "lcdm":
        # LCDM parameters
        required_params = ["H0", "Om0", "Ok0", "Ol0", "Or0", "Obh2"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing LCDM parameter: {param}")

        model = LCDM(
            omega_m=params["Om0"],
            omega_lambda=params["Ol0"],
            h=params["H0"] / 100.0,
            omega_k=params["Ok0"],
            omega_r=params["Or0"],
            omega_b=params["Obh2"],
        )
    elif model_type_lower == "pbuf":
        # PBUF parameters
        required_params = ["H0", "Om0", "Ok0", "Or0", "Obh2", "alpha", "Rmax", "k_sat"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing PBUF parameter: {param}")

        elastic_kwargs = {
            "eps0": params.get("eps0", 0.7),
            "n_alpha": params.get("n_alpha", 0.0),
            "n_eps": params.get("n_eps", 0.0),
            "n_R": params.get("n_R", 0.0),
        }

        model = PBUF(
            omega_m=params["Om0"],
            h=params["H0"] / 100.0,
            alpha=params["alpha"],
            Rmax=params["Rmax"],
            k_sat=params.get("k_sat", 1.0),
            eps0=elastic_kwargs["eps0"],
            n_alpha=elastic_kwargs["n_alpha"],
            n_eps=elastic_kwargs["n_eps"],
            n_R=elastic_kwargs["n_R"],
            omega_k=params["Ok0"],
            omega_r=params["Or0"],
            omega_b=params["Obh2"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lcdm' or 'pbuf'")

    # Load standardized data and compute χ²
    data = load_rsd_dataset()
    data = ensure_standard_dataset(data, "RSD")
    if sigma8 is None:
        return chi_squared_rsd(model, data)
    return chi_squared_rsd(model, data, sigma8_0=sigma8)
