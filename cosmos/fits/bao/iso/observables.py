"""
Compute isotropic BAO observables for a cosmological model.

We compute D_V(z) / r_d, where
    D_V(z) = [ (1+z)^2 * D_A(z)^2 * c*z / H(z) ]^(1/3)
and
    D_A(z) = D_M(z) / (1+z)
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset

from cosmos.helper.constants import C_LIGHT
from cosmos.helper.distances import transverse_comoving_distance, sound_horizon
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.fits._dataset_loader import load_bao_iso_dataset

def compute_bao_dv_over_rd(model, z_values):
    """
    Compute isotropic BAO observable D_V(z)/r_d for given redshifts.

    Parameters
    ----------
    model : LCDM or PBUF instance
    z_values : array-like
        Redshifts at which to compute BAO observable.

    Returns
    -------
    np.ndarray
        D_V(z)/r_d for each z.
    """
    z = np.asarray(z_values, dtype=float)
    results = []

    r_d = sound_horizon(model, z_drag=None)  # baryon drag sound horizon [Mpc]

    for zi in z:
        D_M = transverse_comoving_distance(zi, model)  # [Mpc]
        H_z = model.H(zi)  # [km/s/Mpc]
        D_A = D_M / (1.0 + zi)
        # D_V(z) = [(1+z)^2 * D_A(z)^2 * (c*z)/H(z)]^(1/3)
        # Since D_A = D_M/(1+z), then D_M = D_A*(1+z)
        # So D_V(z) = [D_M^2 * (c*z)/H(z)]^(1/3) = [D_A^2 * (1+z)^2 * (c*z)/H(z)]^(1/3)
        D_V = ((D_A**2 * (1.0 + zi)**2 * (C_LIGHT * zi / (1000.0 * H_z))) ** (1.0 / 3.0))  # [Mpc]
        results.append(D_V / r_d)

    return np.array(results)


def chi2_bao_iso(model_params, model_type="lcdm"):
    """
    Compute χ² between model predictions and isotropic BAO data.

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
    chi_squared_bao_iso function using the PBUF data format.
    """
    from .chi2 import chi_squared_bao_iso

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
    data = load_bao_iso_dataset()
    data = ensure_standard_dataset(data, "BAO_ISO")
    return chi_squared_bao_iso(model, data)
