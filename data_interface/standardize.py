"""
PBUF Data Interface Standardization
===================================

Defines the canonical schema for all cosmological datasets used in PBUF fits.
All data loaders and χ² evaluators must return or accept dictionaries that
conform to this standard.

Schema (PBUF Data Object v1)
----------------------------
Each dataset must be a dict with the following keys:

    {
        "name": str,             # e.g. "Planck2018", "Pantheon+", "DR16"
        "type": str,             # one of: "CMB", "SN", "BAO_ISO", "BAO_ANISO", "CC", "RSD", "WL"
        "z": np.ndarray | None,  # redshifts (None for CMB)
        "obs": np.ndarray,       # observed values (e.g. μ, R, D_V/rd, etc.)
        "err": np.ndarray | None,# 1σ uncertainties (optional if covariance provided)
        "cov": np.ndarray | None,# covariance matrix (optional)
        "meta": dict             # additional info (units, survey, reference, etc.)
    }

All χ² evaluators and loaders must use these fields — no variations allowed.
"""

import numpy as np

# -------------------------------------------------------------------------
# Validation helper
# -------------------------------------------------------------------------

def ensure_standard_dataset(data, expected_type: str):
    """
    Validate and normalize a dataset dictionary to match the PBUF data schema.

    Parameters
    ----------
    data : dict
        Input dataset dictionary (e.g. from loader or pipeline).
    expected_type : str
        Expected dataset type (CMB, SN, BAO_ISO, BAO_ANISO, CC, RSD, WL).

    Returns
    -------
    dict
        Normalized dataset dictionary matching PBUF schema.
    """
    if data is None:
        raise ValueError("No dataset provided to ensure_standard_dataset().")

    # Type enforcement
    dtype = data.get("type", expected_type)
    if dtype != expected_type:
        raise ValueError(f"Dataset type mismatch: expected '{expected_type}', got '{dtype}'")
    data["type"] = dtype

    # WL datasets have different validation requirements
    if expected_type == "WL":
        return _validate_wl_dataset_schema(data)

    # Standard validation for other dataset types
    # Ensure required keys
    defaults = {
        "name": "unknown",
        "z": None,
        "obs": None,
        "err": None,
        "cov": None,
        "meta": {},
    }
    for key, val in defaults.items():
        data.setdefault(key, val)

    # Type & shape sanity checks
    if data["obs"] is not None:
        obs = np.asarray(data["obs"], dtype=float)
        data["obs"] = obs

        if data["err"] is not None:
            err = np.asarray(data["err"], dtype=float)
            if err.shape != obs.shape:
                raise ValueError(f"obs/err shape mismatch: {obs.shape} vs {err.shape}")
            data["err"] = err

        if data["cov"] is not None:
            cov = np.asarray(data["cov"], dtype=float)
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError(f"Invalid covariance matrix shape: {cov.shape}")
            data["cov"] = cov

    return data


def _validate_wl_dataset_schema(data: dict):
    """
    Validate WL dataset against WL-specific schema requirements.
    
    Parameters
    ----------
    data : dict
        WL dataset dictionary
        
    Returns
    -------
    dict
        Validated WL dataset
    """
    import numpy as np
    
    # Ensure required standard keys exist (but may be None for WL)
    defaults = {
        "name": "unknown_wl_survey",
        "z": None,
        "obs": None, 
        "err": None,
        "cov": None,
        "meta": {},
    }
    for key, val in defaults.items():
        data.setdefault(key, val)
    
    # WL-specific required fields
    wl_required = [
        "theta_bins", "data_vector", "covariance", "tomo_pairs",
        "n_of_z", "z_grid", "shear_m"
    ]
    
    for field in wl_required:
        if field not in data:
            raise ValueError(f"Missing required WL field: {field}")
    
    # Validate WL field types and basic structure
    data["theta_bins"] = np.asarray(data["theta_bins"], dtype=float)
    data["data_vector"] = np.asarray(data["data_vector"], dtype=float)
    data["covariance"] = np.asarray(data["covariance"], dtype=float)
    data["z_grid"] = np.asarray(data["z_grid"], dtype=float)
    data["shear_m"] = np.asarray(data["shear_m"], dtype=float)
    
    # Handle tomo_pairs (convert from numpy array if needed)
    tomo_pairs = data["tomo_pairs"]
    if isinstance(tomo_pairs, np.ndarray):
        if tomo_pairs.ndim == 2:
            data["tomo_pairs"] = [(int(tomo_pairs[i, 0]), int(tomo_pairs[i, 1])) for i in range(tomo_pairs.shape[0])]
        else:
            raise ValueError(f"tomo_pairs array must be 2D, got shape {tomo_pairs.shape}")
    elif isinstance(tomo_pairs, (list, tuple)):
        data["tomo_pairs"] = [(int(pair[0]), int(pair[1])) for pair in tomo_pairs]
    else:
        raise ValueError("tomo_pairs must be list/array of pairs")
    
    # Validate n_of_z as list of arrays (handle numpy array from npz files)
    n_of_z = data["n_of_z"]
    if isinstance(n_of_z, np.ndarray):
        # Convert from 2D numpy array (from npz file) to list of 1D arrays
        if n_of_z.ndim == 2:
            data["n_of_z"] = [n_of_z[i] for i in range(n_of_z.shape[0])]
        else:
            raise ValueError(f"n_of_z array must be 2D, got shape {n_of_z.shape}")
    elif isinstance(n_of_z, (list, tuple)):
        data["n_of_z"] = [np.asarray(nz, dtype=float) for nz in n_of_z]
    else:
        raise ValueError("n_of_z must be list/array of arrays")
    
    # Basic shape validation
    if data["theta_bins"].ndim != 1:
        raise ValueError(f"theta_bins must be 1D, got shape {data['theta_bins'].shape}")
    if data["data_vector"].ndim != 1:
        raise ValueError(f"data_vector must be 1D, got shape {data['data_vector'].shape}")
    if data["covariance"].ndim != 2:
        raise ValueError(f"covariance must be 2D, got shape {data['covariance'].shape}")
    if data["covariance"].shape[0] != data["covariance"].shape[1]:
        raise ValueError(f"covariance must be square, got shape {data['covariance'].shape}")
    if data["covariance"].shape[0] != len(data["data_vector"]):
        raise ValueError(f"covariance size {data['covariance'].shape[0]} != data_vector length {len(data['data_vector'])}")
    
    return data


# -------------------------------------------------------------------------
# Utility: dataset summary printer
# -------------------------------------------------------------------------

def describe_dataset(data):
    """
    Print a compact human-readable summary of a dataset dictionary.
    """
    name = data.get("name", "unknown")
    dtype = data.get("type", "unknown")
    npts = len(data["obs"]) if data["obs"] is not None else 0
    print(f"[{dtype}] {name} — {npts} points")

    if data["z"] is not None:
        print(f"  z range: {np.min(data['z']):.3f} – {np.max(data['z']):.3f}")
    if data["cov"] is not None:
        print(f"  Covariance matrix: {data['cov'].shape}")
    elif data["err"] is not None:
        print(f"  Mean σ: {np.mean(data['err']):.4g}")
    if "meta" in data and data["meta"]:
        print(f"  Meta: {data['meta']}")


# -------------------------------------------------------------------------
# Legacy format converters (for existing loaders)
# -------------------------------------------------------------------------

def convert_bao_to_standard(bao_data: dict, obs_type: str = "combined"):
    """
    Convert legacy BAO loader format to standard PBUF format.

    Parameters
    ----------
    bao_data : dict
        Legacy BAO data from bao_loader.py
    obs_type : str
        Which observable to use: "DV" (isotropic), "DM" (transverse),
        "Hz" (radial), or "combined" (average of DM and Hz)

    Returns
    -------
    dict
        Standardized BAO dataset
    """
    if obs_type == "combined":
        # Average of transverse and radial measurements
        obs = (bao_data["DM_rd"] + bao_data["Hz_rd"]) / 2
        obs_name = "DM+Hz averaged"
    elif obs_type == "DV":
        obs = bao_data["DV_rd"]
        obs_name = "D_V isotropic"
    elif obs_type == "DM":
        obs = bao_data["DM_rd"]
        obs_name = "D_M transverse"
    elif obs_type == "Hz":
        obs = bao_data["Hz_rd"]
        obs_name = "H radial"
    else:
        raise ValueError(f"Unknown obs_type: {obs_type}")

    return {
        "name": "BAO_DR16",
        "type": "BAO_ANISO" if obs_type != "DV" else "BAO_ISO",
        "z": bao_data["z"],
        "obs": obs,
        "err": bao_data["err"],
        "cov": bao_data["cov"],
        "meta": {
            "survey": "SDSS DR16",
            "observable": obs_name,
            "reference": "Alam et al. 2021"
        }
    }


def convert_sn_to_standard(sn_data: dict):
    """
    Convert legacy SN loader format to standard PBUF format.

    Parameters
    ----------
    sn_data : dict
        Legacy SN data from sn_loader.py

    Returns
    -------
    dict
        Standardized SN dataset
    """
    return {
        "name": "Pantheon+",
        "type": "SN",
        "z": sn_data["z"],
        "obs": sn_data["obs"],
        "err": sn_data["err"],
        "cov": sn_data["cov"],
        "meta": {
            "survey": "Pantheon+",
            "observable": "distance modulus",
            "units": "mag",
            "reference": "Scolnic et al. 2022"
        }
    }


def convert_cmb_to_standard(cmb_data: dict):
    """
    Convert legacy CMB loader format to standard PBUF format.

    Parameters
    ----------
    cmb_data : dict
        Legacy CMB data from cmb_loader.py

    Returns
    -------
    dict
        Standardized CMB dataset
    """
    # CMB observables: [R, l_a, theta_star]
    obs = cmb_data["mean"]

    # For CMB, we need a covariance matrix, not individual errors
    # Create diagonal error matrix from covariance for compatibility
    cov = cmb_data["cov"]
    err = np.sqrt(np.diag(cov)) if cov is not None else None

    return {
        "name": "Planck2018",
        "type": "CMB",
        "z": None,
        "obs": obs,
        "err": err,
        "cov": cov,
        "meta": {
            "survey": "Planck 2018",
            "observable": "distance priors",
            "parameters": cmb_data.get("labels", ["R", "la", "theta_star"]),
            "reference": "Planck Collaboration 2020"
        }
    }


def convert_cc_to_standard(cc_data: dict):
    """
    Convert legacy CC loader format to standard PBUF format.

    Parameters
    ----------
    cc_data : dict
        Legacy CC data from cc_loader.py

    Returns
    -------
    dict
        Standardized CC dataset
    """
    return {
        "name": "CC_compilation",
        "type": "CC",
        "z": cc_data["z"],
        "obs": cc_data["obs"],
        "err": cc_data["err"],
        "cov": cc_data["cov"],
        "meta": {
            "survey": "Various",
            "observable": "Hubble parameter H(z)",
            "units": "km/s/Mpc",
            "reference": "Compilation"
        }
    }


def convert_rsd_to_standard(rsd_data: dict):
    """
    Convert legacy RSD loader format to standard PBUF format.

    Parameters
    ----------
    rsd_data : dict
        Legacy RSD data from rsd_loader.py

    Returns
    -------
    dict
        Standardized RSD dataset
    """
    return {
        "name": "RSD_compilation",
        "type": "RSD",
        "z": rsd_data["z"],
        "obs": rsd_data["obs"],
        "err": rsd_data["err"],
        "cov": rsd_data["cov"],
        "meta": {
            "survey": "Various",
            "observable": "growth rate fσ₈",
            "reference": "Compilation"
        }
    }


def convert_wl_to_standard(wl_data: dict):
    """
    Convert legacy WL loader format to standard PBUF format.

    Parameters
    ----------
    wl_data : dict
        Legacy WL data from wl_loader.py

    Returns
    -------
    dict
        Standardized WL dataset
    """
    import numpy as np
    
    # WL datasets use different field names than standard schema
    standardized = {
        "name": wl_data.get("name", "WL_survey"),
        "type": "WL",
        "z": None,  # WL uses z_grid instead
        "obs": None,  # WL uses data_vector instead
        "err": None,  # WL uses covariance instead
        "cov": None,  # WL uses covariance instead
        "meta": wl_data.get("meta", {}),
        
        # WL-specific fields
        "theta_bins": np.asarray(wl_data["theta_bins"], dtype=float),
        "data_vector": np.asarray(wl_data["data_vector"], dtype=float),
        "covariance": np.asarray(wl_data["covariance"], dtype=float),
        "tomo_pairs": wl_data["tomo_pairs"],
        "n_of_z": [np.asarray(nz, dtype=float) for nz in wl_data["n_of_z"]],
        "z_grid": np.asarray(wl_data["z_grid"], dtype=float),
        "shear_m": np.asarray(wl_data["shear_m"], dtype=float),
    }
    
    # Optional fields
    if "S8_obs" in wl_data:
        standardized["S8_obs"] = wl_data["S8_obs"]
    
    # Ensure meta has required WL fields
    meta = standardized["meta"]
    meta.setdefault("survey", standardized["name"])
    meta.setdefault("observable", "cosmic shear ξ±(θ)")
    meta.setdefault("theta_units", "arcmin")
    meta.setdefault("data_order", "xi_plus_then_xi_minus")
    meta.setdefault("n_tomo_bins", len(standardized["n_of_z"]))
    
    return standardized


# -------------------------------------------------------------------------
# Batch converter for all datasets
# -------------------------------------------------------------------------

def standardize_all_datasets(datasets: dict):
    """
    Convert a full dataset dictionary to standard format.

    Parameters
    ----------
    datasets : dict
        Legacy dataset dictionary from data_interface.__init__

    Returns
    -------
    dict
        All datasets converted to standard format
    """
    converters = {
        "bao": lambda d: convert_bao_to_standard(d, "combined"),
        "sn": convert_sn_to_standard,
        "cmb": convert_cmb_to_standard,
        "cc": convert_cc_to_standard,
        "rsd": convert_rsd_to_standard,
        "wl": convert_wl_to_standard,
    }

    standardized = {}
    for key, data in datasets.items():
        if key in converters:
            standardized[key] = converters[key](data)
            # Validate the result
            standardized[key] = ensure_standard_dataset(standardized[key], standardized[key]["type"])
        else:
            print(f"⚠️  No converter for dataset type: {key}")

    return standardized
