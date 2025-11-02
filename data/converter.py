"""
Data Converter - Convert raw datasets to standardized .npz format.

This module converts raw datasets from data/raw/ into the standardized
.npz format used by the fitting routines. The output follows the CLI spec
schema for reproducibility and consistency.

Standard format for .npz files:
{
  "name": str,           # dataset identifier
  "z": array or float,   # redshifts (float for CMB)
  "obs": array,          # observed values
  "cov": array,          # covariance matrix
  "labels": list,        # parameter labels
  "n_data": int,         # number of data points
  "meta": dict          # additional metadata
}
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import os
import pandas as pd

# Import existing converters
from data_interface.standardize import (
    convert_cmb_to_standard,
    convert_sn_to_standard,
    convert_bao_to_standard,
    convert_cc_to_standard,
    convert_rsd_to_standard,
    ensure_standard_dataset
)
from data_interface import load_all_datasets


def detect_dataset_type(raw_dir: Path) -> str:
    """
    Detect dataset type by examining file contents.

    Returns
    -------
    str
        Detected type: "CMB", "SN", "BAO", "CC", "RSD", "SH0ES", or "unknown"
    """
    # Look for all data files
    all_files = list(raw_dir.glob("**/*.dat")) + list(raw_dir.glob("**/*.txt")) + \
                list(raw_dir.glob("**/*.json")) + list(raw_dir.glob("**/*.fits"))

    if not all_files:
        return "unknown"

    # Try to detect by content
    for data_file in all_files:
        try:
            # Try JSON files first (CMB, BAO)
            if data_file.suffix == ".json":
                with open(data_file, "r") as f:
                    data = json.load(f)

                # CMB detection
                if "mean" in data and "covariance" in data:
                    return "CMB"

                # BAO detection
                if any(key in data for key in ["DV_rd", "DM_rd", "Hz_rd"]):
                    return "BAO"

            # Try text/dat files (SN, CC, RSD, SH0ES)
            elif data_file.suffix in [".dat", ".txt"]:
                # SH0ES detection - look for lstsq_results.txt
                if "lstsq_results" in data_file.name:
                    return "SH0ES"

                # Try to read as CSV/TSV
                df = pd.read_csv(data_file, sep=r'\s+', comment='#')

                # BAO detection - check for specific types
                if "DV_div_rd" in df.columns or "DV_over_rd" in df.columns:
                    return "BAO_ISO"
                elif (
                    "DM_over_rd" in df.columns
                    and (
                        "DH_over_rd" in df.columns
                        or "Hz_rd_over_c" in df.columns
                    )
                ):
                    return "BAO_ANISO"
                # SN detection - look for distance modulus columns
                elif any(col in df.columns for col in ['MU_SH0ES', 'mu', 'MU', 'm_b_corr']):
                    return "SN"

                # CC detection - look for H(z) like columns
                elif any(col in df.columns for col in ['H', 'Hz', 'H_z']):
                    return "CC"

                # RSD detection - look for fsigma8 like columns
                elif any(col in df.columns for col in ['fsigma8', 'f_s8', 'fs8']):
                    return "RSD"

                # Generic BAO detection
                elif any(col in df.columns for col in ['DV_rd', 'DM_rd', 'Hz_rd']):
                    return "BAO"

            # Try FITS files (SH0ES specific)
            elif data_file.suffix == ".fits":
                # SH0ES uses FITS files for covariance matrices
                if any(keyword in data_file.name for keyword in ['allc', 'alll', 'ally']):
                    return "SH0ES"

        except Exception:
            continue

    return "unknown"


def convert_dataset(source: str, output_path: str, dataset_type: str = None) -> dict:
    """
    Convert raw dataset to standardized .npz format.

    Parameters
    ----------
    source : str
        Source dataset name (directory in data/raw/)
    output_path : str
        Output .npz file path
    dataset_type : str, optional
        Explicit dataset type. If None, will auto-detect from content.

    Returns
    -------
    dict
        Standardized dataset dictionary
    """
    print(f"🔄 Converting {source} to standardized format...")

    # Check if raw data exists
    raw_dir = Path(f"data/raw/{source}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    # Auto-detect type if not specified
    if dataset_type is None:
        detected_type = detect_dataset_type(raw_dir)
        if detected_type != "unknown":
            print(f"📊 Auto-detected dataset type: {detected_type}")
            dataset_type = detected_type
        else:
            print(f"⚠️ Could not auto-detect dataset type, using pattern matching...")
            # Fall back to pattern-based detection
            if "pantheon" in source.lower() or "sn" in source.lower():
                dataset_type = "SN"
            elif "sh0es" in source.lower() or "shoes" in source.lower():
                dataset_type = "SH0ES"
            elif "planck" in source.lower() or "cmb" in source.lower():
                dataset_type = "CMB"
            elif "bao" in source.lower():
                # Try to determine if isotropic or anisotropic from filename
                if "iso" in source.lower() or "dv" in source.lower():
                    dataset_type = "BAO_ISO"
                elif "aniso" in source.lower() or ("dm" in source.lower() and "hz" in source.lower()):
                    dataset_type = "BAO_ANISO"
                else:
                    dataset_type = "BAO_ANISO"  # Default to anisotropic
            elif "cc" in source.lower():
                dataset_type = "CC"
            elif "rsd" in source.lower():
                dataset_type = "RSD"
            else:
                dataset_type = "unknown"

    if dataset_type == "unknown":
        raise ValueError(f"Could not determine dataset type for {source}")

    print(f"🔍 Using dataset type: {dataset_type}")

    # Load raw data based on detected type
    if dataset_type == "CMB":
        data_dict = _load_cmb_data(raw_dir)
    elif dataset_type == "SN":
        data_dict = _load_sn_data(raw_dir)
    elif dataset_type == "BAO_ISO":
        data_dict = _load_bao_iso_data(raw_dir)
    elif dataset_type == "BAO_ANISO":
        data_dict = _load_bao_aniso_data(raw_dir)
    elif dataset_type == "BAO":
        data_dict = _load_bao_data(raw_dir)
    elif dataset_type == "CC":
        data_dict = _load_cc_data(raw_dir)
    elif dataset_type == "RSD":
        data_dict = _load_rsd_data(raw_dir)
    elif dataset_type == "SH0ES":
        data_dict = _load_shoes_data(raw_dir)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    if data_dict is None:
        raise NotImplementedError(f"Conversion not implemented for dataset type: {dataset_type}")

    # Ensure standard format
    try:
        standard_data = ensure_standard_dataset(data_dict, dataset_type)
    except Exception as e:
        print(f"⚠️ Standard validation failed: {e}")
        standard_data = data_dict

    # Convert to CLI spec format
    npz_dict = _convert_to_npz_format(standard_data, source)

    # Save to .npz file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, **npz_dict)

    # Print summary
    print(f"✅ Converted to {output_path}")
    print(f"   Dataset: {npz_dict['name']}")
    print(f"   Points: {npz_dict['n_data']}")
    print(f"   Labels: {npz_dict['labels']}")

    if npz_dict.get('z') is not None:
        if np.isscalar(npz_dict['z']):
            print(f"   Redshift: {npz_dict['z']}")
        else:
            print(f"   z-range: {np.min(npz_dict['z']):.3f} - {np.max(npz_dict['z']):.3f}")

    return npz_dict


def _get_labels_for_dataset(standard_data: dict) -> list:
    """Get appropriate labels for the dataset type."""
    dtype = standard_data.get("type", "unknown")

    if dtype == "CMB":
        return ["R", "l_A", "theta_star"]
    elif dtype == "SN":
        return ["mu"] * len(standard_data["obs"])
    elif dtype == "BAO_ISO":
        return ["D_V/rd"] * len(standard_data["obs"])
    elif dtype == "BAO_ANISO":
        n = len(standard_data["obs"])
        # Alternate between DM/rd and H*rd/c
        labels = []
        for i in range(n//2):
            labels.extend(["D_M/rd", "H*rd/c"])
        if n % 2 == 1:
            labels.append("D_M/rd")
        return labels
    elif dtype == "CC":
        return ["H(z)"] * len(standard_data["obs"])
    elif dtype == "RSD":
        return ["fsigma8"] * len(standard_data["obs"])
    else:
        return [f"obs_{i}" for i in range(len(standard_data["obs"]))]


def _load_cmb_data(raw_dir: Path):
    """Load CMB distance priors data."""
    # Look for JSON files or try standard loader
    json_files = list(raw_dir.glob("*.json"))
    if json_files:
        # Try to load Planck-style JSON
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                if "mean" in data and "covariance" in data:
                    # Convert to standard format
                    return {
                        "name": "Planck2018",
                        "type": "CMB",
                        "z": None,
                        "obs": np.array([data["mean"]["R"], data["mean"]["la"], data["mean"]["theta_star"]]),
                        "err": None,
                        "cov": np.array(data["covariance"]),
                        "meta": {
                            "survey": "Planck 2018",
                            "observable": "distance priors",
                            "parameters": ["R", "la", "theta_star"],
                            "reference": "Planck Collaboration 2020"
                        }
                    }
            except:
                continue

    # Fallback to standard loader
    try:
        from data_interface.cmb_loader import load_cmb_priors
        return load_cmb_priors()
    except:
        return None


def _load_sn_data(raw_dir: Path):
    """Load supernovae data."""
    # Look for Pantheon+ data anywhere in the raw directory structure
    pantheon_files = []
    for root, dirs, files in os.walk(raw_dir.parent.parent / "raw"):
        for file in files:
            if "Pantheon" in file and "SH0ES" in file and file.endswith('.dat'):
                pantheon_files.append(Path(root) / file)

    if pantheon_files:
        for data_file in pantheon_files:
            try:
                # Use pandas to handle headers
                df = pd.read_csv(data_file, sep=r'\s+', comment='#')

                # Check if it's Pantheon+ format by looking for MU_SH0ES column
                if 'MU_SH0ES' in df.columns and 'zHD' in df.columns:
                    z = df['zHD'].values
                    mu = df['MU_SH0ES'].values
                    mu_err = df['MU_SH0ES_ERR_DIAG'].values

                    return {
                        "name": "Pantheon+SH0ES",
                        "type": "SN",
                        "z": z,
                        "obs": mu,
                        "err": mu_err,
                        "cov": None,
                        "meta": {
                            "survey": "Pantheon+SH0ES",
                            "observable": "distance modulus μ",
                            "reference": "Pantheon+ Collaboration"
                        }
                    }
            except Exception as e:
                print(f"Failed to load Pantheon+ from {data_file}: {e}")
                continue

    # Look for other data files in the specified directory
    data_files = list(raw_dir.glob("**/*.dat")) + list(raw_dir.glob("**/*.txt"))
    if data_files:
        # Try to parse Pantheon-style data
        for data_file in data_files:
            try:
                # Use pandas to handle headers
                df = pd.read_csv(data_file, sep=r'\s+', comment='#')

                # Check if it's Pantheon+ format by looking for MU_SH0ES column
                if 'MU_SH0ES' in df.columns:
                    z = df['zHD'].values
                    mu = df['MU_SH0ES'].values
                    mu_err = df['MU_SH0ES_ERR_DIAG'].values
                    return {
                        "z": z,
                        "obs": mu,
                        "err": mu_err,
                        "cov": None
                    }
                else:
                    # Generic format: assume first 3 columns are z, mu, mu_err
                    data = df.values
                    if data.shape[1] >= 3:
                        z, mu, mu_err = data[:, 0], data[:, 1], data[:, 2]
                        return {
                            "z": z,
                            "obs": mu,
                            "err": mu_err,
                            "cov": None
                        }
            except Exception as e:
                print(f"Failed to load {data_file}: {e}")
                continue

    # Fallback to standard loader
    try:
        from data_interface.sn_loader import load_sn_data
        return load_sn_data()
    except:
        return None


def _load_bao_data(raw_dir: Path):
    """Load BAO data."""
    # Look for JSON files
    json_files = list(raw_dir.glob("*.json"))
    if json_files:
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Try different BAO formats
                if "DV_rd" in data or "DM_rd" in data:
                    return data
            except:
                continue

    # Fallback to standard loader
    try:
        from data_interface.bao_loader import load_bao_data
        return load_bao_data()
    except:
        return None


def _load_bao_iso_data(raw_dir: Path):
    """Load isotropic BAO data (D_V/rd measurements only)."""
    # Look for DESI BAO mean files (proper isotropic data)
    mean_files = list(raw_dir.glob("**/desi*mean*.txt"))
    all_z_values = []
    all_dv_values = []

    if mean_files:
        for mean_file in mean_files:
            try:
                # Check if this contains isotropic BAO data
                if "gaussian" in mean_file.name.lower() and "bao" in mean_file.name.lower():
                    with open(mean_file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                z = float(parts[0])
                                value = float(parts[1])
                                quantity = parts[2]

                                if 'DV_over_rs' in quantity or 'DV/rd' in quantity:
                                    all_z_values.append(z)
                                    all_dv_values.append(value)
                            except ValueError:
                                continue

            except Exception as e:
                print(f"Failed to load isotropic BAO from {mean_file}: {e}")
                continue

    if all_z_values and all_dv_values:
        # Sort by redshift
        sorted_indices = np.argsort(all_z_values)
        z_values = np.array([all_z_values[i] for i in sorted_indices])
        dv_values = np.array([all_dv_values[i] for i in sorted_indices])

        # Remove duplicates (same z and DV/rd within tolerance)
        unique_indices = []
        tolerance = 1e-6
        for i in range(len(z_values)):
            is_duplicate = False
            for j in unique_indices:
                if abs(z_values[i] - z_values[j]) < tolerance and abs(dv_values[i] - dv_values[j]) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_indices.append(i)

        z_values = z_values[unique_indices]
        dv_values = dv_values[unique_indices]

        # For now, assume errors are ~5% of DV/rd (typical for BAO)
        DV_err = 0.05 * dv_values

        return {
            "name": "DESI_BAO_iso",
            "type": "BAO_ISO",
            "z": z_values,
            "obs": dv_values,
            "err": DV_err,
            "cov": None,
            "meta": {
                "survey": "DESI",
                "observable": "D_V(z)/r_d isotropic",
                "reference": "DESI Collaboration 2024"
            }
        }

    # Look for text files with DV measurements (fallback)
    txt_files = list(raw_dir.glob("**/*.txt"))
    if txt_files:
        for txt_file in txt_files:
            try:
                # Check if this looks like isotropic BAO data (DV table)
                if "dv" in txt_file.name.lower() or "DV" in txt_file.name:
                    # Try to load as two-column data (z, DV/rd)
                    data = np.loadtxt(txt_file)
                    if data.shape[1] == 2:  # z, DV/rd
                        z, DV_rd = data[:, 0], data[:, 1]

                        # For now, assume errors are ~10% of DV/rd (typical for BAO)
                        DV_err = 0.1 * DV_rd

                        return {
                            "name": "BAO_DR16_ELG_iso",
                            "type": "BAO_ISO",
                            "z": z,
                            "obs": DV_rd,
                            "err": DV_err,
                            "cov": None,
                            "meta": {
                                "survey": "SDSS DR16 ELG",
                                "observable": "D_V(z)/r_d isotropic",
                                "reference": "Alam et al. 2021"
                            }
                        }
            except Exception as e:
                print(f"Failed to load isotropic BAO from {txt_file}: {e}")
                continue

    # Look for CSV files in derived directory
    csv_files = list(raw_dir.glob("**/*.csv"))
    if csv_files:
        for csv_file in csv_files:
            try:
                # Check if this is the isotropic CSV
                if "iso" in csv_file.name.lower() or "dv" in csv_file.name.lower():
                    df = pd.read_csv(csv_file)

                    if "DV_div_rd" in df.columns:
                        z = df["z"].values
                        DV_rd = df["DV_div_rd"].values
                        DV_err = df["sigma_DV_div_rd"].values

                        return {
                            "name": "BAO_DR16_iso",
                            "type": "BAO_ISO",
                            "z": z,
                            "obs": DV_rd,
                            "err": DV_err,
                            "cov": None,
                            "meta": {
                                "survey": "SDSS DR16",
                                "observable": "D_V(z)/r_d isotropic",
                                "reference": "Alam et al. 2021"
                            }
                        }
            except Exception as e:
                print(f"Failed to load isotropic BAO from {csv_file}: {e}")
                continue

    # Fallback to standard loader
    try:
        from data_interface.bao_loader import load_bao_iso_data
        return load_bao_iso_data()
    except:
        return None


def _load_bao_aniso_data(raw_dir: Path):
    """Load anisotropic BAO data (D_M/rd and D_H/rd measurements)."""
    # Look for DESI BAO mean files (proper anisotropic data)
    mean_files = list(raw_dir.glob("**/desi*mean*.txt"))
    all_z_values = []
    all_dm_values = []
    all_dh_values = []

    if mean_files:
        for mean_file in mean_files:
            try:
                # Check if this contains anisotropic BAO data
                if "gaussian" in mean_file.name.lower() and "bao" in mean_file.name.lower():
                    with open(mean_file, 'r') as f:
                        lines = f.readlines()

                    z_values = []
                    dm_values = []
                    dh_values = []
                    current_z = None

                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                z = float(parts[0])
                                value = float(parts[1])
                                quantity = parts[2]

                                if 'DM_over_rs' in quantity or 'DM/rd' in quantity:
                                    if current_z is None or z != current_z:
                                        z_values.append(z)
                                        current_z = z
                                    dm_values.append(value)
                                elif 'DH_over_rs' in quantity or 'H*rd/c' in quantity or 'DH/rd' in quantity:
                                    if current_z is None or z != current_z:
                                        z_values.append(z)
                                        current_z = z
                                    dh_values.append(value)
                            except ValueError:
                                continue

                    if z_values and dm_values and dh_values:
                        # Make sure we have matching pairs
                        min_len = min(len(z_values), len(dm_values), len(dh_values))
                        if min_len > 0:
                            z_values = z_values[:min_len]
                            dm_values = dm_values[:min_len]
                            dh_values = dh_values[:min_len]

                            all_z_values.extend(z_values)
                            all_dm_values.extend(dm_values)
                            all_dh_values.extend(dh_values)
            except Exception as e:
                print(f"Failed to load anisotropic BAO from {mean_file}: {e}")
                continue

    if all_z_values and all_dm_values and all_dh_values:
        # Sort by redshift
        sorted_indices = np.argsort(all_z_values)
        z_values = np.array([all_z_values[i] for i in sorted_indices])
        dm_values = np.array([all_dm_values[i] for i in sorted_indices])
        dh_values = np.array([all_dh_values[i] for i in sorted_indices])

        # Remove duplicates (same z, DM/rd, and DH/rd within tolerance)
        unique_indices = []
        tolerance = 1e-6
        for i in range(len(z_values)):
            is_duplicate = False
            for j in unique_indices:
                if (abs(z_values[i] - z_values[j]) < tolerance and
                    abs(dm_values[i] - dm_values[j]) < tolerance and
                    abs(dh_values[i] - dh_values[j]) < tolerance):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_indices.append(i)

        z_values = z_values[unique_indices]
        dm_values = dm_values[unique_indices]
        dh_values = dh_values[unique_indices]

        # Interleave measurements: [DM1, H1, DM2, H2, ...]
        obs_interleaved = []
        err_interleaved = []
        for i in range(len(z_values)):
            obs_interleaved.extend([dm_values[i], dh_values[i]])
            # Assume 5% errors for both measurements
            err_interleaved.extend([0.05 * dm_values[i], 0.05 * dh_values[i]])

        return {
            "name": "DESI_BAO_aniso",
            "type": "BAO_ANISO",
            "z": z_values,
            "obs": np.array(obs_interleaved),
            "err": np.array(err_interleaved),
            "cov": None,
            "meta": {
                "survey": "DESI",
                "observable": "D_M(z)/r_d and D_H(z)/r_d",
                "reference": "DESI Collaboration 2024"
            }
        }

    # Look for CSV files in derived directory (fallback)
    csv_files = list(raw_dir.glob("**/*.csv"))
    if csv_files:
        for csv_file in csv_files:
            try:
                # Check if this is the anisotropic CSV
                if "aniso" in csv_file.name.lower() or ("dm" in csv_file.name.lower() and "hz" in csv_file.name.lower()):
                    df = pd.read_csv(csv_file)

                    if "DM_over_rd" in df.columns and (
                        "DH_over_rd" in df.columns or "Hz_rd_over_c" in df.columns
                    ):
                        z = df["z"].values

                        # Transverse clustering scale: D_M(z)/r_d
                        DM_rd = df["DM_over_rd"].values
                        DM_err = df["sigma_DM_over_rd"].values

                        # Radial clustering scale: D_H(z)/r_d
                        if "DH_over_rd" in df.columns:
                            DH_rd = df["DH_over_rd"].values
                            if "sigma_DH_over_rd" in df.columns:
                                DH_err = df["sigma_DH_over_rd"].values
                            elif "sigma_Hz_rd_over_c" in df.columns and "Hz_rd_over_c" in df.columns:
                                Hz_rd = df["Hz_rd_over_c"].values
                                Hz_err = df["sigma_Hz_rd_over_c"].values
                                DH_err = Hz_err / (Hz_rd**2)
                            else:
                                raise ValueError("Missing DH uncertainties for BAO_ANISO conversion.")
                        else:
                            Hz_rd = df["Hz_rd_over_c"].values
                            Hz_err = df["sigma_Hz_rd_over_c"].values
                            DH_rd = 1.0 / Hz_rd
                            DH_err = Hz_err / (Hz_rd**2)

                        # Interleave measurements: [DM1, H1, DM2, H2, DM3, H3]
                        obs_interleaved = np.empty((2 * len(z),), dtype=float)
                        obs_interleaved[0::2] = DM_rd  # Even indices: DM measurements
                        obs_interleaved[1::2] = DH_rd  # Odd indices: D_H measurements

                        # Interleave errors: [DM_err1, H_err1, DM_err2, H_err2, DM_err3, H_err3]
                        err_interleaved = np.empty((2 * len(z),), dtype=float)
                        err_interleaved[0::2] = DM_err  # Even indices: DM errors
                        err_interleaved[1::2] = DH_err  # Odd indices: D_H errors

                        return {
                            "name": "BAO_DR16_aniso",
                            "type": "BAO_ANISO",
                            "z": z,
                            "obs": obs_interleaved,
                            "err": err_interleaved,
                            "cov": None,
                            "meta": {
                                "survey": "SDSS DR16",
                                "observable": "D_M(z)/r_d and D_H(z)/r_d",
                                "reference": "Alam et al. 2021"
                            }
                        }
            except Exception as e:
                print(f"Failed to load anisotropic BAO from {csv_file}: {e}")
                continue

    # Fallback to standard loader
    try:
        from data_interface.bao_loader import load_bao_data
        return load_bao_data()
    except:
        return None


def _load_cc_data(raw_dir: Path):
    """Load cosmic chronometer data."""
    txt_files = list(raw_dir.glob("*.txt"))
    if txt_files:
        for txt_file in txt_files:
            try:
                data = np.loadtxt(txt_file, comments="#")
                if data.shape[1] >= 3:  # z, H, H_err
                    z, H, H_err = data[:, 0], data[:, 1], data[:, 2]
                    return {
                        "z": z,
                        "obs": H,
                        "err": H_err,
                        "cov": None
                    }
            except:
                continue

    # Fallback
    try:
        from data_interface.cc_loader import load_cc_data
        return load_cc_data()
    except:
        return None


def _load_shoes_data(raw_dir: Path):
    """Load SH0ES data."""
    # Look for lstsq_results.txt
    txt_files = list(raw_dir.glob("**/*.txt"))
    if txt_files:
        for txt_file in txt_files:
            if "lstsq_results" in txt_file.name:
                try:
                    q, sigma = np.loadtxt(txt_file, unpack=True)
                    # Last parameter is 5*log10(H0)
                    H0_param = q[-1]
                    H0_sigma = sigma[-1]
                    H0 = 10**(H0_param / 5)
                    # Error propagation: dH0 / d(param) = H0 * (1/param) * 5 / ln(10)
                    H0_err = H0 * (H0_sigma / H0_param) * 5 / np.log(10)
                    return {
                        "name": "SH0ES",
                        "z": np.array([0.0]),
                        "obs": np.array([H0]),
                        "err": np.array([H0_err]),
                        "cov": None,
                        "type": "CC"
                    }
                except Exception as e:
                    print(f"Failed to load {txt_file}: {e}")
                    continue

    # Fallback
    return None


def _load_rsd_data(raw_dir: Path):
    """Load RSD data."""
    txt_files = list(raw_dir.glob("**/*.txt"))
    if txt_files:
        for txt_file in txt_files:
            try:
                data = np.loadtxt(txt_file, comments="#")
                if data.shape[1] >= 3:  # z, fsigma8, err
                    z, fsigma8, err = data[:, 0], data[:, 1], data[:, 2]
                    return {
                        "z": z,
                        "obs": fsigma8,
                        "err": err,
                        "cov": None
                    }
            except:
                continue

    # Fallback
    try:
        from data_interface.rsd_loader import load_rsd_data
        return load_rsd_data()
    except:
        return None


def _convert_to_npz_format(standard_data: dict, source: str) -> dict:
    """Convert standard format to CLI .npz specification."""
    npz_dict = {}

    # Basic info
    npz_dict["name"] = standard_data.get("name", source)
    npz_dict["labels"] = _get_labels_for_dataset(standard_data)
    npz_dict["n_data"] = len(standard_data["obs"]) if standard_data["obs"] is not None else 1
    npz_dict["meta"] = standard_data.get("meta", {})

    # Observables and redshift
    npz_dict["obs"] = np.asarray(standard_data["obs"], dtype=float)

    if standard_data.get("z") is not None:
        if standard_data["type"] == "CMB":
            # CMB has single redshift
            npz_dict["z"] = float(standard_data["z"]) if standard_data["z"] is not None else 1089.92
        else:
            # Other datasets have arrays
            npz_dict["z"] = np.asarray(standard_data["z"], dtype=float)
    else:
        npz_dict["z"] = 1089.92 if standard_data.get("type") == "CMB" else None

    # Covariance matrix
    if standard_data.get("cov") is not None:
        npz_dict["cov"] = np.asarray(standard_data["cov"], dtype=float)
    elif standard_data.get("err") is not None:
        # Create diagonal covariance from errors
        err = np.asarray(standard_data["err"], dtype=float)
        npz_dict["cov"] = np.diag(err**2)
    else:
        # Identity matrix as fallback
        n = npz_dict["n_data"]
        npz_dict["cov"] = np.eye(n)

    # Add summary info to metadata
    if npz_dict["z"] is not None:
        if np.isscalar(npz_dict["z"]):
            npz_dict["meta"]["z"] = npz_dict["z"]
        else:
            npz_dict["meta"]["z_min"] = float(np.min(npz_dict["z"]))
            npz_dict["meta"]["z_max"] = float(np.max(npz_dict["z"]))
            npz_dict["meta"]["z_mean"] = float(np.mean(npz_dict["z"]))

    npz_dict["meta"]["dataset_type"] = standard_data.get("type", "unknown")
    npz_dict["meta"]["created_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    return npz_dict


def load_standardized_dataset(npz_path: str) -> dict:
    """Load a standardized .npz dataset."""
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python converter.py <source> <output.npz>")
        sys.exit(1)

    source, output = sys.argv[1], sys.argv[2]
    result = convert_dataset(source, output)
    print(f"\n✅ Conversion completed successfully!")
    print(f"   Output: {output}")
    print(f"   Keys: {list(result.keys())}")
