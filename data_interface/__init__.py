"""
Unified dataset gateway for cosmological fits.

This module provides a single entry point to load all datasets
(CMB, SN, BAO, CC, RSD) either as synthetic test data or as real
observational data from the /data_interface/datasets/ directory.

Usage:
    from data_interface import load_all_datasets
    datasets = load_all_datasets(use_real=True)

    # For standardized format:
    from data_interface import standardize_all_datasets
    std_datasets = standardize_all_datasets(datasets)

Each dataset is returned as a dictionary ready for χ² evaluation:
    {
        "z": np.ndarray,
        "obs": np.ndarray,
        "err": np.ndarray,
        "cov": np.ndarray | None,
        ...
    }
"""

import numpy as np

# Import all loaders and standardization utilities
from .cmb_loader import load_cmb_priors
from .sn_loader import load_sn_data
from .bao_loader import load_bao_data, load_bao_iso_data
from .cc_loader import load_cc_data
from .rsd_loader import load_rsd_data
from .standardize import (
    ensure_standard_dataset,
    describe_dataset,
    convert_bao_to_standard,
    convert_sn_to_standard,
    convert_cmb_to_standard,
    convert_cc_to_standard,
    convert_rsd_to_standard,
    standardize_all_datasets
)


# -------------------------------------------------------------------
# Unified gateway
# -------------------------------------------------------------------
def load_all_datasets(use_real: bool = False, standardize: bool = False):
    """
    Load all cosmological datasets (CMB, SN, BAO, CC, RSD).

    Parameters
    ----------
    use_real : bool
        If True, load from /datasets/ (real data).
        If False, generate synthetic baseline data for internal tests.
    standardize : bool
        If True, return datasets in standardized PBUF format.

    Returns
    -------
    dict
        {
          "cmb": {...},
          "sn": {...},
          "bao": {...},
          "cc": {...},
          "rsd": {...}
        }
    """
    if not use_real:
        print("🔬 Using synthetic datasets for internal validation...")
        datasets = _load_synthetic()
    else:
        print("📡 Loading real datasets from data/standardized/")
        datasets = _load_standardized_datasets()

    if standardize:
        print("📐 Skipping standardization - data already in standardized format")
        # No need to standardize - already in correct format

    return datasets


# -------------------------------------------------------------------
# Standardized data loader (direct from data/standardized/)
# -------------------------------------------------------------------
def _load_standardized_datasets():
    """Load pre-standardized datasets from data/standardized/ directory."""
    import numpy as np
    from pathlib import Path
    
    # Map dataset names to standardized files
    dataset_files = {
        "cmb": "cmb.npz",
        "sn": "sn_pantheon.npz", 
        "bao_iso": "bao_iso.npz",
        "bao_aniso": "bao_aniso.npz",
        "cc": "cc.npz",
        "rsd": "rsd.npz"
    }
    
    standardized_dir = Path(__file__).resolve().parent.parent / "data" / "standardized"
    datasets = {}
    
    for dataset_name, filename in dataset_files.items():
        dataset_path = standardized_dir / filename
        if dataset_path.exists():
            try:
                with np.load(dataset_path, allow_pickle=True) as npz:
                    # Convert numpy arrays to regular python structures
                    dataset_data = {}
                    for key, value in npz.items():
                        if key == "meta":
                            # Handle metadata specially
                            if isinstance(value, np.ndarray):
                                if value.shape == ():
                                    dataset_data[key] = dict(value.item())
                                elif value.size == 1:
                                    dataset_data[key] = dict(value.reshape(-1)[0])
                                else:
                                    dataset_data[key] = value.tolist()
                            else:
                                dataset_data[key] = value
                        else:
                            # Convert numpy arrays to lists for better JSON serialization
                            dataset_data[key] = value.tolist() if isinstance(value, np.ndarray) else value
                    
                    datasets[dataset_name] = dataset_data
                    print(f"[data_interface] Loaded {dataset_name}: {len(dataset_data.get('z', []))} points")
                    
            except Exception as e:
                print(f"[data_interface] Warning: Could not load {dataset_name} from {filename}: {e}")
        else:
            print(f"[data_interface] Warning: Dataset file not found: {dataset_path}")
    
    # Handle BAO datasets - combine bao_iso and bao_aniso into "bao" for compatibility
    if "bao_iso" in datasets and "bao_aniso" in datasets:
        datasets["bao"] = {
            "iso": datasets["bao_iso"],
            "aniso": datasets["bao_aniso"]
        }
    elif "bao_iso" in datasets:
        datasets["bao"] = datasets["bao_iso"]
    elif "bao_aniso" in datasets:
        datasets["bao"] = datasets["bao_aniso"]
    
    if not datasets:
        print("[data_interface] Warning: No standardized datasets loaded, falling back to synthetic")
        return _load_synthetic()
    
    print(f"[data_interface] Loaded {len(datasets)} standardized datasets")
    return datasets


# -------------------------------------------------------------------
# Synthetic placeholders (same scale as your current test suite)
# -------------------------------------------------------------------
def _load_synthetic():
    """Return internal mock datasets matching unit tests."""
    # CMB data (legacy format for conversion)
    cmb = {
        "mean": np.array([1.7532, 78.73, 0.0399]),
        "cov": np.eye(3) * 1e-4,
        "labels": ["R", "la", "theta_star"],
    }

    # SN data (already in standard format, but add type/name for conversion)
    sn = {
        "name": "Pantheon+",
        "type": "SN",
        "z": np.array([0.01, 0.1, 0.5, 1.0, 2.0]),
        "obs": np.array([33.2572, 38.3949, 42.3321, 44.1632, 46.0117]),
        "err": np.full(5, 0.1),
        "cov": None,
    }

    # BAO data (legacy format for conversion)
    bao = {
        "z": np.array([0.38, 0.51, 0.61]),
        "DV_rd": np.array([7.84, 10.01, 11.52]),
        "DM_rd": np.array([8.13, 10.53, 12.25]),
        "Hz_rd": np.array([5.2e-5, 5.6e-5, 6.0e-5]),
        "err": np.full(3, 0.1),
        "cov": None,
    }

    # CC data (legacy format for conversion)
    cc = {
        "z": np.array([0.07, 0.12, 0.20, 0.35, 0.60, 1.0, 1.5, 2.0]),
        "obs": np.array([69.75, 71.57, 74.73, 81.44, 94.73, 120.68, 159.62, 204.37]),
        "err": np.full(8, 3.0),
        "cov": None,
    }

    # RSD data (legacy format for conversion)
    rsd = {
        "z": np.array([0.15, 0.38, 0.51, 0.61, 1.0]),
        "obs": np.array([0.4727, 0.5067, 0.5129, 0.5132, 0.4935]),
        "err": np.full(5, 0.05),
        "cov": None,
    }

    return {"cmb": cmb, "sn": sn, "bao": bao, "cc": cc, "rsd": rsd}


# -------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Legacy Format ===")
    data = load_all_datasets(use_real=False, standardize=False)
    print("✅ Legacy dataset loaded successfully.")
    for key, val in data.items():
        print(f"  {key}: keys = {list(val.keys())}")

    print("\n=== Standardized Format ===")
    data_std = load_all_datasets(use_real=False, standardize=True)
    print("✅ Standardized dataset loaded successfully.")
    for key, val in data_std.items():
        print(f"  {key}: keys = {list(val.keys())}")

    print("\n=== Individual Standardization Tests ===")
    from data_interface.standardize import (
        describe_dataset,
        convert_bao_to_standard,
        convert_sn_to_standard,
        convert_cmb_to_standard
    )

    # Test individual converters
    print("BAO (DM+Hz average):")
    bao_std = convert_bao_to_standard(data["bao"], "combined")
    describe_dataset(bao_std)

    print("\nSN:")
    sn_std = convert_sn_to_standard(data["sn"])
    describe_dataset(sn_std)

    print("\nCMB:")
    cmb_std = convert_cmb_to_standard(data["cmb"])
    describe_dataset(cmb_std)
