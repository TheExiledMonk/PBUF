#!/usr/bin/env python3
"""
Create proper mock datasets for Phase A validation testing.

This script creates mock datasets in the correct formats expected by
each derivation module for comprehensive testing.
"""

import json
import csv
from pathlib import Path


def create_cmb_mock_data():
    """Create mock CMB data in the expected format."""
    mock_data_dir = Path("data/mock_phase_a")
    mock_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CMB data with uncertainties
    cmb_data = {
        "R": 1.7502,
        "R_err": 0.0023,
        "l_A": 301.63,
        "l_A_err": 0.15,
        "theta_star": 1.04119,
        "theta_star_err": 0.00030,
        "covariance": [
            [0.0001, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.000001]
        ]
    }
    
    cmb_file = mock_data_dir / "cmb_planck2018_mock.json"
    with open(cmb_file, 'w') as f:
        json.dump(cmb_data, f, indent=2)
    
    print(f"Created CMB mock data: {cmb_file}")


def create_sn_mock_data():
    """Create mock SN data in CSV format."""
    mock_data_dir = Path("data/mock_phase_a")
    mock_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SN data in CSV format (expected by SN module)
    sn_file = mock_data_dir / "sn_pantheon_plus_mock.csv"
    
    with open(sn_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["CID", "zHD", "zHDERR", "zCMB", "zCMBERR", "MU", "MUERR", "MURES", "MUPULL"])
        
        # Write data rows
        for i in range(100):  # Smaller sample for testing
            z = 0.01 + i * 0.02
            mu = 32.0 + 5 * z  # Approximate distance modulus
            writer.writerow([
                f"SN{i:04d}",
                z,
                0.001,
                z,
                0.001,
                mu,
                0.1,
                0.0,
                0.0
            ])
    
    print(f"Created SN mock data: {sn_file}")


def create_bao_mock_data():
    """Create mock BAO data in CSV format."""
    mock_data_dir = Path("data/mock_phase_a")
    mock_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create BAO data in CSV format
    bao_data = [
        {"z_eff": 0.38, "DM_over_rd": 10.3, "DM_err": 0.4, "DH_over_rd": 25.2, "DH_err": 0.7, "correlation": -0.4},
        {"z_eff": 0.51, "DM_over_rd": 13.7, "DM_err": 0.4, "DH_over_rd": 22.3, "DH_err": 0.5, "correlation": -0.4},
        {"z_eff": 0.61, "DM_over_rd": 16.1, "DM_err": 0.3, "DH_over_rd": 20.9, "DH_err": 0.4, "correlation": -0.4}
    ]
    
    # Save as CSV
    bao_file = mock_data_dir / "bao_compilation_mock.csv"
    with open(bao_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["z_eff", "DM_over_rd", "DM_err", "DH_over_rd", "DH_err", "correlation"])
        writer.writeheader()
        writer.writerows(bao_data)
    
    # Create anisotropic BAO data (same format)
    bao_aniso_file = mock_data_dir / "bao_aniso_boss_dr12_mock.csv"
    with open(bao_aniso_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["z_eff", "DM_over_rd", "DM_err", "DH_over_rd", "DH_err", "correlation"])
        writer.writeheader()
        writer.writerows(bao_data)
    
    print(f"Created BAO mock data: {bao_file}")
    print(f"Created BAO aniso mock data: {bao_aniso_file}")


def main():
    """Create all mock datasets."""
    print("Creating proper mock datasets for Phase A validation...")
    
    create_cmb_mock_data()
    create_sn_mock_data()
    create_bao_mock_data()
    
    print("All mock datasets created successfully!")


if __name__ == "__main__":
    main()