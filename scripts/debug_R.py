#!/usr/bin/env python3
"""
Debug the R calculation specifically.
"""

import numpy as np

def debug_R_calculation():
    """Debug the shift parameter R calculation."""
    
    # Known values
    R_obs = 1.7502
    H0 = 67.4
    Om0 = 0.315
    z_recomb = 1089.8
    D_A = 12.774449300775805
    c = 299792.458
    
    print("Debugging shift parameter R...")
    print(f"Target R = {R_obs}")
    print(f"Parameters: H0={H0}, Om0={Om0}, z*={z_recomb}")
    print(f"D_A = {D_A} Mpc")
    
    # My calculation
    R_mine = np.sqrt(Om0) * H0 * (1 + z_recomb) * D_A / c
    print(f"\nMy calculation: R = sqrt(Om0) * H0 * (1+z*) * D_A / c")
    print(f"R = {np.sqrt(Om0)} * {H0} * {1 + z_recomb} * {D_A} / {c}")
    print(f"R = {R_mine}")
    print(f"Ratio to observed: {R_mine / R_obs}")
    
    # Try without (1+z) factor
    R_no_z = np.sqrt(Om0) * H0 * D_A / c
    print(f"\nWithout (1+z): R = sqrt(Om0) * H0 * D_A / c = {R_no_z}")
    print(f"Ratio to observed: {R_no_z / R_obs}")
    
    # Try different D_A
    D_A_needed = R_obs * c / (np.sqrt(Om0) * H0 * (1 + z_recomb))
    print(f"\nD_A needed for correct R: {D_A_needed} Mpc")
    print(f"Ratio to my D_A: {D_A_needed / D_A}")
    
    # Check if the issue is units
    # Maybe D_A should be in different units?
    D_A_kpc = D_A * 1000  # Convert to kpc
    R_kpc = np.sqrt(Om0) * H0 * (1 + z_recomb) * D_A_kpc / c
    print(f"\nWith D_A in kpc: R = {R_kpc}")
    print(f"Ratio to observed: {R_kpc / R_obs}")

if __name__ == "__main__":
    debug_R_calculation()