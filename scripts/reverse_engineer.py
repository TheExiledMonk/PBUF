#!/usr/bin/env python3
"""
Reverse engineer the correct CMB distance prior calculations.
"""

import numpy as np

def reverse_engineer_cmb():
    """Reverse engineer from known Planck 2018 values."""
    
    # Known Planck 2018 values
    R_obs = 1.7502
    l_A_obs = 301.845
    theta_star_obs = 1.04092  # This might be 100*theta_star
    
    # Standard parameters
    H0 = 67.4
    Om0 = 0.315
    z_recomb = 1089.8
    
    print("Reverse engineering CMB distance priors...")
    print(f"Observed values: R={R_obs}, l_A={l_A_obs}, theta*={theta_star_obs}")
    
    # My computed D_A
    D_A = 12.774449300775805  # From previous calculation
    print(f"Computed D_A = {D_A} Mpc")
    
    # Reverse engineer sound horizon from l_A
    # l_A = π * D_A / r_s  =>  r_s = π * D_A / l_A
    r_s_from_lA = np.pi * D_A / l_A_obs
    print(f"Sound horizon from l_A: r_s = π * D_A / l_A = {r_s_from_lA} Mpc")
    
    # Reverse engineer sound horizon from theta*
    # theta* = r_s / D_A  =>  r_s = theta* * D_A
    r_s_from_theta = theta_star_obs * D_A
    print(f"Sound horizon from theta*: r_s = theta* * D_A = {r_s_from_theta} Mpc")
    
    # Check if theta* is actually 100*theta_star
    theta_star_radians = theta_star_obs / 100
    r_s_from_theta_100 = theta_star_radians * D_A
    print(f"Sound horizon from theta*/100: r_s = (theta*/100) * D_A = {r_s_from_theta_100} Mpc")
    
    # Reverse engineer D_A from R
    # R = sqrt(Om0) * H0 * (1+z*) * D_A / c  =>  D_A = R * c / (sqrt(Om0) * H0 * (1+z*))
    c = 299792.458
    D_A_from_R = R_obs * c / (np.sqrt(Om0) * H0 * (1 + z_recomb))
    print(f"D_A from R: D_A = R * c / (sqrt(Om0) * H0 * (1+z*)) = {D_A_from_R} Mpc")
    
    # Check consistency
    print(f"\nConsistency check:")
    print(f"My D_A: {D_A} Mpc")
    print(f"D_A from R: {D_A_from_R} Mpc")
    print(f"Ratio: {D_A / D_A_from_R}")
    
    # Check if the issue is in the R formula
    # Maybe R = sqrt(Om0) * H0 * D_A / c (without the 1+z factor)
    R_without_1z = np.sqrt(Om0) * H0 * D_A / c
    print(f"\nR without (1+z) factor: {R_without_1z}")
    print(f"Observed R: {R_obs}")
    print(f"Ratio: {R_without_1z / R_obs}")

if __name__ == "__main__":
    reverse_engineer_cmb()