#!/usr/bin/env python3
"""
Debug angular diameter distance calculation.
"""

import numpy as np

def debug_angular_diameter_distance():
    """Debug the angular diameter distance calculation."""
    
    # Standard Planck 2018 parameters
    H0 = 67.4  # km/s/Mpc
    Om0 = 0.315
    z = 1089.8
    
    print(f"Parameters: H0={H0}, Om0={Om0}, z={z}")
    
    # Hubble distance
    c = 299792.458  # km/s
    D_H = c / H0  # Mpc
    print(f"Hubble distance D_H = c/H0 = {D_H} Mpc")
    
    # Dark energy density
    Ol0 = 1 - Om0
    print(f"Dark energy density Ol0 = {Ol0}")
    
    # Test E(z) at a few points
    z_test = [0, 1, 10, 100, 1000]
    for z_val in z_test:
        E_z = np.sqrt(Om0 * (1 + z_val)**3 + Ol0)
        print(f"E({z_val}) = {E_z}")
    
    # Numerical integration
    n_points = 10000
    z_array = np.linspace(0, z, n_points)
    dz = z / (n_points - 1)
    
    E_inv = 1 / np.sqrt(Om0 * (1 + z_array)**3 + Ol0)
    
    print(f"Integration bounds: 0 to {z}")
    print(f"dz = {dz}")
    print(f"E_inv at z=0: {E_inv[0]}")
    print(f"E_inv at z=z_max: {E_inv[-1]}")
    
    # Trapezoidal integration
    integral = np.trapz(E_inv, dx=dz)
    print(f"Integral of 1/E(z) dz = {integral}")
    
    # Comoving distance
    D_C = D_H * integral
    print(f"Comoving distance D_C = D_H * integral = {D_C} Mpc")
    
    # Angular diameter distance
    D_A = D_C / (1 + z)
    print(f"Angular diameter distance D_A = D_C/(1+z) = {D_A} Mpc")
    
    # Expected value check
    print(f"\nExpected D_A for Planck cosmology at z~1090: ~14000 Mpc")
    print(f"Computed D_A: {D_A} Mpc")
    print(f"Ratio: {D_A / 14000}")

if __name__ == "__main__":
    debug_angular_diameter_distance()