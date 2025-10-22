#!/usr/bin/env python3
"""
Debug BAO calculations.
"""

import sys
import numpy as np
sys.path.append('pipelines')

from fit_core.parameter import build_params

def debug_bao_calculations():
    """Debug BAO distance calculations."""
    print("Debugging BAO calculations...")
    
    # Build test parameters
    params = build_params("lcdm")
    H0 = params["H0"]
    Om0 = params["Om0"]
    r_s_drag = params["r_s_drag"]
    
    print(f"Parameters: H0={H0}, Om0={Om0}, r_s_drag={r_s_drag}")
    
    # Test redshift
    z = 0.38
    
    # Compute angular diameter distance
    c_over_H0 = 299792.458 / H0  # Hubble distance in Mpc
    Ol0 = 1 - Om0
    
    # Simple integration
    n_points = 100
    z_array = np.linspace(0, z, n_points)
    dz = z / (n_points - 1)
    E_inv = 1 / np.sqrt(Om0 * (1 + z_array)**3 + Ol0)
    D_C = c_over_H0 * np.trapz(E_inv, dx=dz)
    D_A = D_C / (1 + z)
    
    print(f"\nAt z = {z}:")
    print(f"  D_A = {D_A} Mpc")
    
    # Comoving angular diameter distance
    D_M = (1 + z) * D_A
    print(f"  D_M = (1+z) * D_A = {D_M} Mpc")
    
    # D_M/r_s ratio
    dm_over_rs = D_M / r_s_drag
    print(f"  D_M/r_s = {dm_over_rs}")
    print(f"  Expected D_M/r_s ≈ 10.23")
    
    # Hubble parameter
    H_z = H0 * np.sqrt(Om0 * (1 + z)**3 + Ol0)
    print(f"  H(z) = {H_z} km/s/Mpc")
    
    # H*r_s
    h_times_rs = H_z * r_s_drag
    print(f"  H*r_s = {h_times_rs} km/s")
    print(f"  Expected H*r_s ≈ 81.2 km/s")
    
    # Check if units are wrong
    print(f"\nUnit check:")
    print(f"  r_s_drag in Mpc: {r_s_drag}")
    print(f"  H(z) in km/s/Mpc: {H_z}")
    print(f"  H*r_s should be dimensionless or in km/s")
    print(f"  Current H*r_s = {h_times_rs} (too large)")
    
    # Maybe the expected values are H(z)*r_s in different units?
    # Let's try H(z) in units of 100 km/s/Mpc
    h_z_100 = H_z / 100
    h_times_rs_100 = h_z_100 * r_s_drag
    print(f"  H(z)/100 * r_s = {h_times_rs_100}")

if __name__ == "__main__":
    debug_bao_calculations()