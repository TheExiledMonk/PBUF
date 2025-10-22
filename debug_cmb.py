#!/usr/bin/env python3
"""
Debug script for CMB distance priors.
"""

import sys
import numpy as np
sys.path.append('pipelines')

from fit_core.parameter import build_params
from fit_core.cmb_priors import prepare_background_params, distance_priors, _angular_diameter_distance

def debug_cmb_calculations():
    """Debug CMB distance prior calculations."""
    print("Debugging CMB calculations...")
    
    # Build test parameters
    params = build_params("lcdm")
    print(f"Base parameters: H0={params['H0']}, Om0={params['Om0']}, Obh2={params['Obh2']}")
    
    # Prepare background parameters
    params = prepare_background_params(params)
    print(f"Derived parameters:")
    print(f"  z_recomb = {params['z_recomb']}")
    print(f"  z_drag = {params['z_drag']}")
    print(f"  r_s_drag = {params['r_s_drag']}")
    
    # Check angular diameter distance
    D_A = _angular_diameter_distance(params['z_recomb'], params['H0'], params['Om0'])
    print(f"  D_A(z_recomb) = {D_A} Mpc")
    
    # Compute distance priors step by step
    H0 = params["H0"]
    Om0 = params["Om0"]
    z_recomb = params["z_recomb"]
    r_s_drag = params["r_s_drag"]
    
    print(f"\nDistance prior calculations:")
    print(f"  sqrt(Om0) = {np.sqrt(Om0)}")
    print(f"  H0 * D_A = {H0 * D_A}")
    print(f"  c = 299792.458 km/s")
    
    # Compute shift parameter R
    R = np.sqrt(Om0) * H0 * D_A / 299792.458
    print(f"  R = sqrt(Om0) * H0 * D_A / c = {R}")
    
    # Compute acoustic scale l_A
    l_A = np.pi * D_A / r_s_drag
    print(f"  l_A = pi * D_A / r_s = {l_A}")
    
    # Compute angular scale theta*
    theta_star = r_s_drag / D_A
    print(f"  theta* = r_s / D_A = {theta_star}")
    
    # Compare with expected Planck 2018 values
    print(f"\nComparison with Planck 2018:")
    print(f"  Expected R = 1.7502, computed = {R}")
    print(f"  Expected l_A = 301.845, computed = {l_A}")
    print(f"  Expected theta* = 1.04092, computed = {theta_star}")
    
    # Check if theta* should be in different units
    theta_star_100 = theta_star * 100
    print(f"  theta* * 100 = {theta_star_100}")

if __name__ == "__main__":
    debug_cmb_calculations()