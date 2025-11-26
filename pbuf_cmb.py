#!/usr/bin/env python3
import numpy as np
from numpy.linalg import inv
from math import sqrt, pi

###############################################################################
# SELF-CONTAINED PBUF CMB CHECK (THERMAL OFF)
###############################################################################

# -------------------------------
# 1. Planck 2018 distance priors
# -------------------------------
PLANCK_MEAN = {
    "R": 1.7488,
    "la": 301.76,
    "theta_star": 0.010411,
}

PLANCK_COV = np.array([
    [5.476e-05, -5.56332e-04, -9.717384e-09],
    [-5.56332e-04, 1.96e-02,   3.50238e-07],
    [-9.717384e-09, 3.50238e-07, 9.61e-12],
], dtype=float)

PLANCK_INV = inv(PLANCK_COV)

# -------------------------------
# 2. PBUF parameters (example)
# -------------------------------
H0   = 68.8        # km/s/Mpc
h    = H0 / 100.0
Om0  = 0.24
Ob0  = 0.0228
Ok0  = 0.0224
Or0  = 9.0e-5           # photons + neutrinos
alpha = 0.0224
Rmax  = 5.87e7
k_sat = 0.973891

# CMB redshift (use Planck central value)
z_star = 1089.92
a_star = 1.0 / (1.0 + z_star)

# Photon density from T_CMB ≈ 2.7255 K:
# Omega_gamma h^2 ≈ 2.469e-5 (T/2.7255)^4
T_CMB = 2.7255
Omega_gamma_h2 = 2.469e-5 * (T_CMB / 2.7255)**4
Og0 = Omega_gamma_h2 / (h**2)

# -------------------------------
# 3. Scale-factor grid
# -------------------------------
N = 20000
a_min = 1.0e-6
a_grid = np.logspace(np.log10(a_min), 0.0, N)

# masks for integrals
mask_star = a_grid >= a_star

# -------------------------------
# 4. Elastic Omega_sigma(a)
# -------------------------------
def omega_sigma(a):
    # Example elastic law; replace with your actual Ωσ(a) if needed
    return alpha * (1.0 - np.exp(-a * Rmax)) + k_sat * a

Omega_sigma = omega_sigma(a_grid)

# -------------------------------
# 5. Background E(a) and H(a)
# -------------------------------
def E_of_a(a, Om0, Or0, Ok0, Omega_sigma_arr):
    Om = Om0 / a**3
    Or = Or0 / a**4
    Ok = Ok0 / a**2
    Os = Omega_sigma_arr
    return np.sqrt(Om + Or + Ok + Os)

E_grid = E_of_a(a_grid, Om0, Or0, Ok0, Omega_sigma)

# H(a) = H0 * E(a)
# (we only need E(a) for distance integrals)


# -------------------------------
# 6. Sound horizon r_s
# -------------------------------
c = 299792.458  # km/s

def compute_rs(a_grid, E_grid, Ob0, Og0, a_star):
    """
    Compute sound horizon r_s up to a_drag ≈ a_star.
    Use photon-only Og0 in the baryon-photon ratio.
    """
    # upper limit ~ recombination / drag epoch
    a_drag = a_star

    # restrict to [a_min, a_drag]
    mask = a_grid <= a_drag
    a = a_grid[mask]
    E = E_grid[mask]

    # baryon-photon ratio R_b(a) = 3 ρ_b / (4 ρ_gamma) ∝ a
    R_b = 3.0 * Ob0 / (4.0 * Og0) * a

    # sound speed c_s(a)
    c_s = c / np.sqrt(3.0 * (1.0 + R_b))

    # integral r_s = (c/H0) ∫ c_s/c / (a^2 E(a)) da
    integrand = c_s / (a**2 * E)

    # distances in Mpc: c/H0 has units of Mpc
    return (1.0 / H0) * np.trapz(integrand, a)

r_s = compute_rs(a_grid, E_grid, Ob0, Og0, a_star)


# -------------------------------
# 7. Comoving distance D_M(z_star)
# -------------------------------
def comoving_distance_to_zstar(a_grid, E_grid, a_star):
    """
    D_M(z_star) = (c/H0) ∫_{a_star}^1 da / (a^2 E(a))
    """
    mask = a_grid >= a_star
    a = a_grid[mask]
    E = E_grid[mask]

    integrand = 1.0 / (a**2 * E)
    return (c / H0) * np.trapz(integrand, a)

D_M = comoving_distance_to_zstar(a_grid, E_grid, a_star)


# -------------------------------
# 8. Angular diameter & priors
# -------------------------------
D_A = D_M / (1.0 + z_star)

# Shift parameter R = sqrt(Ω_m) H0 D_M / c
R_cmb = np.sqrt(Om0) * (H0 * D_M / c)

# Acoustic scale l_A = π D_M / r_s
lA = pi * D_M / r_s

# θ* = r_s / D_A
theta = r_s / D_M


# -------------------------------
# 9. χ² vs Planck priors
# -------------------------------
def chi2_CMB(R, lA, theta):
    vec = np.array([R, lA, theta])
    mean = np.array([
        PLANCK_MEAN["R"],
        PLANCK_MEAN["la"],
        PLANCK_MEAN["theta_star"],
    ])
    diff = vec - mean
    return float(diff.T @ PLANCK_INV @ diff)

chi2_val = chi2_CMB(R_cmb, lA, theta)


###############################################################################
# OUTPUT
###############################################################################
print("\n=== SELF-CONTAINED PBUF CMB CHECK (THERMAL OFF) ===\n")
print(f"z_star       = {z_star}")
print(f"r_s          = {r_s:.6f}  # Mpc")
print(f"D_M(z*)      = {D_M:.6f}  # Mpc")
print(f"D_A(z*)      = {D_A:.6f}  # Mpc")
print(f"R            = {R_cmb:.6f}")
print(f"lA           = {lA:.6f}")
print(f"theta_star   = {theta:.9f}")
print(f"\nχ²_vs_Planck = {chi2_val:.6f}\n")

