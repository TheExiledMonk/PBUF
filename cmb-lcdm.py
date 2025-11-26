#!/usr/bin/env python3
"""
Standalone LCDM CMB distance-prior calculator with chi-square.

Computes:
- z_star (Hu & Sugiyama 1996)
- z_drag (Eisenstein & Hu 1998)
- Omega_r0 from T_CMB and N_eff
- r_s(z) via c_s/H(z) integral
- D_M(z_star), D_A(z_star)
- R, l_A, theta_star
- full chi-square vs Planck 2018 compressed priors

All distances in Mpc, H0 in km/s/Mpc.
"""

import numpy as np
from dataclasses import dataclass

C_LIGHT = 299_792.458
T_CMB_DEFAULT = 2.7255
N_EFF_DEFAULT = 3.046

# ------------------------------------------------------------
# Planck 2018 distance priors
# ------------------------------------------------------------
R_obs = 1.7488
lA_obs = 301.76
theta_obs = 0.01041099

obs_vector = np.array([R_obs, lA_obs, theta_obs])

# inverse covariance matrix (your cosmos2 values)
inv_cov = np.array([
    [ 2.56713313e+04,  7.59304286e+02, -1.71477946e+06],
    [ 7.59304286e+02,  1.68753290e+02, -5.38245199e+06],
    [-1.71477946e+06, -5.38245199e+06,  2.98488663e+11]
])


# ------------------------------------------------------------
@dataclass
class LCDMParams:
    H0: float
    Omega_m0: float
    Omega_b0: float
    Omega_k0: float
    T_cmb: float = T_CMB_DEFAULT
    N_eff: float = N_EFF_DEFAULT


@dataclass
class CMBDistances:
    z_star: float
    z_drag: float
    Omega_r0: float
    Omega_gamma0: float
    r_s_zstar: float
    r_s_zdrag: float
    D_M_zstar: float
    D_A_zstar: float
    R: float
    l_A: float
    theta_star: float
    chi2: float


# ------------------------------------------------------------
def omega_gamma0(h, T_cmb):
    return 2.469e-5 * (T_cmb / 2.7255)**4 / (h*h)


def omega_r0_from_Tcmb(H0, T_cmb, N_eff):
    h = H0 / 100.0
    og = omega_gamma0(h, T_cmb)
    orad = og * (1 + 0.2271 * N_eff)
    return orad, og


def z_star_hu_sugiyama(omega_b_h2, omega_m_h2):
    g1 = 0.0783 * omega_b_h2**-0.238 / (1 + 39.5 * omega_b_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * omega_b_h2**1.81)
    return 1048 * (1 + 0.00124 * omega_b_h2**-0.738) * (1 + g1 * omega_m_h2**g2)


def z_drag_eisenstein_hu(omega_b_h2, omega_m_h2):
    b1 = 0.313 * omega_m_h2**-0.419 * (1 + 0.607 * omega_m_h2**0.674)
    b2 = 0.238 * omega_m_h2**0.223
    return 1291*omega_m_h2**0.251/(1+0.659*omega_m_h2**0.828)*(1+b1*omega_b_h2**b2)


def E_of_z(z, p, Or):
    Om = p.Omega_m0
    Ok = p.Omega_k0
    Ol = 1 - Om - Or - Ok
    zp1 = 1 + z
    return np.sqrt(Om*zp1**3 + Or*zp1**4 + Ok*zp1**2 + Ol)


def comoving_distance(z, p, Or, n_steps=50000):
    z_grid = np.linspace(0, z, n_steps)
    integrand = 1/E_of_z(z_grid, p, Or)
    if (n_steps % 2) == 1:
        z_grid = z_grid[:-1]
        integrand = integrand[:-1]
    dz = z/(len(z_grid)-1)
    S = integrand[0] + integrand[-1] + 4*np.sum(integrand[1:-1:2]) + 2*np.sum(integrand[2:-2:2])
    return (C_LIGHT/p.H0) * (S*dz/3)


def transverse_comoving_distance(chi, p, Or):
    Ok = p.Omega_k0
    if abs(Ok) < 1e-8:
        return chi
    d_h = C_LIGHT / p.H0
    sqrt_ok = np.sqrt(abs(Ok))
    x = sqrt_ok * chi / d_h
    if Ok > 0:
        return d_h/sqrt_ok * np.sinh(x)
    else:
        return d_h/sqrt_ok * np.sin(x)


def sound_speed(z, Ob, Og):
    Rb = (3*Ob)/(4*Og) * 1/(1+z)
    return C_LIGHT / np.sqrt(3*(1+Rb))


def sound_horizon(z_start, p, Or, Og, z_max=1e5, n_steps=200000):
    z_grid = np.linspace(z_start, z_max, n_steps)
    E = E_of_z(z_grid, p, Or)
    H = p.H0 * E
    cs = sound_speed(z_grid, p.Omega_b0, Og)
    integrand = cs/H
    if (n_steps % 2) == 1:
        z_grid = z_grid[:-1]
        integrand = integrand[:-1]
    dz = (z_max - z_start)/(len(z_grid)-1)
    S = integrand[0] + integrand[-1] + 4*np.sum(integrand[1:-1:2]) + 2*np.sum(integrand[2:-2:2])
    return S*dz/3


# ------------------------------------------------------------
def compute_cmb_distances(p: LCDMParams) -> CMBDistances:

    h = p.H0/100
    obh2 = p.Omega_b0 * h*h
    omh2 = p.Omega_m0 * h*h

    Or, Og = omega_r0_from_Tcmb(p.H0, p.T_cmb, p.N_eff)

    z_star = z_star_hu_sugiyama(obh2, omh2)
    z_drag = z_drag_eisenstein_hu(obh2, omh2)

    chi = comoving_distance(z_star, p, Or)
    DM = transverse_comoving_distance(chi, p, Or)
    DA = DM / (1+z_star)

    rs_zstar = sound_horizon(z_star, p, Or, Og)
    rs_zdrag = sound_horizon(z_drag, p, Or, Og)

    R = np.sqrt(p.Omega_m0) * p.H0 * DM / C_LIGHT
    lA = np.pi * DM / rs_zstar
    theta = rs_zstar / DM

    # ---------------------------------------------------
    # Chi-square vs Planck priors
    pred = np.array([R, lA, theta])
    delta = pred - obs_vector
    chi2 = float(delta.T @ inv_cov @ delta)
    # ---------------------------------------------------

    return CMBDistances(z_star, z_drag, Or, Og,
                        rs_zstar, rs_zdrag, DM, DA,
                        R, lA, theta, chi2)


# ------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="LCDM CMB distance-prior calculator + chi-square")
    parser.add_argument("--H0", type=float, default=67.66)
    parser.add_argument("--Omega_m0", type=float, default=0.3111)
    parser.add_argument("--Omega_b0", type=float, default=0.049)
    parser.add_argument("--Omega_k0", type=float, default=0.0)
    parser.add_argument("--Tcmb", type=float, default=T_CMB_DEFAULT)
    parser.add_argument("--Neff", type=float, default=N_EFF_DEFAULT)
    args = parser.parse_args()

    p = LCDMParams(args.H0, args.Omega_m0, args.Omega_b0, args.Omega_k0, args.Tcmb, args.Neff)

    cmb = compute_cmb_distances(p)

    print("\nRESULTS FOR INPUT PARAMETERS")
    print(f"H0 = {p.H0}, Omega_m0 = {p.Omega_m0}, Omega_b0 = {p.Omega_b0}, Omega_k0 = {p.Omega_k0}\n")

    print("CMB priors:")
    print(f"  R            = {cmb.R:.6f}")
    print(f"  l_A          = {cmb.l_A:.6f}")
    print(f"  theta_star   = {cmb.theta_star:.8f}")

    print("\nDistances:")
    print(f"  D_M(z*) = {cmb.D_M_zstar:.3f} Mpc")
    print(f"  D_A(z*) = {cmb.D_A_zstar:.6f} Mpc")
    print(f"  r_s(z*) = {cmb.r_s_zstar:.3f} Mpc")
    print(f"  r_s(z_d) = {cmb.r_s_zdrag:.3f} Mpc")

    print("\nRedshifts:")
    print(f"  z_star = {cmb.z_star:.3f}")
    print(f"  z_drag = {cmb.z_drag:.3f}")

    print(f"\nChi-square vs Planck 2018 priors: χ² = {cmb.chi2:.2f}\n")


if __name__ == "__main__":
    main()

