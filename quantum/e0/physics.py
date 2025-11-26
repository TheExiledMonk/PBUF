"""
Physics module for spacetime rigidity calculations.

This module implements core physics calculations for special-relativistic
propagation and rigidity corrections. All calculations use SI base units
with energies in eV.
"""

import numpy as np
from typing import Optional


# Physical constants
c = 299792458.0  # Speed of light in vacuum (m/s)
MPC_TO_M = 3.085677581e22  # Conversion factor: megaparsecs to meters

# Numerical safety parameters
MAX_MASS_ENERGY_RATIO = 0.999999  # Maximum allowed m/E ratio to avoid numerical issues


def v_sr(m_eV: float, E_eV: Optional[float]) -> float:
    """
    Compute special-relativistic speed in m/s.
    
    For massless particles (m=0 or E=None), returns exactly c.
    For massive particles, uses the relativistic formula:
        v = c * sqrt(1 - (m/E)^2)
    
    Parameters
    ----------
    m_eV : float
        Rest mass in electron volts (eV)
    E_eV : float or None
        Total energy in electron volts (eV). None for photons.
    
    Returns
    -------
    float
        Propagation speed in meters per second (m/s)
    
    Raises
    ------
    ValueError
        If energy is less than or equal to mass, or if computation results in non-finite value
    
    Requirements: 2.1, 2.2, 10.1, 10.2, 10.3, 10.4, 11.3
    """
    try:
        # Validate inputs
        if not np.isfinite(m_eV) or m_eV < 0:
            raise ValueError(f"Invalid mass: {m_eV} eV (must be non-negative and finite)")
        
        # Photons or massless particles
        if m_eV == 0.0 or E_eV is None:
            return c
        
        if not np.isfinite(E_eV):
            raise ValueError(f"Invalid energy: {E_eV} eV (must be finite)")
        
        # Validate that energy exceeds mass for massive particles
        if E_eV <= m_eV:
            raise ValueError(f"Energy ({E_eV} eV) must be greater than rest mass ({m_eV} eV)")
        
        # Compute mass-to-energy ratio with safety clipping
        mass_energy_ratio = m_eV / E_eV
        mass_energy_ratio = min(mass_energy_ratio, MAX_MASS_ENERGY_RATIO)
        
        # Special-relativistic speed formula
        v = c * np.sqrt(1.0 - mass_energy_ratio**2)
        
        # Handle NaN or Inf
        if not np.isfinite(v):
            raise ValueError(f"Non-finite velocity computed for m={m_eV} eV, E={E_eV} eV")
        
        return v
    
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Invalid input types for v_sr: {e}")


def delta_sr(m_eV: float, E_eV: Optional[float]) -> float:
    """
    Compute fractional slowdown relative to c due to special relativity.
    
    For massless particles, returns 0.
    For massive particles, uses the approximation:
        δ_sr = 0.5 * (m/E)^2
    
    Parameters
    ----------
    m_eV : float
        Rest mass in electron volts (eV)
    E_eV : float or None
        Total energy in electron volts (eV). None for photons.
    
    Returns
    -------
    float
        Fractional slowdown (dimensionless)
    
    Raises
    ------
    ValueError
        If energy is less than or equal to mass, or if computation results in non-finite value
    
    Requirements: 2.3, 2.4, 10.1, 10.2, 10.3, 10.4, 11.3
    """
    try:
        # Validate inputs
        if not np.isfinite(m_eV) or m_eV < 0:
            raise ValueError(f"Invalid mass: {m_eV} eV (must be non-negative and finite)")
        
        # Photons or massless particles have no slowdown
        if m_eV == 0.0 or E_eV is None:
            return 0.0
        
        if not np.isfinite(E_eV):
            raise ValueError(f"Invalid energy: {E_eV} eV (must be finite)")
        
        # Validate that energy exceeds mass
        if E_eV <= m_eV:
            raise ValueError(f"Energy ({E_eV} eV) must be greater than rest mass ({m_eV} eV)")
        
        # Compute mass-to-energy ratio with safety clipping
        mass_energy_ratio = m_eV / E_eV
        mass_energy_ratio = min(mass_energy_ratio, MAX_MASS_ENERGY_RATIO)
        
        # Fractional slowdown formula
        delta = 0.5 * mass_energy_ratio**2
        
        # Handle NaN or Inf
        if not np.isfinite(delta):
            raise ValueError(f"Non-finite delta_sr computed for m={m_eV} eV, E={E_eV} eV")
        
        return delta
    
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Invalid input types for delta_sr: {e}")


def eta_from_eps(eps0: float, k_eps: float = 1.0) -> float:
    """
    Convert stiffness parameter eps0 to fractional offset eta.
    
    The relationship is:
        η = k_eps * (1 - eps0)
    
    Parameters
    ----------
    eps0 : float
        Dimensionless stiffness parameter (typically near 1.0)
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    
    Returns
    -------
    float
        Fractional offset eta (dimensionless)
    
    Requirements: 3.1
    """
    eta = k_eps * (1.0 - eps0)
    
    # Handle NaN or Inf
    if not np.isfinite(eta):
        raise ValueError(f"Non-finite eta computed for eps0={eps0}, k_eps={k_eps}")
    
    return eta


def delta_rigid(eps0: float, k_eps: float = 1.0) -> float:
    """
    Compute rigidity-induced fractional slowdown.
    
    The rigidity correction is:
        δ_rigid = η = k_eps * (1 - eps0)
    
    Parameters
    ----------
    eps0 : float
        Dimensionless stiffness parameter (typically near 1.0)
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    
    Returns
    -------
    float
        Rigidity-induced fractional slowdown (dimensionless)
    
    Requirements: 2.5, 3.2
    """
    return eta_from_eps(eps0, k_eps)


def verify_units(value: float, unit_name: str, expected_range: Optional[tuple] = None) -> bool:
    """
    Verify that a physical quantity has valid units and is within expected range.
    
    Parameters
    ----------
    value : float
        The value to verify
    unit_name : str
        Name of the unit for error messages
    expected_range : tuple or None, optional
        (min, max) expected range for the value. None means no range check.
    
    Returns
    -------
    bool
        True if value is valid, False otherwise
    
    Requirements: 10.1, 10.3
    """
    # Check for NaN or Inf
    if not np.isfinite(value):
        return False
    
    # Check expected range if provided
    if expected_range is not None:
        min_val, max_val = expected_range
        if value < min_val or value > max_val:
            return False
    
    return True
