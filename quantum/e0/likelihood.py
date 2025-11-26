"""
Likelihood module for statistical inference of spacetime rigidity.

This module implements log-likelihood computation for multi-messenger timing data,
comparing observed arrival-time differences with predictions from special relativity
plus rigidity corrections.
"""

import numpy as np
from typing import Dict, List, Any, Tuple

from .physics import delta_sr, delta_rigid, c, MPC_TO_M


def select_channel_pair(event: Dict[str, Any]) -> Tuple[str, str]:
    channels = event.get('channels', {})
    if not isinstance(channels, dict):
        raise ValueError(f"Event {event.get('id', 'unknown')} channels malformed")

    requested = event.get('likelihood_channels')
    if requested:
        ordered: List[str] = []
        for name in requested:
            if name not in channels:
                raise ValueError(f"Event {event.get('id', 'unknown')} missing requested channel '{name}'")
            if name not in ordered:
                ordered.append(name)
            if len(ordered) == 2:
                return ordered[0], ordered[1]
        raise ValueError(f"Event {event.get('id', 'unknown')} requires at least two likelihood channels")

    if 'gw' in channels:
        prioritized = ['gamma', 'neutrino', 'optical']
        for candidate in prioritized:
            if candidate == 'gw':
                continue
            if candidate in channels:
                return 'gw', candidate
        remaining = sorted(name for name in channels.keys() if name != 'gw')
        if remaining:
            return 'gw', remaining[0]
        raise ValueError(f"Event {event.get('id', 'unknown')} has GW channel only")

    channel_names = sorted(channels.keys())
    if len(channel_names) < 2:
        raise ValueError(f"Event {event.get('id', 'unknown')} has fewer than 2 channels")
    return channel_names[0], channel_names[1]


def event_loglike(event: Dict[str, Any], eps0: float, k_eps: float = 1.0) -> float:
    """
    Compute log-likelihood for a single event given eps0.
    
    The log-likelihood quantifies how well a given eps0 value explains the
    observed timing data. It uses a Gaussian model with uncertainties from
    both measurement errors and intrinsic source lag.
    
    Parameters
    ----------
    event : dict
        Event dictionary with structure:
        {
            'id': str,
            'L_Mpc': float,
            'channels': {
                'channel_name': {
                    't_obs': float,
                    'sigma_t': float,
                    'mass_eV': float,
                    'E_eV': float or None
                }
            },
            'intrinsic_lag_model': {
                'mean': float,
                'sigma': float
            }
        }
    eps0 : float
        Dimensionless stiffness parameter (typically near 1.0)
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    
    Returns
    -------
    float
        Log-likelihood value
    
    Raises
    ------
    ValueError
        If event has invalid structure or computation fails
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 11.3
    """
    try:
        channel_A_name, channel_B_name = select_channel_pair(event)
        channel_A = event['channels'][channel_A_name]
        channel_B = event['channels'][channel_B_name]
        
        # Extract channel data
        t_A = channel_A['t_obs']
        t_B = channel_B['t_obs']
        sigma_A = channel_A['sigma_t']
        sigma_B = channel_B['sigma_t']
        
        m_A = channel_A['mass_eV']
        E_A = channel_A.get('E_eV', None)
        m_B = channel_B['mass_eV']
        E_B = channel_B.get('E_eV', None)
        
        # Extract intrinsic lag model
        lag_mean = event['intrinsic_lag_model']['mean']
        lag_sigma = event['intrinsic_lag_model']['sigma']
        
        # Compute observed time difference
        dt_obs = t_B - t_A
        
        # Compute special-relativistic delays for each channel
        delta_sr_A = delta_sr(m_A, E_A)
        delta_sr_B = delta_sr(m_B, E_B)
        
        # Compute rigidity correction (same for both channels)
        delta_rig = delta_rigid(eps0, k_eps)
        
        # Convert distance from Mpc to meters
        L_m = event['L_Mpc'] * MPC_TO_M
        
        # Compute special-relativistic time difference
        # Δt_sr = (δ_sr,B - δ_sr,A) * (L / c)
        dt_sr = (delta_sr_B - delta_sr_A) * (L_m / c)
        
        # Compute rigidity-induced time difference
        # Δt_rigid = δ_rigid * (L / c)
        dt_rigid = delta_rig * (L_m / c)
        
        # Compute total model prediction
        dt_model = dt_sr + dt_rigid + lag_mean
        
        # Compute total uncertainty
        # σ_tot = sqrt(σ_A^2 + σ_B^2 + σ_src^2)
        sigma_tot = np.sqrt(sigma_A**2 + sigma_B**2 + lag_sigma**2)
        
        # Handle edge case: zero uncertainty
        if sigma_tot == 0:
            # If uncertainty is zero, use a small epsilon to avoid division by zero
            sigma_tot = 1e-10
    
        # Compute log-likelihood using Gaussian formula
        # log L = -0.5 * ((Δt_obs - Δt_model) / σ_tot)^2
        residual = dt_obs - dt_model
        loglike = -0.5 * (residual / sigma_tot)**2
        
        # Check for NaN or Inf
        if not np.isfinite(loglike):
            raise ValueError(f"Non-finite log-likelihood for event {event.get('id', 'unknown')}")
        
        return loglike
    
    except KeyError as e:
        raise ValueError(f"Missing required field in event {event.get('id', 'unknown')}: {e}")
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Invalid event structure for {event.get('id', 'unknown')}: {e}")
    except Exception as e:
        raise ValueError(f"Error computing likelihood for event {event.get('id', 'unknown')}: {e}")


def compute_event_residual(event: Dict[str, Any], eps0: float, k_eps: float = 1.0) -> Dict[str, Any]:
    """
    Compute residual information for a single event.
    
    Parameters
    ----------
    event : dict
        Event dictionary
    eps0 : float
        Dimensionless stiffness parameter
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    
    Returns
    -------
    dict
        Dictionary with keys: 'event_id', 'dt_obs', 'dt_model', 'residual', 'sigma_tot'
    
    Requirements: 8.5
    """
    try:
        channel_A_name, channel_B_name = select_channel_pair(event)
        channel_A = event['channels'][channel_A_name]
        channel_B = event['channels'][channel_B_name]
        
        # Extract channel data
        t_A = channel_A['t_obs']
        t_B = channel_B['t_obs']
        sigma_A = channel_A['sigma_t']
        sigma_B = channel_B['sigma_t']
        
        m_A = channel_A['mass_eV']
        E_A = channel_A.get('E_eV', None)
        m_B = channel_B['mass_eV']
        E_B = channel_B.get('E_eV', None)
        
        # Extract intrinsic lag model
        lag_mean = event['intrinsic_lag_model']['mean']
        lag_sigma = event['intrinsic_lag_model']['sigma']
        
        # Compute observed time difference
        dt_obs = t_B - t_A
        
        # Compute special-relativistic delays
        delta_sr_A = delta_sr(m_A, E_A)
        delta_sr_B = delta_sr(m_B, E_B)
        
        # Compute rigidity correction
        delta_rig = delta_rigid(eps0, k_eps)
        
        # Convert distance from Mpc to meters
        L_m = event['L_Mpc'] * MPC_TO_M
        
        # Compute model prediction
        dt_sr = (delta_sr_B - delta_sr_A) * (L_m / c)
        dt_rigid = delta_rig * (L_m / c)
        dt_model = dt_sr + dt_rigid + lag_mean
        
        # Compute total uncertainty
        sigma_tot = np.sqrt(sigma_A**2 + sigma_B**2 + lag_sigma**2)
        
        # Compute residual
        residual = dt_obs - dt_model
        
        return {
            'event_id': event.get('id', 'unknown'),
            'dt_obs': dt_obs,
            'dt_model': dt_model,
            'residual': residual,
            'sigma_tot': sigma_tot
        }
    except Exception as e:
        return {
            'event_id': event.get('id', 'unknown'),
            'dt_obs': np.nan,
            'dt_model': np.nan,
            'residual': np.nan,
            'sigma_tot': np.nan,
            'error': str(e)
        }


def total_loglike(events: List[Dict[str, Any]], eps0: float, k_eps: float = 1.0) -> float:
    """
    Compute total log-likelihood across all events with graceful degradation.
    
    The total log-likelihood is the sum of individual event log-likelihoods,
    assuming events are independent measurements. If individual events fail,
    they are skipped and processing continues.
    
    Parameters
    ----------
    events : list of dict
        List of event dictionaries
    eps0 : float
        Dimensionless stiffness parameter (typically near 1.0)
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    
    Returns
    -------
    float
        Total log-likelihood summed across all events
    
    Requirements: 4.5, 11.5
    """
    if not events:
        return 0.0
    
    total = 0.0
    failed_count = 0
    
    for event in events:
        try:
            loglike = event_loglike(event, eps0, k_eps)
            total += loglike
        except ValueError as e:
            # Log error but continue with other events (graceful degradation)
            try:
                from .logger import log_error
                log_error(f"Failed to compute likelihood for event {event.get('id', 'unknown')}: {e}")
            except ImportError:
                # Fallback if logger not available
                from .events import _log_error
                _log_error(f"Failed to compute likelihood for event {event.get('id', 'unknown')}: {e}")
            failed_count += 1
            continue
        except Exception as e:
            # Catch unexpected errors
            try:
                from .logger import log_error
                log_error(f"Unexpected error for event {event.get('id', 'unknown')}: {e}")
            except ImportError:
                from .events import _log_error
                _log_error(f"Unexpected error for event {event.get('id', 'unknown')}: {e}")
            failed_count += 1
            continue
    
    return total
