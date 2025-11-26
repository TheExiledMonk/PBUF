"""
Synthetic data generator for validation and testing.

This module generates synthetic multi-messenger events with known eps0 values
to validate the rigidity analysis pipeline and test parameter recovery.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, List


def generate_synthetic_event(
    event_id: str,
    L_Mpc: float,
    eps0_true: float,
    k_eps: float = 1.0,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate a synthetic event with known eps0 value.
    
    Creates an event with two channels: a photon and a massive particle.
    The timing is computed from the true eps0 value with added Gaussian noise.
    
    Parameters
    ----------
    event_id : str
        Unique identifier for the event
    L_Mpc : float
        Source distance in megaparsecs
    eps0_true : float
        True eps0 value to use for generating timing
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    seed : int or None, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Synthetic event dictionary
    
    Requirements: 7.1, 7.2
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Physical constants
    c = 299792458.0  # m/s
    MPC_TO_M = 3.085677581e22  # m per Mpc
    
    # Channel parameters
    # Photon channel
    m_photon = 0.0
    E_photon = None
    sigma_photon = 0.001  # 1 ms timing uncertainty
    
    # Massive particle channel (e.g., neutrino-like)
    m_massive = 0.1  # eV
    E_massive = 1e6  # 1 MeV
    sigma_massive = 0.1  # 100 ms timing uncertainty
    
    # Compute special-relativistic delays
    delta_sr_photon = 0.0
    delta_sr_massive = 0.5 * (m_massive / E_massive)**2
    
    # Compute rigidity correction
    eta = k_eps * (1.0 - eps0_true)
    delta_rigid = eta
    
    # Convert distance to meters
    L_m = L_Mpc * MPC_TO_M
    
    # Compute time differences
    dt_sr = (delta_sr_massive - delta_sr_photon) * (L_m / c)
    dt_rigid = delta_rigid * (L_m / c)
    
    # Intrinsic lag model
    lag_mean = 0.0
    lag_sigma = 2.0  # 2 second intrinsic uncertainty
    
    # Total model prediction
    dt_model = dt_sr + dt_rigid + lag_mean
    
    # Add Gaussian noise
    sigma_tot = np.sqrt(sigma_photon**2 + sigma_massive**2 + lag_sigma**2)
    dt_obs = dt_model + np.random.normal(0, sigma_tot)
    
    # Set photon arrival time as reference (t=0)
    t_photon = 0.0
    t_massive = dt_obs
    
    # Construct event dictionary
    event = {
        "id": event_id,
        "L_Mpc": L_Mpc,
        "channels": {
            "gamma": {
                "t_obs": t_photon,
                "sigma_t": sigma_photon,
                "mass_eV": m_photon,
                "E_eV": E_photon
            },
            "neutrino": {
                "t_obs": t_massive,
                "sigma_t": sigma_massive,
                "mass_eV": m_massive,
                "E_eV": E_massive
            }
        },
        "intrinsic_lag_model": {
            "mean": lag_mean,
            "sigma": lag_sigma
        }
    }
    
    return event


def generate_photon_only_event(
    event_id: str,
    L_Mpc: float,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate a photon-only event to test eps0 ≈ 1.0 recovery.
    
    With only photons, there should be no mass-dependent effects,
    so eps0 should be recovered as approximately 1.0.
    
    Parameters
    ----------
    event_id : str
        Unique identifier for the event
    L_Mpc : float
        Source distance in megaparsecs
    seed : int or None, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Synthetic photon-only event dictionary
    
    Requirements: 7.3
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Two photon channels with slightly different arrival times
    # (simulating different energy bands or detection methods)
    sigma_1 = 0.001  # 1 ms
    sigma_2 = 0.002  # 2 ms
    
    # Intrinsic lag
    lag_mean = 0.0
    lag_sigma = 1.0
    
    # Generate small random time difference (from intrinsic lag)
    sigma_tot = np.sqrt(sigma_1**2 + sigma_2**2 + lag_sigma**2)
    dt_obs = np.random.normal(lag_mean, sigma_tot)
    
    # Construct event
    event = {
        "id": event_id,
        "L_Mpc": L_Mpc,
        "channels": {
            "gamma_low": {
                "t_obs": 0.0,
                "sigma_t": sigma_1,
                "mass_eV": 0.0,
                "E_eV": None
            },
            "gamma_high": {
                "t_obs": dt_obs,
                "sigma_t": sigma_2,
                "mass_eV": 0.0,
                "E_eV": None
            }
        },
        "intrinsic_lag_model": {
            "mean": lag_mean,
            "sigma": lag_sigma
        }
    }
    
    return event


def generate_sensitivity_events(
    base_id: str,
    eps0_true: float,
    k_eps: float = 1.0,
    seed: int = None
) -> List[Dict[str, Any]]:
    """
    Generate events with varying timing precision and distances for sensitivity studies.
    
    Parameters
    ----------
    base_id : str
        Base identifier for events (will be appended with suffixes)
    eps0_true : float
        True eps0 value
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    seed : int or None, optional
        Random seed for reproducibility
    
    Returns
    -------
    list of dict
        List of synthetic events with varying parameters
    
    Requirements: 7.3
    """
    events = []
    
    # Vary distance: near, medium, far
    distances = [10.0, 50.0, 200.0]  # Mpc
    
    for i, L_Mpc in enumerate(distances):
        event_seed = seed + i if seed is not None else None
        event = generate_synthetic_event(
            event_id=f"{base_id}_dist_{int(L_Mpc)}Mpc",
            L_Mpc=L_Mpc,
            eps0_true=eps0_true,
            k_eps=k_eps,
            seed=event_seed
        )
        events.append(event)
    
    # Vary timing precision: high, medium, low
    precisions = [0.0001, 0.001, 0.01]  # seconds
    
    for i, sigma in enumerate(precisions):
        event_seed = seed + 100 + i if seed is not None else None
        if event_seed is not None:
            np.random.seed(event_seed)
        
        # Generate event with custom precision
        event = generate_synthetic_event(
            event_id=f"{base_id}_prec_{sigma*1000:.1f}ms",
            L_Mpc=40.0,
            eps0_true=eps0_true,
            k_eps=k_eps,
            seed=event_seed
        )
        
        # Override timing uncertainties
        event['channels']['gamma']['sigma_t'] = sigma
        event['channels']['neutrino']['sigma_t'] = sigma * 10
        
        events.append(event)
    
    return events


def main():
    """
    CLI entry point for synthetic data generation.
    
    Requirements: 7.1
    """
    parser = argparse.ArgumentParser(
        description='Generate synthetic test data for rigidity analysis validation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/test_events/',
        help='Output directory for synthetic events'
    )
    
    parser.add_argument(
        '--eps0',
        type=float,
        default=0.995,
        help='True eps0 value for synthetic data'
    )
    
    parser.add_argument(
        '--k-eps',
        type=float,
        default=1.0,
        help='Rigidity coupling parameter'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--n-events',
        type=int,
        default=10,
        help='Number of standard synthetic events to generate'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic test data with eps0={args.eps0}")
    print(f"Output directory: {output_dir}")
    
    # Generate main synthetic event
    print("\nGenerating synthetic_event.json...")
    synthetic_event = generate_synthetic_event(
        event_id="synthetic_test",
        L_Mpc=40.0,
        eps0_true=args.eps0,
        k_eps=args.k_eps,
        seed=args.seed
    )
    
    with open(output_dir / 'synthetic_event.json', 'w') as f:
        json.dump(synthetic_event, f, indent=2)
    
    print(f"  Created: {output_dir / 'synthetic_event.json'}")
    
    # Generate photon-only event
    print("\nGenerating photon_only_event.json...")
    photon_event = generate_photon_only_event(
        event_id="photon_only_test",
        L_Mpc=40.0,
        seed=args.seed + 1000
    )
    
    with open(output_dir / 'photon_only_event.json', 'w') as f:
        json.dump(photon_event, f, indent=2)
    
    print(f"  Created: {output_dir / 'photon_only_event.json'}")
    
    # Generate sensitivity test events
    print("\nGenerating sensitivity test events...")
    sensitivity_events = generate_sensitivity_events(
        base_id="sensitivity",
        eps0_true=args.eps0,
        k_eps=args.k_eps,
        seed=args.seed + 2000
    )
    
    for event in sensitivity_events:
        filename = f"{event['id']}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(event, f, indent=2)
        print(f"  Created: {output_dir / filename}")
    
    # Generate multiple standard events for performance testing
    if args.n_events > 1:
        print(f"\nGenerating {args.n_events} standard events for performance testing...")
        for i in range(args.n_events):
            event = generate_synthetic_event(
                event_id=f"perf_test_{i:03d}",
                L_Mpc=40.0 + i * 5.0,  # Vary distance slightly
                eps0_true=args.eps0,
                k_eps=args.k_eps,
                seed=args.seed + 3000 + i
            )
            
            filename = f"perf_test_{i:03d}.json"
            with open(output_dir / filename, 'w') as f:
                json.dump(event, f, indent=2)
        
        print(f"  Created {args.n_events} performance test events")
    
    print(f"\nSynthetic data generation complete!")
    print(f"Total files created: {len(list(output_dir.glob('*.json')))}")


if __name__ == '__main__':
    main()
