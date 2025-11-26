"""
Visualization module for rigidity analysis results.

This module generates plots for posterior distributions and event residuals.
Plotting is optional and gracefully handles matplotlib unavailability.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for saving files
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_posterior(
    eps_values: np.ndarray,
    loglikes: np.ndarray,
    output_path: Path,
    best_eps0: Optional[float] = None
) -> None:
    """
    Plot posterior distribution of eps0 with credible regions.
    
    Creates a visualization showing:
    - Likelihood distribution across eps0 values
    - Best-fit value marked with vertical line
    - 68% and 95% credible regions shaded
    
    Parameters
    ----------
    eps_values : np.ndarray
        Array of eps0 values from scan
    loglikes : np.ndarray
        Array of log-likelihood values corresponding to eps_values
    output_path : Path
        Directory where plot will be saved
    best_eps0 : float, optional
        Best-fit eps0 value to mark on plot. If None, uses maximum likelihood value.
    
    Requirements: 6.3
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Convert log-likelihoods to normalized likelihoods
    # Subtract maximum for numerical stability
    max_loglike = np.max(loglikes)
    likelihoods = np.exp(loglikes - max_loglike)
    
    # Normalize to get posterior (assuming flat prior)
    posterior = likelihoods / np.trapz(likelihoods, eps_values)
    
    # Find best-fit if not provided
    if best_eps0 is None:
        best_idx = np.argmax(loglikes)
        best_eps0 = eps_values[best_idx]
    
    # Compute credible intervals
    cumulative = np.cumsum(posterior) * (eps_values[1] - eps_values[0])
    cumulative = cumulative / cumulative[-1]  # Normalize to 1
    
    # Find 68% credible interval (±1σ equivalent)
    lower_68_idx = np.searchsorted(cumulative, 0.16)
    upper_68_idx = np.searchsorted(cumulative, 0.84)
    
    # Find 95% credible interval (±2σ equivalent)
    lower_95_idx = np.searchsorted(cumulative, 0.025)
    upper_95_idx = np.searchsorted(cumulative, 0.975)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot posterior distribution
    ax.plot(eps_values, posterior, 'b-', linewidth=2, label='Posterior')
    
    # Shade 95% credible region
    ax.fill_between(
        eps_values[lower_95_idx:upper_95_idx+1],
        posterior[lower_95_idx:upper_95_idx+1],
        alpha=0.2,
        color='blue',
        label='95% credible region'
    )
    
    # Shade 68% credible region
    ax.fill_between(
        eps_values[lower_68_idx:upper_68_idx+1],
        posterior[lower_68_idx:upper_68_idx+1],
        alpha=0.3,
        color='blue',
        label='68% credible region'
    )
    
    # Mark best-fit value
    ax.axvline(best_eps0, color='red', linestyle='--', linewidth=2, label=f'Best fit: {best_eps0:.6f}')
    
    # Labels and title
    ax.set_xlabel('eps0 (Stiffness Parameter)', fontsize=12)
    ax.set_ylabel('Posterior Probability Density', fontsize=12)
    ax.set_title('Spacetime Rigidity Posterior Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_path / f'posterior_plot_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_event_residuals(
    events: List[Dict[str, Any]],
    best_eps0: float,
    output_path: Path,
    k_eps: float = 1.0
) -> None:
    """
    Plot per-event residuals with error bars.
    
    Creates a visualization showing:
    - Residual (observed - model) for each event
    - Error bars from total uncertainties
    - Horizontal line at zero for reference
    
    Parameters
    ----------
    events : list of dict
        List of event dictionaries
    best_eps0 : float
        Best-fit eps0 value for computing model predictions
    output_path : Path
        Directory where plot will be saved
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    
    Requirements: 6.3
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Import here to avoid circular dependency
    from .output import compute_event_residuals
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Compute residuals for all events
    residuals = compute_event_residuals(events, best_eps0, k_eps)
    
    if not residuals:
        return
    
    # Extract data for plotting
    event_ids = [r['event_id'] for r in residuals]
    residual_values = [r['residual'] for r in residuals]
    uncertainties = [r['sigma_tot'] for r in residuals]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot residuals with error bars
    x_positions = np.arange(len(event_ids))
    ax.errorbar(
        x_positions,
        residual_values,
        yerr=uncertainties,
        fmt='o',
        markersize=6,
        capsize=5,
        capthick=2,
        color='blue',
        ecolor='gray',
        label='Event residuals'
    )
    
    # Add horizontal line at zero
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    
    # Labels and title
    ax.set_xlabel('Event', fontsize=12)
    ax.set_ylabel('Residual (Observed - Model) [seconds]', fontsize=12)
    ax.set_title(f'Event Residuals at Best-Fit eps0 = {best_eps0:.6f}', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(event_ids, rotation=45, ha='right', fontsize=9)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Tight layout
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_path / f'residuals_plot_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()


def is_matplotlib_available() -> bool:
    """
    Check if matplotlib is available for plotting.
    
    Returns
    -------
    bool
        True if matplotlib is available, False otherwise
    """
    return MATPLOTLIB_AVAILABLE
