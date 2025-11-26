"""
Output generation module for rigidity analysis results.

This module handles writing analysis results to JSON and text formats,
including best-fit parameters, uncertainties, and event-by-event residuals.
"""

import html
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from .likelihood import event_loglike, select_channel_pair
from .physics import c, MPC_TO_M, delta_sr, delta_rigid
from .stiffness import (
    fractional_speed_offset,
    compute_E0_event,
    coupling_general,
)


def get_version_metadata() -> Dict[str, str]:
    """
    Get versioning metadata for output files.
    
    Returns
    -------
    dict
        Dictionary with keys: 'tool_version', 'git_commit', 'python_version', 'timestamp'
    
    Requirements: 6.2
    """
    # Get git commit hash
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            timeout=5
        ).decode().strip()
    except Exception:
        git_commit = 'unknown'
    
    # Get Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Get timestamp in ISO format with UTC indicator
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    return {
        'tool_version': '1.0.0',
        'git_commit': git_commit,
        'python_version': python_version,
        'timestamp': timestamp
    }


def write_json_output(
    output_path: Path,
    best_eps0: float,
    uncertainty: Dict[str, float],
    n_events: int,
    scan_range: List[float],
    scan_steps: int,
    k_eps: float
) -> None:
    """
    Write eps0_posterior.json with best-fit, uncertainties, and metadata.
    
    Parameters
    ----------
    output_path : Path
        Path to output directory
    best_eps0 : float
        Best-fit eps0 value
    uncertainty : dict
        Dictionary with credible interval bounds
    n_events : int
        Number of events processed
    scan_range : list of float
        [min, max] eps0 scan range
    scan_steps : int
        Number of scan steps
    k_eps : float
        Rigidity coupling parameter
    
    Raises
    ------
    PermissionError
        If output directory cannot be created or file cannot be written
    OSError
        If disk is full or other I/O error occurs
    
    Requirements: 6.1, 6.2, 11.1, 11.2, 11.3
    """
    try:
        # Create output directory if it doesn't exist
        output_path.mkdir(exist_ok=True, parents=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating output directory {output_path}: {e}")
    except OSError as e:
        raise OSError(f"Failed to create output directory {output_path}: {e}")
    
    # Get versioning metadata
    metadata = get_version_metadata()
    metadata.update({
        'n_events': n_events,
        'scan_range': scan_range,
        'scan_steps': scan_steps,
        'k_eps': k_eps
    })
    
    # Construct output data
    output_data = {
        'best_eps0': best_eps0,
        'uncertainty': uncertainty,
        'metadata': metadata
    }
    
    # Write to JSON file
    json_file = output_path / 'eps0_posterior.json'
    try:
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    except PermissionError as e:
        raise PermissionError(f"Permission denied writing to {json_file}: {e}")
    except OSError as e:
        # Catch disk full and other I/O errors
        raise OSError(f"Failed to write {json_file}: {e}")


def compute_event_residuals(
    events: List[Dict[str, Any]],
    best_eps0: float,
    k_eps: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Compute residuals for each event at best-fit eps0.
    
    Parameters
    ----------
    events : list of dict
        List of event dictionaries
    best_eps0 : float
        Best-fit eps0 value
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    
    Returns
    -------
    list of dict
        List of residual information for each event with keys:
        'event_id', 'L_Mpc', 'n_channels', 'dt_obs', 'dt_model', 'residual', 'sigma_tot'
    
    Requirements: 6.4, 6.5
    """
    residuals = []
    
    for event in events:
        try:
            # Select the same channel pair used by the likelihood
            try:
                channel_A_name, channel_B_name = select_channel_pair(event)
            except ValueError:
                continue
            
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
            delta_rig = delta_rigid(best_eps0, k_eps)
            
            # Convert distance
            L_m = event['L_Mpc'] * MPC_TO_M
            
            # Compute model prediction
            dt_sr = (delta_sr_B - delta_sr_A) * (L_m / c)
            dt_rigid = delta_rig * (L_m / c)
            dt_model = dt_sr + dt_rigid + lag_mean
            
            # Compute total uncertainty
            sigma_tot = np.sqrt(sigma_A**2 + sigma_B**2 + lag_sigma**2)
            
            # Compute residual
            residual = dt_obs - dt_model
            
            # Optional E0 summary
            dt_int_bound = max(lag_sigma, 0.0)
            eps_max = None
            e0_min = None
            try:
                eps_max = fractional_speed_offset(dt_obs, dt_int_bound, event['L_Mpc'])
                if eps_max <= 0.0:
                    eps_max = None
            except ValueError:
                eps_max = None
            if (
                eps_max is not None
                and E_A not in (None, 0.0)
                and E_B not in (None, 0.0)
                and E_A > 0.0
                and E_B > 0.0
            ):
                try:
                    e0_min = compute_E0_event(
                        E_B,
                        m_B,
                        E_A,
                        m_A,
                        dt_obs,
                        dt_int_bound,
                        event['L_Mpc'],
                        coupling=coupling_general,
                    )
                except ValueError:
                    e0_min = None
            
            residuals.append({
                'event_id': event['id'],
                'L_Mpc': event['L_Mpc'],
                'n_channels': len(event['channels']),
                'dt_obs': dt_obs,
                'dt_model': dt_model,
                'residual': residual,
                'sigma_tot': sigma_tot,
                'eps_max': eps_max,
                'E0_min': e0_min,
            })
            
        except Exception:
            # Skip events that fail residual computation
            continue
    
    return residuals


def write_text_report(
    output_path: Path,
    best_eps0: float,
    uncertainty: Dict[str, float],
    events: List[Dict[str, Any]],
    scan_range: List[float],
    scan_steps: int,
    k_eps: float,
    runtime: float,
    residuals: List[Dict[str, Any]] | None = None,
) -> None:
    """
    Write rigidity_report.txt with formatted summary table and residuals.
    
    Parameters
    ----------
    output_path : Path
        Path to output directory
    best_eps0 : float
        Best-fit eps0 value
    uncertainty : dict
        Dictionary with credible interval bounds
    events : list of dict
        List of event dictionaries
    scan_range : list of float
        [min, max] eps0 scan range
    scan_steps : int
        Number of scan steps
    k_eps : float
        Rigidity coupling parameter
    runtime : float
        Total runtime in seconds
    
    Raises
    ------
    PermissionError
        If output directory cannot be created or file cannot be written
    OSError
        If disk is full or other I/O error occurs
    
    Requirements: 6.3, 6.4, 6.5, 11.1, 11.2, 11.3, 13.2
    """
    try:
        # Create output directory if it doesn't exist
        output_path.mkdir(exist_ok=True, parents=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating output directory {output_path}: {e}")
    except OSError as e:
        raise OSError(f"Failed to create output directory {output_path}: {e}")
    
    # Compute event residuals if not supplied
    if residuals is None:
        residuals = compute_event_residuals(events, best_eps0, k_eps)
    
    # Open text report file
    report_file = output_path / 'rigidity_report.txt'
    
    try:
        with open(report_file, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("Rigidity Test Build - Analysis Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Configuration section
            f.write("Configuration:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  eps0 range: [{scan_range[0]:.3f}, {scan_range[1]:.3f}]\n")
            f.write(f"  Scan steps: {scan_steps}\n")
            f.write(f"  k_eps: {k_eps:.3f}\n")
            f.write(f"  Number of events: {len(events)}\n")
            f.write("\n")
            
            # Results section
            f.write("Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Best-fit eps0: {best_eps0:.6f}\n")
            f.write(f"  68% credible interval: [{uncertainty['lower_68']:.6f}, {uncertainty['upper_68']:.6f}]\n")
            f.write(f"  95% credible interval: [{uncertainty['lower_95']:.6f}, {uncertainty['upper_95']:.6f}]\n")
            f.write("\n")
            
            # Event summary table
            f.write("Event Summary:\n")
            f.write("-" * 80 + "\n")
            
            # Table header
            header = (
                f"{'Event ID':<20} {'L (Mpc)':<10} {'Channels':<10} "
                f"{'Δt_obs (s)':<12} {'Δt_model (s)':<14} {'Residual (s)':<12} "
                f"{'ε_max':<10} {'E0_min':<12}"
            )
            f.write(header + "\n")
            f.write("-" * 80 + "\n")
            
            # Table rows
            for res in residuals:
                row = (
                    f"{res['event_id']:<20} "
                    f"{res['L_Mpc']:<10.1f} "
                    f"{res['n_channels']:<10} "
                    f"{res['dt_obs']:<12.6f} "
                    f"{res['dt_model']:<14.6f} "
                    f"{res['residual']:<12.6f} "
                    f"{(res['eps_max'] if res['eps_max'] is not None else float('nan')):<10.3e} "
                    f"{(res['E0_min'] if res['E0_min'] is not None else float('nan')):<12.3e}"
                )
                f.write(row + "\n")
            
            f.write("\n")
            
            # Runtime information
            f.write("Performance:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total runtime: {runtime:.2f} seconds\n")
            f.write(f"  Events processed: {len(events)}\n")
            f.write(f"  Average time per event: {runtime/len(events):.4f} seconds\n")
            f.write("\n")
            
            # Footer
            f.write("=" * 80 + "\n")
    
    except PermissionError as e:
        raise PermissionError(f"Permission denied writing to {report_file}: {e}")
    except OSError as e:
        raise OSError(f"Failed to write {report_file}: {e}")


def write_html_report(
    output_path: Path,
    best_eps0: float,
    uncertainty: Dict[str, float],
    events: List[Dict[str, Any]],
    scan_range: List[float],
    scan_steps: int,
    k_eps: float,
    runtime: float,
    residuals: List[Dict[str, Any]] | None = None,
) -> None:
    """
    Write an HTML summary for easier consumption in the browser.
    """
    try:
        output_path.mkdir(exist_ok=True, parents=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating output directory {output_path}: {e}")
    except OSError as e:
        raise OSError(f"Failed to create output directory {output_path}: {e}")

    if residuals is None:
        residuals = compute_event_residuals(events, best_eps0, k_eps)

    html_file = output_path / "rigidity_report.html"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt(value: float | None, spec: str) -> str:
        if value is None or not np.isfinite(value):
            return "—"
        return format(value, spec)

    try:
        with open(html_file, "w", encoding="utf-8") as handle:
            handle.write(
                f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rigidity Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background-color: #f8f9fb; color: #111; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ margin: 0.25rem 0 1rem; color: #444; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
    th, td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid #e0e0e0; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{ background: #0e1116; color: #fefefe; }}
    tfoot td {{ font-style: italic; }}
    .section {{ margin-top: 1.5rem; }}
    code {{ background: #eef0f4; padding: 0.1rem 0.3rem; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Rigidity Test Build – Analysis Report</h1>
  <p class="meta">Generated {generated}</p>

  <section class="section">
    <h2>Configuration</h2>
    <ul>
      <li>eps0 range: [{scan_range[0]:.3f}, {scan_range[1]:.3f}]</li>
      <li>Scan steps: {scan_steps}</li>
      <li>k_eps: {k_eps:.3f}</li>
      <li>Number of events: {len(events)}</li>
    </ul>
  </section>

  <section class="section">
    <h2>Results</h2>
    <ul>
      <li>Best-fit eps0: {best_eps0:.6f}</li>
      <li>68% credible interval: [{uncertainty['lower_68']:.6f}, {uncertainty['upper_68']:.6f}]</li>
      <li>95% credible interval: [{uncertainty['lower_95']:.6f}, {uncertainty['upper_95']:.6f}]</li>
    </ul>
  </section>

  <section class="section">
    <h2>Event Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Event ID</th>
          <th>L (Mpc)</th>
          <th>Channels</th>
          <th>Δt_obs (s)</th>
          <th>Δt_model (s)</th>
          <th>Residual (s)</th>
          <th>ε_max</th>
          <th>E0_min</th>
        </tr>
      </thead>
      <tbody>
"""
            )
            for res in residuals:
                handle.write(
                    "        <tr>"
                    f"<td>{html.escape(str(res['event_id']))}</td>"
                    f"<td>{res['L_Mpc']:.1f}</td>"
                    f"<td>{res['n_channels']}</td>"
                    f"<td>{res['dt_obs']:.6f}</td>"
                    f"<td>{res['dt_model']:.6f}</td>"
                    f"<td>{res['residual']:.6f}</td>"
                    f"<td>{fmt(res.get('eps_max'), '.3e')}</td>"
                    f"<td>{fmt(res.get('E0_min'), '.3e')}</td>"
                    "</tr>\n"
                )
            handle.write(
                f"""      </tbody>
      <tfoot>
        <tr><td colspan="8">Total runtime: {runtime:.2f}s — Average per event: {runtime/len(events):.4f}s</td></tr>
      </tfoot>
    </table>
  </section>
</body>
</html>
"""
            )
    except PermissionError as e:
        raise PermissionError(f"Permission denied writing to {html_file}: {e}")
    except OSError as e:
        raise OSError(f"Failed to write {html_file}: {e}")
