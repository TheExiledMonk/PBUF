"""
Main driver for spacetime rigidity analysis.

This module provides the command-line interface and orchestrates the fitting
process to estimate the eps0 parameter from multi-messenger timing data.
"""

import argparse
import json
import yaml
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

# Import local modules
from .events import load_events_from_directory, validate_event
from .likelihood import total_loglike, event_loglike, compute_event_residual
from .logger import log_info, log_warn, log_error, log_debug, log_runtime_summary, set_debug_mode
from .output import write_json_output, write_text_report, write_html_report, compute_event_residuals


def _compute_loglike_for_eps0(eps0: float, events: List[Dict[str, Any]], k_eps: float) -> float:
    """
    Helper function to compute log-likelihood for a single eps0 value.
    
    This function is used for parallel processing.
    
    Parameters
    ----------
    eps0 : float
        Stiffness parameter value
    events : list of dict
        List of validated event dictionaries
    k_eps : float
        Rigidity coupling parameter
    
    Returns
    -------
    float
        Total log-likelihood for this eps0 value
    
    Requirements: 8.5
    """
    return total_loglike(events, eps0, k_eps)


def scan_eps0(
    events: List[Dict[str, Any]],
    eps_range: Tuple[float, float],
    steps: int = 500,
    k_eps: float = 1.0,
    progress: bool = False,
    threads: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform parameter space scan over eps0 values.
    
    This function evaluates the log-likelihood across a grid of eps0 values
    to identify the best-fit parameter and construct the posterior distribution.
    Supports parallel processing for multi-core CPUs.
    
    Parameters
    ----------
    events : list of dict
        List of validated event dictionaries
    eps_range : tuple of (float, float)
        (min, max) range for eps0 scan
    steps : int, optional
        Number of scan steps (default: 500)
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    progress : bool, optional
        Show progress bar if True (default: False)
    threads : int, optional
        Number of parallel threads (default: 1, no parallelization)
    
    Returns
    -------
    eps_values : np.ndarray
        Array of eps0 values scanned
    loglikes : np.ndarray
        Array of log-likelihood values corresponding to eps_values
    
    Requirements: 5.1, 5.2, 8.5, 13.3
    """
    eps_min, eps_max = eps_range
    
    # Create eps0 grid using numpy.linspace
    eps_values = np.linspace(eps_min, eps_max, steps)
    
    # Initialize log-likelihood array
    loglikes = np.zeros(steps)
    
    # Determine if we should use parallel processing
    use_parallel = threads > 1 and len(eps_values) > 10
    
    if use_parallel:
        # Parallel processing mode
        log_info(f"Using parallel processing with {threads} threads")
        
        # Create partial function with fixed events and k_eps
        compute_func = partial(_compute_loglike_for_eps0, events=events, k_eps=k_eps)
        
        # Optional progress bar
        iterator = eps_values
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(eps_values, desc="Scanning eps0")
            except ImportError:
                pass
        
        # Use multiprocessing pool
        try:
            with Pool(processes=threads) as pool:
                if progress:
                    # For progress bar, use imap for incremental results
                    loglikes = list(pool.imap(compute_func, iterator))
                else:
                    # Without progress bar, use map for better performance
                    loglikes = pool.map(compute_func, eps_values)
            
            loglikes = np.array(loglikes)
        
        except Exception as e:
            log_error(f"Parallel processing failed, falling back to serial: {e}")
            # Fall back to serial processing
            use_parallel = False
    
    if not use_parallel:
        # Serial processing mode
        iterator = eps_values
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(eps_values, desc="Scanning eps0")
            except ImportError:
                # tqdm not available, proceed without progress bar
                pass
        
        # Compute log-likelihood for each eps0 value
        for i, eps0 in enumerate(iterator):
            loglikes[i] = total_loglike(events, eps0, k_eps)
    
    return eps_values, loglikes


def compute_credible_intervals(
    eps_values: np.ndarray,
    loglikes: np.ndarray
) -> Dict[str, float]:
    """
    Compute credible intervals from log-likelihood distribution.
    
    Parameters
    ----------
    eps_values : np.ndarray
        Array of eps0 values
    loglikes : np.ndarray
        Array of log-likelihood values
    
    Returns
    -------
    dict
        Dictionary with keys: 'lower_68', 'upper_68', 'lower_95', 'upper_95'
    
    Requirements: 5.4
    """
    # Convert log-likelihood to normalized probability
    # Subtract maximum for numerical stability
    loglike_shifted = loglikes - np.max(loglikes)
    probs = np.exp(loglike_shifted)
    # Use trapezoid (trapz is deprecated in newer numpy)
    try:
        probs = probs / np.trapezoid(probs, eps_values)  # Normalize
    except AttributeError:
        probs = probs / np.trapz(probs, eps_values)  # Fallback for older numpy
    
    # Compute cumulative distribution
    cumulative = np.cumsum(probs) * (eps_values[1] - eps_values[0])
    
    # Find credible interval bounds
    # 68% credible interval (1-sigma equivalent)
    lower_68_idx = np.searchsorted(cumulative, 0.16)
    upper_68_idx = np.searchsorted(cumulative, 0.84)
    
    # 95% credible interval (2-sigma equivalent)
    lower_95_idx = np.searchsorted(cumulative, 0.025)
    upper_95_idx = np.searchsorted(cumulative, 0.975)
    
    # Handle edge cases
    lower_68_idx = min(lower_68_idx, len(eps_values) - 1)
    upper_68_idx = min(upper_68_idx, len(eps_values) - 1)
    lower_95_idx = min(lower_95_idx, len(eps_values) - 1)
    upper_95_idx = min(upper_95_idx, len(eps_values) - 1)
    
    return {
        'lower_68': float(eps_values[lower_68_idx]),
        'upper_68': float(eps_values[upper_68_idx]),
        'lower_95': float(eps_values[lower_95_idx]),
        'upper_95': float(eps_values[upper_95_idx])
    }


def compute_rigidity(
    events: List[Dict[str, Any]],
    eps_range: Tuple[float, float] = (0.9, 1.1),
    steps: int = 500,
    k_eps: float = 1.0,
    progress: bool = False,
    threads: int = 1
) -> Dict[str, Any]:
    """
    Main entry point for rigidity computation.
    
    This function orchestrates the complete analysis: scanning the parameter
    space, identifying the best-fit eps0, and computing uncertainty estimates.
    
    Parameters
    ----------
    events : list of dict
        List of validated event dictionaries
    eps_range : tuple of (float, float), optional
        (min, max) range for eps0 scan (default: (0.9, 1.1))
    steps : int, optional
        Number of scan steps (default: 500)
    k_eps : float, optional
        Rigidity coupling parameter (default: 1.0)
    progress : bool, optional
        Show progress bar if True (default: False)
    threads : int, optional
        Number of parallel threads (default: 1, no parallelization)
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'best_eps0': Best-fit eps0 value
        - 'eps_values': Array of scanned eps0 values
        - 'loglikes': Array of log-likelihood values
        - 'uncertainty': Dict with credible intervals
        - 'n_events': Number of events processed
    
    Requirements: 5.3, 5.4, 5.5, 8.5
    """
    # Perform parameter space scan
    eps_values, loglikes = scan_eps0(events, eps_range, steps, k_eps, progress, threads)
    
    # Identify best-fit eps0 as maximum likelihood value
    best_idx = np.argmax(loglikes)
    best_eps0 = float(eps_values[best_idx])
    
    # Compute uncertainty estimates
    uncertainty = compute_credible_intervals(eps_values, loglikes)
    
    # Package results
    results = {
        'best_eps0': best_eps0,
        'eps_values': eps_values,
        'loglikes': loglikes,
        'uncertainty': uncertainty,
        'n_events': len(events)
    }
    
    return results


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file with error handling.
    
    Parameters
    ----------
    config_path : str or None, optional
        Path to config file. If None, looks for config.yaml or config.json
        in current directory.
    
    Returns
    -------
    dict
        Configuration dictionary, empty if no config file found or error occurs
    
    Requirements: 8.1, 11.1, 11.2
    """
    if config_path is None:
        # Look for default config files
        for filename in ['config.yaml', 'config.yml', 'config.json']:
            try:
                if Path(filename).exists():
                    config_path = filename
                    break
            except Exception:
                continue
    
    if config_path is None:
        return {}
    
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            log_warn(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    log_error(f"Invalid YAML in config file {config_path}: {e}")
                    return {}
            elif config_path.suffix == '.json':
                try:
                    config = json.load(f)
                except json.JSONDecodeError as e:
                    log_error(f"Invalid JSON in config file {config_path}: {e}")
                    return {}
            else:
                log_error(f"Unsupported config file format: {config_path.suffix}")
                return {}
        
        return config if config is not None else {}
    
    except FileNotFoundError:
        log_warn(f"Config file not found: {config_path}")
        return {}
    except PermissionError as e:
        log_error(f"Permission denied reading config file {config_path}: {e}")
        return {}
    except Exception as e:
        log_error(f"Failed to load config file {config_path}: {e}")
        return {}





def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    parser = argparse.ArgumentParser(
        description='Rigidity Test Build - Estimate spacetime rigidity from multi-messenger timing data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/events/',
        help='Path to directory containing event files'
    )
    
    parser.add_argument(
        '--eps-min',
        type=float,
        default=None,
        help='Minimum eps0 value for scan'
    )
    
    parser.add_argument(
        '--eps-max',
        type=float,
        default=None,
        help='Maximum eps0 value for scan'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of scan steps'
    )
    
    parser.add_argument(
        '--k-eps',
        type=float,
        default=None,
        help='Rigidity coupling parameter'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with full tracebacks'
    )
    
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show progress bar for long scans'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-event residuals to console'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of parallel threads for event evaluation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML or JSON)'
    )
    
    return parser.parse_args()


def main():
    """
    CLI entry point for rigidity analysis.
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration file
    config = load_config(args.config)
    
    # Merge config and CLI args (CLI args override config)
    data_dir = args.data_dir if args.data_dir != 'data/events/' else config.get('data_dir', 'data/events/')
    eps_min = args.eps_min if args.eps_min is not None else config.get('eps_min', 0.9)
    eps_max = args.eps_max if args.eps_max is not None else config.get('eps_max', 1.1)
    steps = args.steps if args.steps is not None else config.get('steps', 500)
    k_eps = args.k_eps if args.k_eps is not None else config.get('k_eps', 1.0)
    output_dir = args.output_dir if args.output_dir != 'outputs/' else config.get('output_dir', 'outputs/')
    debug = args.debug or config.get('debug', False)
    progress = args.progress or config.get('progress', False)
    verbose = args.verbose or config.get('verbose', False)
    threads = args.threads if args.threads != 1 else config.get('threads', 1)
    
    # Set debug mode for logging
    set_debug_mode(debug)
    
    # Record start time
    start_time = datetime.now()
    
    try:
        log_info("Starting rigidity analysis")
        
        # Load events from directory
        log_info(f"Loading events from {data_dir}")
        events = load_events_from_directory(data_dir)
        
        if not events:
            print(f"Error: No valid events found in {data_dir}")
            log_error(f"No valid events found in {data_dir}")
            sys.exit(1)
        
        log_info(f"Loaded {len(events)} events")
        print(f"Loaded {len(events)} events from {data_dir}")
        
        # Validate all events
        valid_events = []
        for event in events:
            is_valid, error_msg = validate_event(event)
            if is_valid:
                valid_events.append(event)
            else:
                log_error(f"Event {event.get('id', 'unknown')} validation failed: {error_msg}")
                if verbose:
                    print(f"Warning: Skipping invalid event {event.get('id', 'unknown')}: {error_msg}")
        
        if not valid_events:
            print("Error: No valid events after validation")
            log_error("No valid events after validation")
            sys.exit(1)
        
        log_info(f"Validated {len(valid_events)} events")
        if len(valid_events) < len(events):
            print(f"Warning: {len(events) - len(valid_events)} events failed validation (see logs/errors.txt)")
        
        # Perform eps0 scan
        log_info(f"Scanning eps0 from {eps_min} to {eps_max} ({steps} steps)")
        print(f"Scanning eps0 range [{eps_min}, {eps_max}] with {steps} steps...")
        
        results = compute_rigidity(
            valid_events,
            eps_range=(eps_min, eps_max),
            steps=steps,
            k_eps=k_eps,
            progress=progress,
            threads=threads
        )
        
        best_eps0 = results['best_eps0']
        uncertainty = results['uncertainty']
        
        log_info(f"Scan complete. Best eps0: {best_eps0}")
        print(f"\nBest-fit eps0: {best_eps0:.6f}")
        print(f"68% credible interval: [{uncertainty['lower_68']:.6f}, {uncertainty['upper_68']:.6f}]")
        print(f"95% credible interval: [{uncertainty['lower_95']:.6f}, {uncertainty['upper_95']:.6f}]")
        
        # Verbose mode: print per-event residuals
        if verbose:
            print("\nPer-Event Residuals:")
            print(f"{'Event ID':<20} {'dt_obs (s)':<12} {'dt_model (s)':<12} {'Residual (s)':<12} {'Sigma (s)':<12}")
            print("-" * 80)
            
            for event in valid_events:
                residual_info = compute_event_residual(event, best_eps0, k_eps)
                
                if 'error' in residual_info:
                    print(f"{residual_info['event_id']:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
                else:
                    event_id = residual_info['event_id']
                    dt_obs = residual_info['dt_obs']
                    dt_model = residual_info['dt_model']
                    residual = residual_info['residual']
                    sigma_tot = residual_info['sigma_tot']
                    
                    print(f"{event_id:<20} {dt_obs:<12.6f} {dt_model:<12.6f} {residual:<12.6f} {sigma_tot:<12.6f}")
            
            print()
        
        # Create output directory
        output_path = Path(output_dir)
        
        # Write JSON output with error handling
        log_info(f"Writing outputs to {output_dir}")
        try:
            write_json_output(
                output_path,
                best_eps0,
                uncertainty,
                len(valid_events),
                [eps_min, eps_max],
                steps,
                k_eps
            )
            
            json_output = output_path / 'eps0_posterior.json'
            print(f"\nResults written to {json_output}")
        
        except PermissionError as e:
            print(f"Error: Permission denied writing output files to {output_dir}")
            log_error(f"Permission denied writing JSON output: {e}")
            if debug:
                raise
            sys.exit(1)
        
        except OSError as e:
            print(f"Error: Failed to write output files (disk full or I/O error)")
            log_error(f"I/O error writing JSON output: {e}")
            if debug:
                raise
            sys.exit(1)
        
        # Compute runtime
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        # Pre-compute residuals once for both text + HTML reports
        residuals = compute_event_residuals(valid_events, best_eps0, k_eps)
        
        # Write text report with error handling
        try:
            write_text_report(
                output_path,
                best_eps0,
                uncertainty,
                valid_events,
                [eps_min, eps_max],
                steps,
                k_eps,
                runtime,
                residuals=residuals,
            )
            
            report_output = output_path / 'rigidity_report.txt'
            print(f"Report written to {report_output}")
        
        except PermissionError as e:
            print(f"Warning: Permission denied writing report file")
            log_error(f"Permission denied writing text report: {e}")
            # Don't exit, JSON output was successful
        
        except OSError as e:
            print(f"Warning: Failed to write report file (disk full or I/O error)")
            log_error(f"I/O error writing text report: {e}")
            # Don't exit, JSON output was successful

        # Write HTML report
        try:
            write_html_report(
                output_path,
                best_eps0,
                uncertainty,
                valid_events,
                [eps_min, eps_max],
                steps,
                k_eps,
                runtime,
                residuals=residuals,
            )
            html_output = output_path / 'rigidity_report.html'
            print(f"HTML report written to {html_output}")
        except PermissionError as e:
            print("Warning: Permission denied writing HTML report")
            log_error(f"Permission denied writing HTML report: {e}")
        except OSError as e:
            print("Warning: Failed to write HTML report (disk full or I/O error)")
            log_error(f"I/O error writing HTML report: {e}")
        
        # Log runtime summary
        log_runtime_summary(len(valid_events), runtime)
        print(f"Total runtime: {runtime:.1f}s")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        log_error("Analysis interrupted by user (KeyboardInterrupt)")
        sys.exit(130)  # Standard exit code for SIGINT
    
    except Exception as e:
        log_error(f"Fatal error: {e}", exception=e if debug else None)
        if debug:
            raise
        else:
            print(f"Error: {e}")
            print("Run with --debug for full traceback")
            sys.exit(1)


if __name__ == '__main__':
    main()
