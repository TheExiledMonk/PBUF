"""Events module for loading and validating multi-messenger event data."""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from quantum.pipeline.npz_loader import load_real_events_from_npz


# Unit conversion table for common energy units
ENERGY_UNIT_CONVERSIONS = {
    'eV': 1.0,
    'keV': 1e3,
    'MeV': 1e6,
    'GeV': 1e9,
    'TeV': 1e12,
    'PeV': 1e15
}


def load_event(filepath: str) -> Dict[str, Any]:
    """
    Load a single event from JSON or CSV file.
    
    Args:
        filepath: Path to the event file (JSON or CSV)
        
    Returns:
        Dictionary containing event data with standardized structure
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or parsing fails
        PermissionError: If file cannot be read due to permissions
    
    Requirements: 11.1, 11.2
    """
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Event file not found: {filepath}")

        suffix = filepath.suffix.lower()
        if suffix == '.json':
            return _load_json_event(filepath)
        elif suffix == '.csv':
            return _load_csv_event(filepath)
        elif suffix == '.npz':
            return load_real_events_from_npz(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .json or .csv")

    except PermissionError as e:
        raise PermissionError(f"Permission denied reading file {filepath}: {e}")
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Unexpected error loading {filepath}: {e}")


def _load_json_event(filepath: Path) -> Dict[str, Any]:
    """
    Load event from JSON file with error handling.
    
    Requirements: 11.1, 11.2
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [_apply_unit_conversions(item, filepath) for item in data]
        
        return _apply_unit_conversions(data, filepath)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in {filepath} at line {e.lineno}: {e.msg}")
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading {filepath}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading {filepath}: {e}")


def _load_csv_event(filepath: Path) -> Dict[str, Any]:
    """
    Load event from CSV file with error handling.
    
    Expected CSV format:
    event_id,L_Mpc,channel,t_obs,sigma_t,mass_eV,E_eV,lag_mean,lag_sigma
    
    Requirements: 11.1, 11.2
    """
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            raise ValueError("CSV file is empty")
        
        # Extract event metadata from first row
        first_row = rows[0]
        event_id = first_row.get('event_id', '')
        
        try:
            L_Mpc = float(first_row.get('L_Mpc', 0))
            lag_mean = float(first_row.get('lag_mean', 0))
            lag_sigma = float(first_row.get('lag_sigma', 0))
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in CSV header: {e}")
        
        # Build channels dictionary
        channels = {}
        for row_num, row in enumerate(rows, start=2):  # Start at 2 (header is 1)
            channel_name = row.get('channel', '')
            if not channel_name:
                continue
            
            try:
                # Parse E_eV - handle empty string or None
                E_eV_str = row.get('E_eV', '')
                if E_eV_str == '' or E_eV_str is None or E_eV_str.lower() == 'none':
                    E_eV = None
                else:
                    E_eV = float(E_eV_str)
                
                channels[channel_name] = {
                    't_obs': float(row.get('t_obs', 0)),
                    'sigma_t': float(row.get('sigma_t', 0)),
                    'mass_eV': float(row.get('mass_eV', 0)),
                    'E_eV': E_eV
                }
            except ValueError as e:
                raise ValueError(f"Invalid numeric value at row {row_num}, channel '{channel_name}': {e}")
        
        # Construct standardized event structure
        event = {
            'id': event_id,
            'L_Mpc': L_Mpc,
            'channels': channels,
            'intrinsic_lag_model': {
                'mean': lag_mean,
                'sigma': lag_sigma
            }
        }
        
        # Apply unit conversions if needed
        event = _apply_unit_conversions(event, filepath)
        
        return event
        
    except (ValueError, KeyError) as e:
        raise ValueError(f"Error parsing CSV {filepath}: {e}")
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading {filepath}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading CSV {filepath}: {e}")


def _apply_unit_conversions(event: Dict[str, Any], filepath: Path) -> Dict[str, Any]:
    """
    Apply automatic unit conversions for common energy units.
    
    Detects if energy values are in keV, MeV, etc. and converts to eV.
    Issues warnings for mixed units.
    
    Args:
        event: Event dictionary
        filepath: Path to source file (for warning messages)
    
    Returns:
        Event dictionary with energies converted to eV
    
    Requirements: 10.5, 11.1
    """
    # Check if event has unit metadata
    if 'units' in event:
        units = event['units']
        energy_unit = units.get('energy', 'eV')
        
        if energy_unit in ENERGY_UNIT_CONVERSIONS:
            conversion_factor = ENERGY_UNIT_CONVERSIONS[energy_unit]
            
            if conversion_factor != 1.0:
                _log_warn(f"Converting energy units from {energy_unit} to eV in {filepath}")
                
                # Convert channel energies
                for channel_name, channel_data in event.get('channels', {}).items():
                    if 'mass_eV' in channel_data:
                        channel_data['mass_eV'] *= conversion_factor
                    if 'E_eV' in channel_data and channel_data['E_eV'] is not None:
                        channel_data['E_eV'] *= conversion_factor
        else:
            _log_warn(f"Unknown energy unit '{energy_unit}' in {filepath}, assuming eV")
    
    # Check for mixed units by examining field names
    channels = event.get('channels', {})
    detected_units = set()
    
    for channel_name, channel_data in channels.items():
        for field_name in channel_data.keys():
            # Check if field name contains unit suffix
            for unit in ENERGY_UNIT_CONVERSIONS.keys():
                if unit != 'eV' and unit in field_name:
                    detected_units.add(unit)
    
    if detected_units:
        _log_warn(f"Detected mixed energy units in {filepath}: {detected_units}. Ensure all energies are in eV.")
    
    return event


def load_events_from_directory(dirpath: str) -> List[Dict[str, Any]]:
    """
    Load all events from a directory with graceful error handling.
    
    Args:
        dirpath: Path to directory containing event files
        
    Returns:
        List of event dictionaries
        
    Note:
        - Skips files that fail to load (logs errors to logs/errors.txt)
        - Supports both JSON and CSV files
        - Only processes .json and .csv files
        - Continues processing if individual events fail (graceful degradation)
    
    Requirements: 11.1, 11.2, 11.5
    """
    try:
        dirpath = Path(dirpath)
        
        if not dirpath.exists():
            _log_error(f"Directory not found: {dirpath}")
            return []
        
        if not dirpath.is_dir():
            _log_error(f"Path is not a directory: {dirpath}")
            return []
    
    except PermissionError as e:
        _log_error(f"Permission denied accessing directory {dirpath}: {e}")
        return []
    except Exception as e:
        _log_error(f"Error accessing directory {dirpath}: {e}")
        return []
    
    events = []
    failed_count = 0
    
    try:
        # Find all JSON and CSV files
        event_files = list(dirpath.glob('*.json')) + list(dirpath.glob('*.csv'))
        
        if not event_files:
            _log_warn(f"No event files (.json or .csv) found in {dirpath}")
            return []
        
        for filepath in event_files:
            try:
                event = load_event(str(filepath))
                if isinstance(event, list):
                    events.extend(event)
                else:
                    events.append(event)
            except FileNotFoundError as e:
                _log_error(f"File not found: {filepath}")
                failed_count += 1
            except PermissionError as e:
                _log_error(f"Permission denied reading {filepath}: {e}")
                failed_count += 1
            except ValueError as e:
                _log_error(f"Failed to parse {filepath}: {e}")
                failed_count += 1
            except Exception as e:
                _log_error(f"Unexpected error loading {filepath}: {e}")
                failed_count += 1
        
        if failed_count > 0:
            _log_warn(f"Failed to load {failed_count} out of {len(event_files)} event files")
    
    except Exception as e:
        _log_error(f"Error scanning directory {dirpath}: {e}")
    
    return events


def validate_event(event: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate event structure and data.
    
    Args:
        event: Event dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if event passes all validation checks
        - error_message: Empty string if valid, otherwise description of error
    """
    # Check required top-level fields
    if 'id' not in event:
        return False, "Missing required field: 'id'"
    
    if 'L_Mpc' not in event:
        return False, "Missing required field: 'L_Mpc'"
    
    if 'channels' not in event:
        return False, "Missing required field: 'channels'"
    
    if 'intrinsic_lag_model' not in event:
        return False, "Missing required field: 'intrinsic_lag_model'"
    
    # Validate L_Mpc
    try:
        L_Mpc = float(event['L_Mpc'])
        if L_Mpc <= 0:
            return False, f"L_Mpc must be positive, got {L_Mpc}"
    except (ValueError, TypeError):
        return False, f"L_Mpc must be a number, got {event['L_Mpc']}"
    
    # Validate channels
    channels = event['channels']
    if not isinstance(channels, dict):
        return False, "channels must be a dictionary"
    
    if len(channels) < 2:
        return False, f"Event must have at least 2 channels, got {len(channels)}"
    
    # Validate each channel
    for channel_name, channel_data in channels.items():
        if not isinstance(channel_data, dict):
            return False, f"Channel '{channel_name}' data must be a dictionary"
        
        # Check required channel fields
        required_fields = ['t_obs', 'sigma_t', 'mass_eV']
        for field in required_fields:
            if field not in channel_data:
                return False, f"Channel '{channel_name}' missing required field: '{field}'"
        
        # Validate sigma_t is positive
        try:
            sigma_t = float(channel_data['sigma_t'])
            if sigma_t <= 0:
                return False, f"Channel '{channel_name}' sigma_t must be positive, got {sigma_t}"
        except (ValueError, TypeError):
            return False, f"Channel '{channel_name}' sigma_t must be a number"
        
        # Validate mass_eV
        try:
            mass_eV = float(channel_data['mass_eV'])
            if mass_eV < 0:
                return False, f"Channel '{channel_name}' mass_eV cannot be negative, got {mass_eV}"
        except (ValueError, TypeError):
            return False, f"Channel '{channel_name}' mass_eV must be a number"
        
        # Validate E_eV for massive particles
        if mass_eV > 0:
            if 'E_eV' not in channel_data:
                return False, f"Channel '{channel_name}' with mass > 0 must have E_eV field"
            
            E_eV = channel_data['E_eV']
            if E_eV is None:
                return False, f"Channel '{channel_name}' with mass > 0 must have non-null E_eV"
            
            try:
                E_eV = float(E_eV)
                if E_eV <= mass_eV:
                    return False, f"Channel '{channel_name}' E_eV ({E_eV}) must be greater than mass_eV ({mass_eV})"
            except (ValueError, TypeError):
                return False, f"Channel '{channel_name}' E_eV must be a number"
    
    # Validate intrinsic_lag_model
    lag_model = event['intrinsic_lag_model']
    if not isinstance(lag_model, dict):
        return False, "intrinsic_lag_model must be a dictionary"
    
    required_lag_fields = ['mean', 'sigma']
    for field in required_lag_fields:
        if field not in lag_model:
            return False, f"intrinsic_lag_model missing required field: '{field}'"
    
    # Validate lag model values are numeric
    try:
        float(lag_model['mean'])
        float(lag_model['sigma'])
    except (ValueError, TypeError):
        return False, "intrinsic_lag_model mean and sigma must be numbers"
    
    # Validate lag sigma is non-negative
    try:
        lag_sigma = float(lag_model['sigma'])
        if lag_sigma < 0:
            return False, f"intrinsic_lag_model sigma must be non-negative, got {lag_sigma}"
    except (ValueError, TypeError):
        return False, "intrinsic_lag_model sigma must be a number"

    if 'likelihood_channels' in event:
        likelihood = event['likelihood_channels']
        if not isinstance(likelihood, (list, tuple)):
            return False, "likelihood_channels must be provided as a list"
        if len(likelihood) < 2:
            return False, "likelihood_channels must reference at least two channels"
        for name in likelihood:
            if name not in channels:
                return False, f"likelihood channel '{name}' not found in channels"
    
    return True, ""


def _log_error(message: str) -> None:
    """
    Log error message to logs/errors.txt with timestamp.
    
    Args:
        message: Error message to log
    
    Requirements: 11.2
    """
    # Import logger module
    try:
        from .logger import log_error
        log_error(message)
    except ImportError:
        # Fallback if logger module not available
        from datetime import datetime
        logs_dir = Path('logs')
        try:
            logs_dir.mkdir(exist_ok=True)
        except Exception:
            pass
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] ERROR: {message}\n"
        error_file = logs_dir / 'errors.txt'
        try:
            with open(error_file, 'a') as f:
                f.write(log_entry)
        except Exception:
            pass


def _log_warn(message: str) -> None:
    """
    Log warning message to logs/run.log with timestamp.
    
    Args:
        message: Warning message to log
    
    Requirements: 10.5, 11.1
    """
    # Import logger module
    try:
        from .logger import log_warn
        log_warn(message)
    except ImportError:
        # Fallback if logger module not available
        from datetime import datetime
        logs_dir = Path('logs')
        try:
            logs_dir.mkdir(exist_ok=True)
        except Exception:
            pass
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] WARN: {message}\n"
        log_file = logs_dir / 'run.log'
        try:
            with open(log_file, 'a') as f:
                f.write(log_entry)
        except Exception:
            pass
