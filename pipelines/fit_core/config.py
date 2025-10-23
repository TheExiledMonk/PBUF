"""
Configuration file support and advanced options for PBUF cosmology pipeline.

This module provides configuration file parsing, default parameter management,
and advanced options for optimizer selection and output formatting.
"""

import json
import configparser
from typing import Dict, Any, Optional, Union
from pathlib import Path
import os

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ConfigurationManager:
    """
    Manages configuration files and advanced options for the PBUF pipeline.
    
    Supports multiple configuration file formats (JSON, YAML, INI) and provides
    a unified interface for parameter overrides, optimizer settings, and output options.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_data = {}
        self.config_file = config_file
        
        # Load configuration if file provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Supports JSON, YAML, and INI formats based on file extension.
        
        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Determine format from extension
        extension = config_path.suffix.lower()
        
        try:
            if extension == '.json':
                self._load_json(config_path)
            elif extension in ['.yaml', '.yml']:
                self._load_yaml(config_path)
            elif extension in ['.ini', '.cfg']:
                self._load_ini(config_path)
            else:
                # Try to auto-detect format
                self._auto_detect_format(config_path)
        
        except Exception as e:
            raise ValueError(f"Error loading configuration file {config_file}: {e}")
    
    def _load_json(self, config_path: Path) -> None:
        """Load JSON configuration file."""
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)
    
    def _load_yaml(self, config_path: Path) -> None:
        """Load YAML configuration file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML configuration files. Install with: pip install PyYAML")
        
        with open(config_path, 'r') as f:
            self.config_data = yaml.safe_load(f)
    
    def _load_ini(self, config_path: Path) -> None:
        """Load INI configuration file."""
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Convert to nested dictionary
        self.config_data = {}
        for section_name in config.sections():
            section = {}
            for key, value in config[section_name].items():
                # Try to convert to appropriate type
                section[key] = self._convert_value(value)
            self.config_data[section_name] = section
    
    def _auto_detect_format(self, config_path: Path) -> None:
        """Auto-detect configuration file format."""
        with open(config_path, 'r') as f:
            content = f.read().strip()
        
        # Try JSON first
        try:
            self.config_data = json.loads(content)
            return
        except json.JSONDecodeError:
            pass
        
        # Try YAML
        if HAS_YAML:
            try:
                self.config_data = yaml.safe_load(content)
                return
            except:
                pass
        
        # Fall back to treating as INI
        config = configparser.ConfigParser()
        config.read_string(content)
        self._load_ini_from_parser(config)
    
    def _load_ini_from_parser(self, config: configparser.ConfigParser) -> None:
        """Load configuration from ConfigParser object."""
        self.config_data = {}
        for section_name in config.sections():
            section = {}
            for key, value in config[section_name].items():
                section[key] = self._convert_value(value)
            self.config_data[section_name] = section
    
    def _convert_value(self, value: str) -> Union[str, int, float, bool, list]:
        """Convert string value to appropriate type."""
        # Handle boolean values
        if value.lower() in ['true', 'yes', 'on', '1']:
            return True
        elif value.lower() in ['false', 'no', 'off', '0']:
            return False
        
        # Handle comma-separated lists (for INI files)
        if ',' in value:
            # Split and strip whitespace
            items = [item.strip() for item in value.split(',')]
            # Try to convert each item
            converted_items = []
            for item in items:
                if item.lower() in ['true', 'yes', 'on', '1']:
                    converted_items.append(True)
                elif item.lower() in ['false', 'no', 'off', '0']:
                    converted_items.append(False)
                else:
                    try:
                        if '.' in item or 'e' in item.lower():
                            converted_items.append(float(item))
                        else:
                            converted_items.append(int(item))
                    except ValueError:
                        converted_items.append(item)
            return converted_items
        
        # Try numeric conversion
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    
    def get_parameter_overrides(self) -> Dict[str, Any]:
        """
        Get parameter overrides from configuration.
        
        Returns:
            Dictionary of parameter overrides
        """
        return self.config_data.get('parameters', {})
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration.
        
        Returns:
            Dictionary with optimizer settings
        """
        optimizer_section = self.config_data.get('optimizer', {})
        
        # Default optimizer configuration
        default_config = {
            'method': 'minimize',
            'algorithm': 'L-BFGS-B',
            'options': {
                'maxiter': 1000,
                'ftol': 1e-9,
                'gtol': 1e-9
            }
        }
        
        # Merge with user configuration
        config = default_config.copy()
        config.update(optimizer_section)
        
        return config
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output configuration.
        
        Returns:
            Dictionary with output settings
        """
        output_section = self.config_data.get('output', {})
        
        # Default output configuration
        default_config = {
            'format': 'human',
            'precision': 6,
            'save_results': False,
            'results_file': None,
            'include_diagnostics': True
        }
        
        # Merge with user configuration
        config = default_config.copy()
        config.update(output_section)
        
        return config
    
    def get_integrity_config(self) -> Dict[str, Any]:
        """
        Get integrity check configuration.
        
        Returns:
            Dictionary with integrity check settings
        """
        integrity_section = self.config_data.get('integrity', {})
        
        # Default integrity configuration
        default_config = {
            'enabled': False,
            'tolerances': {
                'h_ratios': 1e-4,
                'recombination': 1e-4,
                'sound_horizon': 1e-4,
                'covariance_eigenvalues': 1e-12
            },
            'fail_on_error': False
        }
        
        # Merge with user configuration
        config = default_config.copy()
        if integrity_section:
            config.update(integrity_section)
            # Merge tolerances separately to preserve defaults
            if 'tolerances' in integrity_section:
                config['tolerances'].update(integrity_section['tolerances'])
        
        return config
    
    def get_datasets_config(self) -> Dict[str, Any]:
        """
        Get dataset configuration.
        
        Returns:
            Dictionary with dataset settings
        """
        datasets_config = self.config_data.get('datasets', {})
        
        # Add default dataset registry configuration if not present
        if 'registry' not in datasets_config:
            datasets_config['registry'] = {
                'enabled': True,
                'manifest_path': 'data/datasets_manifest.json',
                'registry_path': 'data/registry/',
                'cache_path': 'data/cache/',
                'auto_fetch': True,
                'verify_on_load': True,
                'fallback_to_legacy': False
            }
        
        return datasets_config
    
    def get_dataset_registry_config(self) -> Dict[str, Any]:
        """
        Get dataset registry configuration.
        
        Returns:
            Dictionary with dataset registry settings
        """
        datasets_config = self.get_datasets_config()
        registry_config = datasets_config.get('registry', {})
        
        # Default registry configuration
        default_config = {
            'enabled': True,
            'manifest_path': 'data/datasets_manifest.json',
            'registry_path': 'data/registry/',
            'cache_path': 'data/cache/',
            'auto_fetch': True,
            'verify_on_load': True,
            'fallback_to_legacy': False,
            'download': {
                'timeout': 300,
                'max_retries': 3,
                'retry_delay': 1.0,
                'max_retry_delay': 60.0,
                'retry_backoff_factor': 2.0,
                'chunk_size': 8192,
                'concurrent_downloads': 3
            },
            'verification': {
                'verify_checksums': True,
                'verify_file_sizes': True,
                'verify_schemas': True,
                'checksum_algorithm': 'sha256'
            },
            'logging': {
                'structured_logging': True,
                'log_level': 'INFO',
                'log_file': None,
                'audit_trail_enabled': True
            },
            'cache': {
                'enabled': True,
                'max_size_gb': 10.0
            }
        }
        
        # Merge with user configuration
        config = default_config.copy()
        
        # Handle nested configuration merging
        for key, value in registry_config.items():
            if key in ['download', 'verification', 'logging', 'cache'] and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        
        return config
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """
        Get optimization configuration.
        
        Returns:
            Dictionary with optimization settings
        """
        optimization_section = self.config_data.get('optimization', {})
        
        # Default optimization configuration
        default_config = {
            'optimize_parameters': [],
            'frozen_parameters': [],
            'use_precomputed': True,
            'save_results': True,
            'convergence_tolerance': 1e-6,
            'covariance_scaling': 1.0,
            'warm_start': False,
            'dry_run': False
        }
        
        # Merge with user configuration
        config = default_config.copy()
        config.update(optimization_section)
        
        # Validate optimization parameters
        self._validate_optimization_config(config)
        
        return config
    
    def _validate_optimization_config(self, config: Dict[str, Any]) -> None:
        """
        Validate optimization configuration parameters.
        
        Args:
            config: Optimization configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate optimize_parameters is a list
        if not isinstance(config['optimize_parameters'], list):
            raise ValueError("optimize_parameters must be a list")
        
        # Validate frozen_parameters is a list
        if not isinstance(config['frozen_parameters'], list):
            raise ValueError("frozen_parameters must be a list")
        
        # Validate covariance_scaling is a positive number
        if not isinstance(config['covariance_scaling'], (int, float)) or config['covariance_scaling'] <= 0:
            raise ValueError("covariance_scaling must be a positive number")
        
        # Validate convergence_tolerance is a positive number
        if not isinstance(config['convergence_tolerance'], (int, float)) or config['convergence_tolerance'] <= 0:
            raise ValueError("convergence_tolerance must be a positive number")
        
        # Validate boolean flags
        for flag in ['use_precomputed', 'save_results', 'warm_start', 'dry_run']:
            if not isinstance(config[flag], bool):
                raise ValueError(f"{flag} must be a boolean")
        
        # Check for conflicts between optimize_parameters and frozen_parameters
        optimize_set = set(config['optimize_parameters'])
        frozen_set = set(config['frozen_parameters'])
        conflicts = optimize_set.intersection(frozen_set)
        if conflicts:
            raise ValueError(f"Parameters cannot be both optimized and frozen: {list(conflicts)}")
        
        # Validate parameter names (basic check for common parameters)
        # Import parameter definitions to get actual valid parameters
        try:
            from .parameter import DEFAULTS
            valid_lcdm_params = set(DEFAULTS.get('lcdm', {}).keys())
            valid_pbuf_params = set(DEFAULTS.get('pbuf', {}).keys())
            all_valid_params = valid_lcdm_params.union(valid_pbuf_params)
        except ImportError:
            # Fallback to hardcoded list if import fails
            valid_lcdm_params = {'H0', 'Om0', 'Obh2', 'ns', 'Neff', 'Tcmb'}
            valid_pbuf_params = {'H0', 'Om0', 'Obh2', 'ns', 'Neff', 'Tcmb', 'alpha', 'Rmax', 'eps0', 'n_eps', 'k_sat'}
            all_valid_params = valid_lcdm_params.union(valid_pbuf_params)
        
        for param in config['optimize_parameters']:
            if param not in all_valid_params:
                # Issue warning but don't fail - model-specific validation will happen later
                print(f"Warning: Unknown parameter '{param}' in optimize_parameters. "
                      f"Valid parameters include: {sorted(all_valid_params)}")
        
        for param in config['frozen_parameters']:
            if param not in all_valid_params:
                print(f"Warning: Unknown parameter '{param}' in frozen_parameters. "
                      f"Valid parameters include: {sorted(all_valid_params)}")
    
    def save_config(self, output_file: str, format: str = 'json') -> None:
        """
        Save current configuration to file.
        
        Args:
            output_file: Path to output file
            format: Output format ('json', 'yaml', or 'ini')
        """
        output_path = Path(output_file)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.config_data, f, indent=2)
        
        elif format in ['yaml', 'yml']:
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML output. Install with: pip install PyYAML")
            
            with open(output_path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
        
        elif format in ['ini', 'cfg']:
            config = configparser.ConfigParser()
            
            # Convert nested dictionary to INI format
            for section_name, section_data in self.config_data.items():
                config.add_section(section_name)
                for key, value in section_data.items():
                    config.set(section_name, key, str(value))
            
            with open(output_path, 'w') as f:
                config.write(f)
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def create_example_config(self, output_file: str, format: str = 'json') -> None:
        """
        Create an example configuration file with all available options.
        
        Args:
            output_file: Path to output file
            format: Output format ('json', 'yaml', or 'ini')
        """
        example_config = {
            'parameters': {
                'H0': 67.4,
                'Om0': 0.315,
                'Obh2': 0.02237,
                'ns': 0.9649,
                'alpha': 5e-4,
                'Rmax': 1e9,
                'eps0': 0.7,
                'n_eps': 0.0,
                'k_sat': 0.9762
            },
            'optimization': {
                'optimize_parameters': ['k_sat', 'alpha'],
                'frozen_parameters': ['Neff', 'Tcmb'],
                'use_precomputed': True,
                'save_results': True,
                'convergence_tolerance': 1e-6,
                'covariance_scaling': 1.0,
                'warm_start': False,
                'dry_run': False
            },
            'optimizer': {
                'method': 'minimize',
                'algorithm': 'L-BFGS-B',
                'options': {
                    'maxiter': 1000,
                    'ftol': 1e-9,
                    'gtol': 1e-9
                }
            },
            'output': {
                'format': 'human',
                'precision': 6,
                'save_results': True,
                'results_file': 'results.json',
                'include_diagnostics': True
            },
            'integrity': {
                'enabled': True,
                'tolerances': {
                    'h_ratios': 1e-4,
                    'recombination': 1e-4,
                    'sound_horizon': 1e-4,
                    'covariance_eigenvalues': 1e-12
                },
                'fail_on_error': False
            },
            'datasets': {
                'default_datasets': ['cmb', 'bao', 'sn'],
                'data_directory': './data',
                'custom_covariance': None
            }
        }
        
        # Temporarily set config data and save
        original_config = self.config_data
        self.config_data = example_config
        self.save_config(output_file, format)
        self.config_data = original_config


def load_configuration(config_file: Optional[str] = None) -> ConfigurationManager:
    """
    Load configuration from file or create default configuration.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        ConfigurationManager instance
    """
    return ConfigurationManager(config_file)


def add_optimization_arguments(parser: 'argparse.ArgumentParser') -> None:
    """
    Add optimization-related command-line arguments to an argument parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    # Optimization control
    parser.add_argument(
        "--optimize",
        type=str,
        help="Comma-separated list of parameters to optimize (e.g., 'k_sat,alpha' or 'H0,Om0')"
    )
    
    parser.add_argument(
        "--cov-scale",
        type=float,
        default=1.0,
        help="Covariance scaling factor for optimization (default: 1.0)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform optimization without saving results to parameter store"
    )
    
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Use recent optimization results as starting point if available"
    )


def parse_optimization_parameters(optimize_str: Optional[str]) -> list:
    """
    Parse optimization parameter string into list.
    
    Args:
        optimize_str: Comma-separated parameter string or None
        
    Returns:
        List of parameter names to optimize
    """
    if not optimize_str:
        return []
    
    # Split by comma and strip whitespace
    params = [param.strip() for param in optimize_str.split(',')]
    
    # Filter out empty strings
    params = [param for param in params if param]
    
    return params


def merge_optimization_config(
    config_optimization: Dict[str, Any],
    cli_args: 'argparse.Namespace'
) -> Dict[str, Any]:
    """
    Merge optimization configuration from config file and command-line arguments.
    Command-line arguments take precedence over config file settings.
    
    Args:
        config_optimization: Optimization config from configuration file
        cli_args: Parsed command-line arguments
        
    Returns:
        Merged optimization configuration
    """
    # Start with config file settings
    merged_config = config_optimization.copy()
    
    # Override with command-line arguments if provided
    if hasattr(cli_args, 'optimize') and cli_args.optimize:
        merged_config['optimize_parameters'] = parse_optimization_parameters(cli_args.optimize)
    
    if hasattr(cli_args, 'cov_scale') and cli_args.cov_scale != 1.0:
        merged_config['covariance_scaling'] = cli_args.cov_scale
    
    if hasattr(cli_args, 'dry_run') and cli_args.dry_run:
        merged_config['dry_run'] = True
    
    if hasattr(cli_args, 'warm_start') and cli_args.warm_start:
        merged_config['warm_start'] = True
    
    return merged_config


def find_config_file() -> Optional[str]:
    """
    Search for configuration file in standard locations.
    
    Searches in order:
    1. ./pbuf_config.json
    2. ./pbuf_config.yaml
    3. ./pbuf_config.ini
    4. ~/.pbuf_config.json
    5. ~/.pbuf_config.yaml
    6. ~/.pbuf_config.ini
    
    Returns:
        Path to first found configuration file, or None
    """
    # Standard config file names
    config_names = [
        'pbuf_config.json',
        'pbuf_config.yaml',
        'pbuf_config.yml',
        'pbuf_config.ini',
        'pbuf_config.cfg'
    ]
    
    # Search locations
    search_paths = [
        Path('.'),  # Current directory
        Path.home()  # Home directory
    ]
    
    for search_path in search_paths:
        for config_name in config_names:
            config_file = search_path / config_name
            if config_file.exists():
                return str(config_file)
    
    return None


# Example configuration templates
EXAMPLE_JSON_CONFIG = """
{
  "parameters": {
    "H0": 67.4,
    "Om0": 0.315,
    "Obh2": 0.02237,
    "ns": 0.9649,
    "alpha": 5e-4,
    "Rmax": 1e9,
    "eps0": 0.7,
    "n_eps": 0.0,
    "k_sat": 0.9762
  },
  "optimization": {
    "optimize_parameters": ["k_sat", "alpha"],
    "frozen_parameters": ["Neff", "Tcmb"],
    "use_precomputed": true,
    "save_results": true,
    "convergence_tolerance": 1e-6,
    "covariance_scaling": 1.0,
    "warm_start": false,
    "dry_run": false
  },
  "optimizer": {
    "method": "minimize",
    "algorithm": "L-BFGS-B",
    "options": {
      "maxiter": 1000,
      "ftol": 1e-9,
      "gtol": 1e-9
    }
  },
  "output": {
    "format": "human",
    "precision": 6,
    "save_results": true,
    "results_file": "results.json",
    "include_diagnostics": true
  },
  "integrity": {
    "enabled": true,
    "tolerances": {
      "h_ratios": 1e-4,
      "recombination": 1e-4,
      "sound_horizon": 1e-4,
      "covariance_eigenvalues": 1e-12
    },
    "fail_on_error": false
  },
  "datasets": {
    "default_datasets": ["cmb", "bao", "sn"],
    "data_directory": "./data",
    "custom_covariance": null,
    "registry": {
      "enabled": true,
      "manifest_path": "data/datasets_manifest.json",
      "registry_path": "data/registry/",
      "cache_path": "data/cache/",
      "auto_fetch": true,
      "verify_on_load": true,
      "fallback_to_legacy": false,
      "download": {
        "timeout": 300,
        "max_retries": 3,
        "retry_delay": 1.0,
        "max_retry_delay": 60.0,
        "retry_backoff_factor": 2.0,
        "chunk_size": 8192,
        "concurrent_downloads": 3
      },
      "verification": {
        "verify_checksums": true,
        "verify_file_sizes": true,
        "verify_schemas": true,
        "checksum_algorithm": "sha256"
      },
      "logging": {
        "structured_logging": true,
        "log_level": "INFO",
        "log_file": null,
        "audit_trail_enabled": true
      },
      "cache": {
        "enabled": true,
        "max_size_gb": 10.0
      }
    }
  }
}
"""

EXAMPLE_YAML_CONFIG = """
parameters:
  H0: 67.4
  Om0: 0.315
  Obh2: 0.02237
  ns: 0.9649
  alpha: 5.0e-4
  Rmax: 1.0e9
  eps0: 0.7
  n_eps: 0.0
  k_sat: 0.9762

optimization:
  optimize_parameters:
    - k_sat
    - alpha
  frozen_parameters:
    - Neff
    - Tcmb
  use_precomputed: true
  save_results: true
  convergence_tolerance: 1.0e-6
  covariance_scaling: 1.0
  warm_start: false
  dry_run: false

optimizer:
  method: minimize
  algorithm: L-BFGS-B
  options:
    maxiter: 1000
    ftol: 1.0e-9
    gtol: 1.0e-9

output:
  format: human
  precision: 6
  save_results: true
  results_file: results.json
  include_diagnostics: true

integrity:
  enabled: true
  tolerances:
    h_ratios: 1.0e-4
    recombination: 1.0e-4
    sound_horizon: 1.0e-4
    covariance_eigenvalues: 1.0e-12
  fail_on_error: false

datasets:
  default_datasets:
    - cmb
    - bao
    - sn
  data_directory: ./data
  custom_covariance: null
  registry:
    enabled: true
    manifest_path: data/datasets_manifest.json
    registry_path: data/registry/
    cache_path: data/cache/
    auto_fetch: true
    verify_on_load: true
    fallback_to_legacy: false
    download:
      timeout: 300
      max_retries: 3
      retry_delay: 1.0
      max_retry_delay: 60.0
      retry_backoff_factor: 2.0
      chunk_size: 8192
      concurrent_downloads: 3
    verification:
      verify_checksums: true
      verify_file_sizes: true
      verify_schemas: true
      checksum_algorithm: sha256
    logging:
      structured_logging: true
      log_level: INFO
      log_file: null
      audit_trail_enabled: true
    cache:
      enabled: true
      max_size_gb: 10.0
"""

EXAMPLE_INI_CONFIG = """
[parameters]
H0 = 67.4
Om0 = 0.315
Obh2 = 0.02237
ns = 0.9649
alpha = 5e-4
Rmax = 1e9
eps0 = 0.7
n_eps = 0.0
k_sat = 0.9762

[optimization]
optimize_parameters = k_sat,alpha
frozen_parameters = Neff,Tcmb
use_precomputed = true
save_results = true
convergence_tolerance = 1e-6
covariance_scaling = 1.0
warm_start = false
dry_run = false

[optimizer]
method = minimize
algorithm = L-BFGS-B
maxiter = 1000
ftol = 1e-9
gtol = 1e-9

[output]
format = human
precision = 6
save_results = true
results_file = results.json
include_diagnostics = true

[integrity]
enabled = true
h_ratios_tolerance = 1e-4
recombination_tolerance = 1e-4
sound_horizon_tolerance = 1e-4
covariance_eigenvalues_tolerance = 1e-12
fail_on_error = false

[datasets]
default_datasets = cmb,bao,sn
data_directory = ./data

[datasets.registry]
enabled = true
manifest_path = data/datasets_manifest.json
registry_path = data/registry/
cache_path = data/cache/
auto_fetch = true
verify_on_load = true
fallback_to_legacy = false
download_timeout = 300
max_retries = 3
retry_delay = 1.0
max_retry_delay = 60.0
retry_backoff_factor = 2.0
chunk_size = 8192
concurrent_downloads = 3
verify_checksums = true
verify_file_sizes = true
verify_schemas = true
checksum_algorithm = sha256
structured_logging = true
log_level = INFO
audit_trail_enabled = true
cache_enabled = true
cache_max_size_gb = 10.0
"""