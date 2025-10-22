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
    
    def _convert_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert string value to appropriate type."""
        # Handle boolean values
        if value.lower() in ['true', 'yes', 'on', '1']:
            return True
        elif value.lower() in ['false', 'no', 'off', '0']:
            return False
        
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
        return self.config_data.get('datasets', {})
    
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
    "custom_covariance": null
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
"""