"""
Configuration integration for CMB raw parameter processing.

This module provides configuration management for the CMB derivation pipeline,
including integration with the data preparation framework configuration system,
environment variable support, and deployment-specific settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

try:
    from ..derivation.cmb_models import CMBConfig
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from derivation.cmb_models import CMBConfig


@dataclass
class DataPreparationConfig:
    """
    Configuration settings for the data preparation framework.
    
    This extends the framework configuration to include CMB-specific settings
    while maintaining compatibility with existing configuration patterns.
    """
    
    # Core framework settings
    framework_enabled: bool = True
    output_path: str = "data/prepared/"
    cache_path: str = "data/cache/"
    
    # Processing behavior
    auto_process: bool = True
    validate_outputs: bool = True
    fallback_to_legacy: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # Logging and monitoring
    structured_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    performance_monitoring: bool = False
    
    # Module-specific configurations
    cmb: CMBConfig = field(default_factory=CMBConfig.get_default_config)
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "framework_enabled": self.framework_enabled,
            "output_path": self.output_path,
            "cache_path": self.cache_path,
            "auto_process": self.auto_process,
            "validate_outputs": self.validate_outputs,
            "fallback_to_legacy": self.fallback_to_legacy,
            "performance": {
                "parallel_processing": self.parallel_processing,
                "max_workers": self.max_workers,
                "memory_limit_gb": self.memory_limit_gb
            },
            "logging": {
                "structured_logging": self.structured_logging,
                "log_level": self.log_level,
                "log_file": self.log_file,
                "performance_monitoring": self.performance_monitoring
            },
            "derivation": {
                "cmb": self.cmb.to_dict()
            },
            "environment_overrides": self.environment_overrides
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataPreparationConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update top-level settings
        for key in ["framework_enabled", "output_path", "cache_path",
                   "auto_process", "validate_outputs", "fallback_to_legacy"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # Update performance settings
        if "performance" in config_dict:
            perf_config = config_dict["performance"]
            for key in ["parallel_processing", "max_workers", "memory_limit_gb"]:
                if key in perf_config:
                    setattr(config, key, perf_config[key])
        
        # Update logging settings
        if "logging" in config_dict:
            logging_config = config_dict["logging"]
            for key in ["structured_logging", "log_level", "log_file", "performance_monitoring"]:
                if key in logging_config:
                    setattr(config, key, logging_config[key])
        
        # Update CMB configuration
        if "derivation" in config_dict and "cmb" in config_dict["derivation"]:
            config.cmb = CMBConfig.from_dict(config_dict["derivation"]["cmb"])
        elif "cmb" in config_dict:
            config.cmb = CMBConfig.from_dict(config_dict["cmb"])
        
        # Update environment overrides
        if "environment_overrides" in config_dict:
            config.environment_overrides = config_dict["environment_overrides"]
        
        return config


class DataPreparationConfigManager:
    """Manages data preparation configuration with CMB integration."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file
        self._config = DataPreparationConfig()
        self._loaded_from_file = False
        
        # Load configuration if file provided or found
        if config_file:
            self.load_config(config_file)
        else:
            # Try to find configuration file automatically
            auto_config_file = self._find_config_file()
            if auto_config_file:
                self.load_config(auto_config_file)
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract data preparation configuration
            if "data_preparation" in config_data:
                prep_config = config_data["data_preparation"]
            elif "preparation" in config_data:
                prep_config = config_data["preparation"]
            else:
                # Assume the entire file is data preparation config
                prep_config = config_data
            
            self._config = DataPreparationConfig.from_dict(prep_config)
            self._loaded_from_file = True
            self.config_file = config_file
            
        except Exception as e:
            raise ValueError(f"Error loading configuration file {config_file}: {e}")
    
    def _find_config_file(self) -> Optional[str]:
        """
        Search for configuration file in standard locations.
        
        Returns:
            Path to first found configuration file, or None
        """
        # Standard config file names
        config_names = [
            'data_preparation_config.json',
            'pbuf_config.json',
            '.data_preparation_config.json'
        ]
        
        # Search locations
        search_paths = [
            Path('.'),  # Current directory
            Path('config/'),  # Config directory
            Path('.kiro/config/'),  # Kiro config directory
            Path('pipelines/data_preparation/'),  # Module directory
            Path.home(),  # Home directory
            Path.home() / '.config' / 'pbuf'  # XDG config directory
        ]
        
        for search_path in search_paths:
            for config_name in config_names:
                config_file = search_path / config_name
                if config_file.exists():
                    return str(config_file)
        
        return None
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides."""
        # Check for environment variables
        env_overrides = {}
        
        # Framework paths
        if "PBUF_OUTPUT_PATH" in os.environ:
            env_overrides["output_path"] = os.environ["PBUF_OUTPUT_PATH"]
        
        if "PBUF_CACHE_PATH" in os.environ:
            env_overrides["cache_path"] = os.environ["PBUF_CACHE_PATH"]
        
        # Framework behavior
        if "PBUF_FRAMEWORK_ENABLED" in os.environ:
            env_overrides["framework_enabled"] = os.environ["PBUF_FRAMEWORK_ENABLED"].lower() in ["true", "1", "yes"]
        
        if "PBUF_AUTO_PROCESS" in os.environ:
            env_overrides["auto_process"] = os.environ["PBUF_AUTO_PROCESS"].lower() in ["true", "1", "yes"]
        
        if "PBUF_VALIDATE_OUTPUTS" in os.environ:
            env_overrides["validate_outputs"] = os.environ["PBUF_VALIDATE_OUTPUTS"].lower() in ["true", "1", "yes"]
        
        # Performance settings
        if "PBUF_PARALLEL_PROCESSING" in os.environ:
            env_overrides["parallel_processing"] = os.environ["PBUF_PARALLEL_PROCESSING"].lower() in ["true", "1", "yes"]
        
        if "PBUF_MAX_WORKERS" in os.environ:
            try:
                env_overrides["max_workers"] = int(os.environ["PBUF_MAX_WORKERS"])
            except ValueError:
                pass
        
        if "PBUF_MEMORY_LIMIT_GB" in os.environ:
            try:
                env_overrides["memory_limit_gb"] = float(os.environ["PBUF_MEMORY_LIMIT_GB"])
            except ValueError:
                pass
        
        # Logging
        if "PBUF_LOG_LEVEL" in os.environ:
            env_overrides["log_level"] = os.environ["PBUF_LOG_LEVEL"]
        
        if "PBUF_LOG_FILE" in os.environ:
            env_overrides["log_file"] = os.environ["PBUF_LOG_FILE"]
        
        if "PBUF_PERFORMANCE_MONITORING" in os.environ:
            env_overrides["performance_monitoring"] = os.environ["PBUF_PERFORMANCE_MONITORING"].lower() in ["true", "1", "yes"]
        
        # Apply overrides
        for key, value in env_overrides.items():
            setattr(self._config, key, value)
        
        # Update CMB configuration from environment
        self._config.cmb = self._config.cmb.update_from_environment()
        
        # Apply configured environment overrides
        current_env = os.environ.get("PBUF_ENV", "default")
        if current_env in self._config.environment_overrides:
            env_config = self._config.environment_overrides[current_env]
            for key, value in env_config.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                elif key == "cmb" and isinstance(value, dict):
                    # Update CMB configuration
                    cmb_config_dict = self._config.cmb.to_dict()
                    cmb_config_dict.update(value)
                    self._config.cmb = CMBConfig.from_dict(cmb_config_dict)
    
    def get_config(self) -> DataPreparationConfig:
        """Get the current configuration."""
        return self._config
    
    def get_cmb_config(self) -> CMBConfig:
        """Get the CMB-specific configuration."""
        return self._config.cmb
    
    def save_config(self, output_file: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_file: Optional path to output file. If not provided, uses the loaded config file.
        """
        if not output_file:
            if not self.config_file:
                raise ValueError("No output file specified and no config file was loaded")
            output_file = self.config_file
        
        config_dict = {"data_preparation": self._config.to_dict()}
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def create_example_config(self, output_file: str) -> None:
        """
        Create an example configuration file with all available options.
        
        Args:
            output_file: Path to output file
        """
        example_config = DataPreparationConfig()
        
        # Set example CMB configuration
        example_config.cmb = CMBConfig.get_development_config()
        
        # Add some example environment overrides
        example_config.environment_overrides = {
            "development": {
                "log_level": "DEBUG",
                "validate_outputs": False,
                "performance_monitoring": True,
                "cmb": {
                    "cache_computations": False,
                    "performance_monitoring": True,
                    "jacobian_step_size": 1e-5
                }
            },
            "production": {
                "log_level": "WARNING",
                "validate_outputs": True,
                "performance_monitoring": False,
                "parallel_processing": True,
                "cmb": {
                    "cache_computations": True,
                    "performance_monitoring": False,
                    "jacobian_step_size": 1e-6
                }
            },
            "testing": {
                "framework_enabled": False,
                "fallback_to_legacy": True,
                "auto_process": False,
                "cmb": {
                    "use_raw_parameters": False,
                    "fallback_to_legacy": True
                }
            }
        }
        
        config_dict = {"data_preparation": example_config.to_dict()}
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate paths exist or can be created
        for path_attr in ["output_path", "cache_path"]:
            path_value = getattr(self._config, path_attr)
            path_obj = Path(path_value)
            
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation_results["errors"].append(
                    f"Cannot create directory for {path_attr}: {e}"
                )
                validation_results["valid"] = False
        
        # Validate numeric settings
        if self._config.max_workers <= 0:
            validation_results["errors"].append("max_workers must be positive")
            validation_results["valid"] = False
        
        if self._config.memory_limit_gb <= 0:
            validation_results["errors"].append("memory_limit_gb must be positive")
            validation_results["valid"] = False
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self._config.log_level not in valid_log_levels:
            validation_results["errors"].append(
                f"Invalid log_level: {self._config.log_level}. Must be one of {valid_log_levels}"
            )
            validation_results["valid"] = False
        
        # Validate CMB configuration
        try:
            self._config.cmb.validate()
        except ValueError as e:
            validation_results["errors"].append(f"Invalid CMB configuration: {e}")
            validation_results["valid"] = False
        
        # Check for log file path validity
        if self._config.log_file:
            log_path = Path(self._config.log_file)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation_results["warnings"].append(
                    f"Cannot create log file directory: {e}"
                )
        
        return validation_results


def get_data_preparation_config(config_file: Optional[str] = None) -> DataPreparationConfig:
    """
    Get data preparation configuration.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        DataPreparationConfig instance
    """
    manager = DataPreparationConfigManager(config_file)
    return manager.get_config()


def get_cmb_config(config_file: Optional[str] = None) -> CMBConfig:
    """
    Get CMB-specific configuration from the data preparation framework.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        CMBConfig instance
    """
    manager = DataPreparationConfigManager(config_file)
    return manager.get_cmb_config()


def create_example_data_preparation_config(output_file: str) -> None:
    """
    Create an example data preparation configuration file.
    
    Args:
        output_file: Path to output file
    """
    manager = DataPreparationConfigManager()
    manager.create_example_config(output_file)


def validate_deployment_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate configuration for deployment readiness.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Dictionary with validation results including deployment-specific checks
    """
    manager = DataPreparationConfigManager(config_file)
    validation_results = manager.validate_config()
    
    # Additional deployment-specific checks
    config = manager.get_config()
    
    # Check production readiness
    if config.log_level == "DEBUG":
        validation_results["warnings"].append(
            "DEBUG log level may impact performance in production"
        )
    
    if not config.validate_outputs:
        validation_results["warnings"].append(
            "Output validation is disabled - may miss data quality issues"
        )
    
    if config.performance_monitoring and config.parallel_processing:
        validation_results["warnings"].append(
            "Performance monitoring with parallel processing may impact performance"
        )
    
    # Check CMB-specific deployment readiness
    cmb_config = config.cmb
    
    if cmb_config.jacobian_step_size > 1e-5:
        validation_results["warnings"].append(
            "Large jacobian_step_size may reduce numerical accuracy"
        )
    
    if not cmb_config.cache_computations:
        validation_results["warnings"].append(
            "Computation caching is disabled - may impact performance"
        )
    
    if cmb_config.performance_monitoring and not config.performance_monitoring:
        validation_results["warnings"].append(
            "CMB performance monitoring enabled but framework monitoring disabled"
        )
    
    return validation_results


# Environment variable documentation
ENVIRONMENT_VARIABLES = {
    # Framework-level variables
    "PBUF_OUTPUT_PATH": "Path for prepared dataset outputs",
    "PBUF_CACHE_PATH": "Path for caching intermediate results",
    "PBUF_FRAMEWORK_ENABLED": "Enable/disable data preparation framework (true/false)",
    "PBUF_AUTO_PROCESS": "Automatically process datasets when loaded (true/false)",
    "PBUF_VALIDATE_OUTPUTS": "Validate output datasets (true/false)",
    "PBUF_PARALLEL_PROCESSING": "Enable parallel processing (true/false)",
    "PBUF_MAX_WORKERS": "Maximum number of worker processes (integer)",
    "PBUF_MEMORY_LIMIT_GB": "Memory limit in GB (float)",
    "PBUF_LOG_LEVEL": "Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)",
    "PBUF_LOG_FILE": "Path to log file",
    "PBUF_PERFORMANCE_MONITORING": "Enable performance monitoring (true/false)",
    "PBUF_ENV": "Environment name for configuration overrides",
    
    # CMB-specific variables
    "CMB_USE_RAW_PARAMETERS": "Enable raw parameter processing (true/false)",
    "CMB_Z_RECOMBINATION": "Recombination redshift (float)",
    "CMB_JACOBIAN_STEP_SIZE": "Numerical differentiation step size (float)",
    "CMB_VALIDATION_TOLERANCE": "Covariance validation tolerance (float)",
    "CMB_FALLBACK_TO_LEGACY": "Auto-fallback to legacy mode (true/false)",
    "CMB_CACHE_COMPUTATIONS": "Cache expensive computations (true/false)",
    "CMB_PERFORMANCE_MONITORING": "Enable CMB performance monitoring (true/false)"
}


def print_environment_variables_help():
    """Print help information about supported environment variables."""
    print("Supported Environment Variables:")
    print("=" * 50)
    
    print("\nFramework Configuration:")
    print("-" * 25)
    for var, desc in ENVIRONMENT_VARIABLES.items():
        if var.startswith("PBUF_"):
            print(f"{var:30} {desc}")
    
    print("\nCMB Module Configuration:")
    print("-" * 25)
    for var, desc in ENVIRONMENT_VARIABLES.items():
        if var.startswith("CMB_"):
            print(f"{var:30} {desc}")
    
    print("\nExample usage:")
    print("export PBUF_LOG_LEVEL=DEBUG")
    print("export CMB_USE_RAW_PARAMETERS=true")
    print("export PBUF_ENV=development")