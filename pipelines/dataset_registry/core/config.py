"""
Configuration management for the dataset registry system.

This module provides configuration file support, default value handling,
and environment-specific configuration overrides for the dataset registry.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DatasetRegistryConfig:
    """Configuration settings for the dataset registry system."""
    
    # Core registry settings
    registry_enabled: bool = True
    manifest_path: str = "data/datasets_manifest.json"
    registry_path: str = "data/registry/"
    cache_path: str = "data/cache/"
    
    # Dataset fetching behavior
    auto_fetch: bool = True
    verify_on_load: bool = True
    fallback_to_legacy: bool = False
    
    # Download settings
    download_timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    retry_backoff_factor: float = 2.0
    chunk_size: int = 8192
    
    # Verification settings
    verify_checksums: bool = True
    verify_file_sizes: bool = True
    verify_schemas: bool = True
    checksum_algorithm: str = "sha256"
    
    # Logging and observability
    structured_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    audit_trail_enabled: bool = True
    
    # Performance settings
    concurrent_downloads: int = 3
    cache_enabled: bool = True
    cache_max_size_gb: float = 10.0
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy compatibility
    legacy_dataset_paths: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "registry_enabled": self.registry_enabled,
            "manifest_path": self.manifest_path,
            "registry_path": self.registry_path,
            "cache_path": self.cache_path,
            "auto_fetch": self.auto_fetch,
            "verify_on_load": self.verify_on_load,
            "fallback_to_legacy": self.fallback_to_legacy,
            "download": {
                "timeout": self.download_timeout,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "max_retry_delay": self.max_retry_delay,
                "retry_backoff_factor": self.retry_backoff_factor,
                "chunk_size": self.chunk_size,
                "concurrent_downloads": self.concurrent_downloads
            },
            "verification": {
                "verify_checksums": self.verify_checksums,
                "verify_file_sizes": self.verify_file_sizes,
                "verify_schemas": self.verify_schemas,
                "checksum_algorithm": self.checksum_algorithm
            },
            "logging": {
                "structured_logging": self.structured_logging,
                "log_level": self.log_level,
                "log_file": self.log_file,
                "audit_trail_enabled": self.audit_trail_enabled
            },
            "cache": {
                "enabled": self.cache_enabled,
                "max_size_gb": self.cache_max_size_gb
            },
            "environment_overrides": self.environment_overrides,
            "legacy_dataset_paths": self.legacy_dataset_paths
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetRegistryConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update top-level settings
        for key in ["registry_enabled", "manifest_path", "registry_path", "cache_path",
                   "auto_fetch", "verify_on_load", "fallback_to_legacy"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # Update download settings
        if "download" in config_dict:
            download_config = config_dict["download"]
            for key in ["timeout", "max_retries", "retry_delay", "max_retry_delay",
                       "retry_backoff_factor", "chunk_size", "concurrent_downloads"]:
                if key in download_config:
                    if key == "timeout":
                        config.download_timeout = download_config[key]
                    elif key == "max_retries":
                        config.max_retries = download_config[key]
                    elif key == "retry_delay":
                        config.retry_delay = download_config[key]
                    elif key == "max_retry_delay":
                        config.max_retry_delay = download_config[key]
                    elif key == "retry_backoff_factor":
                        config.retry_backoff_factor = download_config[key]
                    elif key == "chunk_size":
                        config.chunk_size = download_config[key]
                    elif key == "concurrent_downloads":
                        config.concurrent_downloads = download_config[key]
        
        # Update verification settings
        if "verification" in config_dict:
            verification_config = config_dict["verification"]
            for key in ["verify_checksums", "verify_file_sizes", "verify_schemas", "checksum_algorithm"]:
                if key in verification_config:
                    setattr(config, key, verification_config[key])
        
        # Update logging settings
        if "logging" in config_dict:
            logging_config = config_dict["logging"]
            for key in ["structured_logging", "log_level", "log_file", "audit_trail_enabled"]:
                if key in logging_config:
                    setattr(config, key, logging_config[key])
        
        # Update cache settings
        if "cache" in config_dict:
            cache_config = config_dict["cache"]
            if "enabled" in cache_config:
                config.cache_enabled = cache_config["enabled"]
            if "max_size_gb" in cache_config:
                config.cache_max_size_gb = cache_config["max_size_gb"]
        
        # Update environment overrides and legacy paths
        if "environment_overrides" in config_dict:
            config.environment_overrides = config_dict["environment_overrides"]
        if "legacy_dataset_paths" in config_dict:
            config.legacy_dataset_paths = config_dict["legacy_dataset_paths"]
        
        return config


class DatasetRegistryConfigManager:
    """Manages dataset registry configuration with environment-specific overrides."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file
        self._config = DatasetRegistryConfig()
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
            
            # Extract dataset registry configuration
            if "dataset_registry" in config_data:
                registry_config = config_data["dataset_registry"]
            elif "datasets" in config_data:
                # Check if this is a legacy datasets config that needs migration
                registry_config = self._migrate_legacy_config(config_data["datasets"])
            else:
                # Assume the entire file is dataset registry config
                registry_config = config_data
            
            self._config = DatasetRegistryConfig.from_dict(registry_config)
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
            'pbuf_config.json',
            'dataset_registry_config.json',
            '.dataset_registry_config.json'
        ]
        
        # Search locations
        search_paths = [
            Path('.'),  # Current directory
            Path.home(),  # Home directory
            Path.home() / '.config' / 'pbuf'  # XDG config directory
        ]
        
        for search_path in search_paths:
            for config_name in config_names:
                config_file = search_path / config_name
                if config_file.exists():
                    return str(config_file)
        
        return None
    
    def _migrate_legacy_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate legacy dataset configuration to new format.
        
        Args:
            legacy_config: Legacy configuration dictionary
            
        Returns:
            Migrated configuration dictionary
        """
        migrated_config = {}
        
        # Map legacy settings to new format
        if "data_directory" in legacy_config:
            migrated_config["cache_path"] = legacy_config["data_directory"]
        
        if "default_datasets" in legacy_config:
            # Store as legacy dataset paths for backward compatibility
            migrated_config["legacy_dataset_paths"] = {
                name: f"data/{name}.dat" for name in legacy_config["default_datasets"]
            }
        
        if "custom_covariance" in legacy_config:
            migrated_config["legacy_dataset_paths"] = migrated_config.get("legacy_dataset_paths", {})
            if legacy_config["custom_covariance"]:
                migrated_config["legacy_dataset_paths"]["custom_covariance"] = legacy_config["custom_covariance"]
        
        # Enable fallback to legacy for migrated configurations
        migrated_config["fallback_to_legacy"] = True
        
        return migrated_config
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides."""
        # Check for environment variables
        env_overrides = {}
        
        # Registry paths
        if "PBUF_REGISTRY_PATH" in os.environ:
            env_overrides["registry_path"] = os.environ["PBUF_REGISTRY_PATH"]
        
        if "PBUF_MANIFEST_PATH" in os.environ:
            env_overrides["manifest_path"] = os.environ["PBUF_MANIFEST_PATH"]
        
        if "PBUF_CACHE_PATH" in os.environ:
            env_overrides["cache_path"] = os.environ["PBUF_CACHE_PATH"]
        
        # Registry behavior
        if "PBUF_REGISTRY_ENABLED" in os.environ:
            env_overrides["registry_enabled"] = os.environ["PBUF_REGISTRY_ENABLED"].lower() in ["true", "1", "yes"]
        
        if "PBUF_AUTO_FETCH" in os.environ:
            env_overrides["auto_fetch"] = os.environ["PBUF_AUTO_FETCH"].lower() in ["true", "1", "yes"]
        
        if "PBUF_VERIFY_ON_LOAD" in os.environ:
            env_overrides["verify_on_load"] = os.environ["PBUF_VERIFY_ON_LOAD"].lower() in ["true", "1", "yes"]
        
        # Logging
        if "PBUF_LOG_LEVEL" in os.environ:
            env_overrides["log_level"] = os.environ["PBUF_LOG_LEVEL"]
        
        # Apply overrides
        for key, value in env_overrides.items():
            setattr(self._config, key, value)
        
        # Apply configured environment overrides
        current_env = os.environ.get("PBUF_ENV", "default")
        if current_env in self._config.environment_overrides:
            env_config = self._config.environment_overrides[current_env]
            for key, value in env_config.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
    
    def get_config(self) -> DatasetRegistryConfig:
        """Get the current configuration."""
        return self._config
    
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
        
        config_dict = {"dataset_registry": self._config.to_dict()}
        
        with open(output_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def create_example_config(self, output_file: str) -> None:
        """
        Create an example configuration file with all available options.
        
        Args:
            output_file: Path to output file
        """
        example_config = DatasetRegistryConfig()
        
        # Add some example environment overrides
        example_config.environment_overrides = {
            "development": {
                "log_level": "DEBUG",
                "verify_on_load": False,
                "cache_enabled": False
            },
            "production": {
                "log_level": "WARNING",
                "verify_on_load": True,
                "cache_enabled": True,
                "audit_trail_enabled": True
            },
            "testing": {
                "registry_enabled": False,
                "fallback_to_legacy": True,
                "auto_fetch": False
            }
        }
        
        # Add example legacy dataset paths
        example_config.legacy_dataset_paths = {
            "cmb": "data/cmb_planck2018.dat",
            "bao": "data/bao_compilation.dat",
            "sn": "data/sn_pantheon_plus.dat"
        }
        
        config_dict = {"dataset_registry": example_config.to_dict()}
        
        with open(output_file, 'w') as f:
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
        for path_attr in ["manifest_path", "registry_path", "cache_path"]:
            path_value = getattr(self._config, path_attr)
            path_obj = Path(path_value)
            
            if path_attr == "manifest_path":
                # Manifest file should exist if registry is enabled
                if self._config.registry_enabled and not path_obj.exists():
                    validation_results["warnings"].append(
                        f"Manifest file does not exist: {path_value}"
                    )
            else:
                # Directory paths should be creatable
                try:
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    validation_results["errors"].append(
                        f"Cannot create directory for {path_attr}: {e}"
                    )
                    validation_results["valid"] = False
        
        # Validate numeric settings
        if self._config.download_timeout <= 0:
            validation_results["errors"].append("download_timeout must be positive")
            validation_results["valid"] = False
        
        if self._config.max_retries < 0:
            validation_results["errors"].append("max_retries must be non-negative")
            validation_results["valid"] = False
        
        if self._config.concurrent_downloads <= 0:
            validation_results["errors"].append("concurrent_downloads must be positive")
            validation_results["valid"] = False
        
        if self._config.cache_max_size_gb <= 0:
            validation_results["errors"].append("cache_max_size_gb must be positive")
            validation_results["valid"] = False
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self._config.log_level not in valid_log_levels:
            validation_results["errors"].append(
                f"Invalid log_level: {self._config.log_level}. Must be one of {valid_log_levels}"
            )
            validation_results["valid"] = False
        
        # Validate checksum algorithm
        valid_algorithms = ["sha256", "sha1", "md5"]
        if self._config.checksum_algorithm not in valid_algorithms:
            validation_results["errors"].append(
                f"Invalid checksum_algorithm: {self._config.checksum_algorithm}. Must be one of {valid_algorithms}"
            )
            validation_results["valid"] = False
        
        return validation_results


def get_dataset_registry_config(config_file: Optional[str] = None) -> DatasetRegistryConfig:
    """
    Get dataset registry configuration.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        DatasetRegistryConfig instance
    """
    manager = DatasetRegistryConfigManager(config_file)
    return manager.get_config()


def create_example_dataset_registry_config(output_file: str) -> None:
    """
    Create an example dataset registry configuration file.
    
    Args:
        output_file: Path to output file
    """
    manager = DatasetRegistryConfigManager()
    manager.create_example_config(output_file)