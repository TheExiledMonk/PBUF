"""
Integration between dataset registry configuration and PBUF configuration system.

This module provides seamless integration between the dataset registry configuration
and the existing PBUF configuration system, ensuring consistent configuration
management across the entire pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from .config import DatasetRegistryConfig, DatasetRegistryConfigManager


class PBUFDatasetRegistryConfigIntegration:
    """Integrates dataset registry configuration with PBUF configuration system."""
    
    def __init__(self, pbuf_config_manager=None):
        """
        Initialize configuration integration.
        
        Args:
            pbuf_config_manager: Optional PBUF ConfigurationManager instance
        """
        self.pbuf_config_manager = pbuf_config_manager
        self._registry_config_manager = None
        self._cached_config = None
    
    def get_registry_config(self) -> DatasetRegistryConfig:
        """
        Get dataset registry configuration from PBUF configuration system.
        
        Returns:
            DatasetRegistryConfig instance
        """
        if self._cached_config is not None:
            return self._cached_config
        
        # Try to get configuration from PBUF config manager first
        if self.pbuf_config_manager:
            try:
                pbuf_registry_config = self.pbuf_config_manager.get_dataset_registry_config()
                config = DatasetRegistryConfig.from_dict(pbuf_registry_config)
                self._cached_config = config
                return config
            except Exception:
                # Fall back to standalone configuration
                pass
        
        # Fall back to standalone dataset registry configuration
        if not self._registry_config_manager:
            self._registry_config_manager = DatasetRegistryConfigManager()
        
        config = self._registry_config_manager.get_config()
        self._cached_config = config
        return config
    
    def update_pbuf_config(self, registry_config: DatasetRegistryConfig) -> None:
        """
        Update PBUF configuration with dataset registry settings.
        
        Args:
            registry_config: Dataset registry configuration to integrate
        """
        if not self.pbuf_config_manager:
            return
        
        # Convert registry config to PBUF format
        registry_dict = registry_config.to_dict()
        
        # Update PBUF configuration
        if not hasattr(self.pbuf_config_manager, 'config_data'):
            self.pbuf_config_manager.config_data = {}
        
        if 'datasets' not in self.pbuf_config_manager.config_data:
            self.pbuf_config_manager.config_data['datasets'] = {}
        
        self.pbuf_config_manager.config_data['datasets']['registry'] = registry_dict
        
        # Clear cached config to force reload
        self._cached_config = None
    
    def migrate_legacy_dataset_config(self) -> Dict[str, Any]:
        """
        Migrate legacy dataset configuration to registry format.
        
        Returns:
            Dictionary with migration results
        """
        migration_results = {
            "migrated": False,
            "legacy_settings_found": [],
            "new_settings_created": [],
            "warnings": []
        }
        
        if not self.pbuf_config_manager:
            migration_results["warnings"].append("No PBUF configuration manager available")
            return migration_results
        
        datasets_config = self.pbuf_config_manager.get_datasets_config()
        
        # Check for legacy settings that need migration
        legacy_mappings = {
            "data_directory": "cache_path",
            "default_datasets": "legacy_dataset_paths"
        }
        
        registry_config = datasets_config.get('registry', {})
        migrated_settings = {}
        
        for legacy_key, new_key in legacy_mappings.items():
            if legacy_key in datasets_config and new_key not in registry_config:
                migration_results["legacy_settings_found"].append(legacy_key)
                
                if legacy_key == "data_directory":
                    migrated_settings["cache_path"] = datasets_config[legacy_key]
                    migration_results["new_settings_created"].append("cache_path")
                
                elif legacy_key == "default_datasets":
                    # Convert default datasets to legacy dataset paths
                    legacy_paths = {}
                    for dataset in datasets_config[legacy_key]:
                        legacy_paths[dataset] = f"data/{dataset}.dat"
                    migrated_settings["legacy_dataset_paths"] = legacy_paths
                    migration_results["new_settings_created"].append("legacy_dataset_paths")
                
                migration_results["migrated"] = True
        
        # Enable fallback to legacy if migration occurred
        if migration_results["migrated"]:
            migrated_settings["fallback_to_legacy"] = True
            migration_results["new_settings_created"].append("fallback_to_legacy")
        
        # Update configuration if migration occurred
        if migrated_settings:
            if 'datasets' not in self.pbuf_config_manager.config_data:
                self.pbuf_config_manager.config_data['datasets'] = {}
            if 'registry' not in self.pbuf_config_manager.config_data['datasets']:
                self.pbuf_config_manager.config_data['datasets']['registry'] = {}
            
            self.pbuf_config_manager.config_data['datasets']['registry'].update(migrated_settings)
            
            # Clear cached config to force reload
            self._cached_config = None
        
        return migration_results
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the integrated configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "pbuf_integration": True,
            "registry_config_valid": True
        }
        
        try:
            config = self.get_registry_config()
            
            # Validate registry configuration
            if self._registry_config_manager:
                registry_validation = self._registry_config_manager.validate_config()
                validation_results["registry_config_valid"] = registry_validation["valid"]
                validation_results["errors"].extend(registry_validation["errors"])
                validation_results["warnings"].extend(registry_validation["warnings"])
            
            # Validate PBUF integration
            if not self.pbuf_config_manager:
                validation_results["pbuf_integration"] = False
                validation_results["warnings"].append("No PBUF configuration manager available")
            else:
                # Check if registry configuration is properly integrated
                try:
                    pbuf_registry_config = self.pbuf_config_manager.get_dataset_registry_config()
                    if not pbuf_registry_config:
                        validation_results["warnings"].append("No registry configuration found in PBUF config")
                except Exception as e:
                    validation_results["warnings"].append(f"Error accessing PBUF registry config: {e}")
            
            # Check for configuration conflicts
            if config.fallback_to_legacy and not config.legacy_dataset_paths:
                validation_results["warnings"].append(
                    "fallback_to_legacy is enabled but no legacy_dataset_paths are configured"
                )
            
            # Validate paths are accessible
            for path_attr in ["manifest_path", "registry_path", "cache_path"]:
                path_value = getattr(config, path_attr)
                path_obj = Path(path_value)
                
                if not path_obj.parent.exists():
                    try:
                        path_obj.parent.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        validation_results["errors"].append(
                            f"Cannot create directory for {path_attr}: {e}"
                        )
                        validation_results["valid"] = False
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Configuration validation failed: {e}")
        
        return validation_results
    
    def create_deployment_config(self, environment: str = "production") -> Dict[str, Any]:
        """
        Create deployment-ready configuration for specified environment.
        
        Args:
            environment: Target environment (development, testing, production)
            
        Returns:
            Dictionary with deployment configuration
        """
        base_config = DatasetRegistryConfig()
        
        # Environment-specific settings
        if environment == "development":
            base_config.log_level = "DEBUG"
            base_config.verify_on_load = False
            base_config.cache_enabled = False
            base_config.structured_logging = True
        
        elif environment == "testing":
            base_config.registry_enabled = False
            base_config.fallback_to_legacy = True
            base_config.auto_fetch = False
            base_config.log_level = "WARNING"
        
        elif environment == "production":
            base_config.log_level = "WARNING"
            base_config.verify_on_load = True
            base_config.cache_enabled = True
            base_config.audit_trail_enabled = True
            base_config.structured_logging = True
        
        # Add environment-specific overrides
        base_config.environment_overrides = {
            environment: base_config.to_dict()
        }
        
        return {"dataset_registry": base_config.to_dict()}


def get_integrated_registry_config(pbuf_config_manager=None) -> DatasetRegistryConfig:
    """
    Get dataset registry configuration integrated with PBUF configuration system.
    
    Args:
        pbuf_config_manager: Optional PBUF ConfigurationManager instance
        
    Returns:
        DatasetRegistryConfig instance
    """
    integration = PBUFDatasetRegistryConfigIntegration(pbuf_config_manager)
    return integration.get_registry_config()


def migrate_legacy_dataset_configuration(pbuf_config_manager) -> Dict[str, Any]:
    """
    Migrate legacy dataset configuration to registry format.
    
    Args:
        pbuf_config_manager: PBUF ConfigurationManager instance
        
    Returns:
        Dictionary with migration results
    """
    integration = PBUFDatasetRegistryConfigIntegration(pbuf_config_manager)
    return integration.migrate_legacy_dataset_config()


def validate_integrated_configuration(pbuf_config_manager=None) -> Dict[str, Any]:
    """
    Validate integrated dataset registry configuration.
    
    Args:
        pbuf_config_manager: Optional PBUF ConfigurationManager instance
        
    Returns:
        Dictionary with validation results
    """
    integration = PBUFDatasetRegistryConfigIntegration(pbuf_config_manager)
    return integration.validate_configuration()