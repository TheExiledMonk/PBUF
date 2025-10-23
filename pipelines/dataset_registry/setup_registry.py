#!/usr/bin/env python3
"""
Setup script for PBUF Dataset Registry deployment.

This script provides an interactive setup process for deploying the dataset registry
system in a new or existing PBUF installation.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from .core.config import DatasetRegistryConfig, create_example_dataset_registry_config
from .migration_tools import DatasetMigrationTool, create_migration_script
from .deployment_validation import DeploymentValidator


class RegistrySetup:
    """Interactive setup for dataset registry deployment."""
    
    def __init__(self):
        self.setup_config = {}
        self.installation_type = None
    
    def run_interactive_setup(self) -> Dict[str, Any]:
        """Run interactive setup process."""
        print("PBUF Dataset Registry Setup")
        print("=" * 40)
        
        # Determine installation type
        self._determine_installation_type()
        
        # Configure paths
        self._configure_paths()
        
        # Configure registry settings
        self._configure_registry_settings()
        
        # Configure environment
        self._configure_environment()
        
        # Perform setup
        return self._perform_setup()
    
    def _determine_installation_type(self) -> None:
        """Determine if this is a new installation or migration."""
        print("\n1. Installation Type")
        print("-------------------")
        
        # Check for existing PBUF configuration
        existing_configs = []
        for config_name in ['pbuf_config.json', 'pbuf_config.yaml', 'pbuf_config.yml']:
            if Path(config_name).exists():
                existing_configs.append(config_name)
        
        if existing_configs:
            print(f"Found existing PBUF configuration: {', '.join(existing_configs)}")
            print("1. Migrate existing installation")
            print("2. Fresh installation (will backup existing config)")
            
            choice = input("Choose option (1-2): ").strip()
            
            if choice == "1":
                self.installation_type = "migration"
                self.setup_config['existing_config'] = existing_configs[0]
            else:
                self.installation_type = "fresh"
        else:
            print("No existing PBUF configuration found.")
            self.installation_type = "fresh"
        
        print(f"Installation type: {self.installation_type}")
    
    def _configure_paths(self) -> None:
        """Configure file system paths."""
        print("\n2. Path Configuration")
        print("--------------------")
        
        # Default paths
        defaults = {
            'manifest_path': 'data/datasets_manifest.json',
            'registry_path': 'data/registry/',
            'cache_path': 'data/cache/'
        }
        
        self.setup_config['paths'] = {}
        
        for path_type, default_path in defaults.items():
            prompt = f"{path_type.replace('_', ' ').title()} [{default_path}]: "
            user_input = input(prompt).strip()
            
            if user_input:
                self.setup_config['paths'][path_type] = user_input
            else:
                self.setup_config['paths'][path_type] = default_path
        
        print(f"Configured paths: {self.setup_config['paths']}")
    
    def _configure_registry_settings(self) -> None:
        """Configure registry behavior settings."""
        print("\n3. Registry Settings")
        print("-------------------")
        
        settings = {}
        
        # Auto-fetch setting
        auto_fetch = input("Enable automatic dataset fetching? [Y/n]: ").strip().lower()
        settings['auto_fetch'] = auto_fetch not in ['n', 'no', 'false']
        
        # Verification settings
        verify_on_load = input("Enable dataset verification on load? [Y/n]: ").strip().lower()
        settings['verify_on_load'] = verify_on_load not in ['n', 'no', 'false']
        
        # Legacy fallback
        if self.installation_type == "migration":
            fallback = input("Enable fallback to legacy dataset loading? [Y/n]: ").strip().lower()
            settings['fallback_to_legacy'] = fallback not in ['n', 'no', 'false']
        else:
            settings['fallback_to_legacy'] = False
        
        # Logging level
        print("Logging levels: DEBUG, INFO, WARNING, ERROR")
        log_level = input("Log level [INFO]: ").strip().upper()
        settings['log_level'] = log_level if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'] else 'INFO'
        
        self.setup_config['registry_settings'] = settings
        print(f"Registry settings: {settings}")
    
    def _configure_environment(self) -> None:
        """Configure environment-specific settings."""
        print("\n4. Environment Configuration")
        print("---------------------------")
        
        print("Environment types:")
        print("1. Development (debug logging, no verification)")
        print("2. Testing (registry disabled, legacy fallback)")
        print("3. Production (minimal logging, full verification)")
        print("4. Custom")
        
        env_choice = input("Choose environment type (1-4) [3]: ").strip()
        
        if env_choice == "1":
            env_type = "development"
        elif env_choice == "2":
            env_type = "testing"
        elif env_choice == "4":
            env_type = "custom"
        else:
            env_type = "production"
        
        self.setup_config['environment'] = env_type
        print(f"Environment: {env_type}")
    
    def _perform_setup(self) -> Dict[str, Any]:
        """Perform the actual setup based on configuration."""
        print("\n5. Performing Setup")
        print("------------------")
        
        results = {
            'success': False,
            'steps_completed': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Create directories
            self._create_directories(results)
            
            # Step 2: Handle existing installation
            if self.installation_type == "migration":
                self._perform_migration(results)
            else:
                self._create_fresh_installation(results)
            
            # Step 3: Create configuration
            self._create_configuration(results)
            
            # Step 4: Validate setup
            self._validate_setup(results)
            
            results['success'] = len(results['errors']) == 0
            
        except Exception as e:
            results['errors'].append(f"Setup failed: {e}")
        
        return results
    
    def _create_directories(self, results: Dict[str, Any]) -> None:
        """Create required directories."""
        print("Creating directories...")
        
        paths = self.setup_config['paths']
        
        for path_type, path_value in paths.items():
            path_obj = Path(path_value)
            
            if path_type.endswith('_path') and not path_type == 'manifest_path':
                # Directory path
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    print(f"  Created directory: {path_value}")
                except Exception as e:
                    results['errors'].append(f"Failed to create directory {path_value}: {e}")
                    return
            else:
                # File path - create parent directory
                try:
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  Created parent directory for: {path_value}")
                except Exception as e:
                    results['errors'].append(f"Failed to create parent directory for {path_value}: {e}")
                    return
        
        results['steps_completed'].append('directories_created')
    
    def _perform_migration(self, results: Dict[str, Any]) -> None:
        """Perform migration from existing installation."""
        print("Performing migration...")
        
        migration_tool = DatasetMigrationTool(
            pbuf_config_path=self.setup_config['existing_config'],
            dry_run=False
        )
        
        migration_result = migration_tool.migrate_full_installation()
        
        if migration_result.success:
            print(f"  Migrated {migration_result.datasets_migrated} datasets")
            results['steps_completed'].append('migration_completed')
        else:
            results['errors'].extend(migration_result.errors)
            results['warnings'].extend(migration_result.warnings)
    
    def _create_fresh_installation(self, results: Dict[str, Any]) -> None:
        """Create fresh installation."""
        print("Creating fresh installation...")
        
        # Create empty manifest
        manifest_path = Path(self.setup_config['paths']['manifest_path'])
        
        empty_manifest = {
            "manifest_version": "1.0",
            "datasets": {}
        }
        
        try:
            with open(manifest_path, 'w') as f:
                json.dump(empty_manifest, f, indent=2)
            
            print(f"  Created empty manifest: {manifest_path}")
            results['steps_completed'].append('manifest_created')
            
        except Exception as e:
            results['errors'].append(f"Failed to create manifest: {e}")
    
    def _create_configuration(self, results: Dict[str, Any]) -> None:
        """Create configuration file."""
        print("Creating configuration...")
        
        # Build configuration
        config = DatasetRegistryConfig()
        
        # Apply path settings
        paths = self.setup_config['paths']
        config.manifest_path = paths['manifest_path']
        config.registry_path = paths['registry_path']
        config.cache_path = paths['cache_path']
        
        # Apply registry settings
        settings = self.setup_config['registry_settings']
        config.auto_fetch = settings['auto_fetch']
        config.verify_on_load = settings['verify_on_load']
        config.fallback_to_legacy = settings['fallback_to_legacy']
        config.log_level = settings['log_level']
        
        # Apply environment-specific settings
        env_type = self.setup_config['environment']
        if env_type == "development":
            config.log_level = "DEBUG"
            config.verify_on_load = False
            config.cache_enabled = False
        elif env_type == "testing":
            config.registry_enabled = False
            config.fallback_to_legacy = True
            config.auto_fetch = False
        elif env_type == "production":
            config.log_level = "WARNING"
            config.verify_on_load = True
            config.cache_enabled = True
            config.audit_trail_enabled = True
        
        # Create PBUF configuration
        pbuf_config = {
            "datasets": {
                "registry": config.to_dict()
            }
        }
        
        # Save configuration
        config_path = "pbuf_config.json"
        if self.installation_type == "migration" and self.setup_config.get('existing_config'):
            config_path = self.setup_config['existing_config']
        
        try:
            # Backup existing config if it exists
            if Path(config_path).exists():
                backup_path = f"{config_path}.backup"
                shutil.copy2(config_path, backup_path)
                print(f"  Backed up existing config to: {backup_path}")
            
            # Load existing config and merge
            existing_config = {}
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
            
            # Merge configurations
            if 'datasets' not in existing_config:
                existing_config['datasets'] = {}
            
            existing_config['datasets']['registry'] = config.to_dict()
            
            # Save merged configuration
            with open(config_path, 'w') as f:
                json.dump(existing_config, f, indent=2)
            
            print(f"  Created configuration: {config_path}")
            results['steps_completed'].append('configuration_created')
            
        except Exception as e:
            results['errors'].append(f"Failed to create configuration: {e}")
    
    def _validate_setup(self, results: Dict[str, Any]) -> None:
        """Validate the setup."""
        print("Validating setup...")
        
        try:
            validator = DeploymentValidator(config_path="pbuf_config.json", verbose=False)
            validation_results = validator.run_full_validation()
            
            if validation_results['overall_status'] == 'PASS':
                print("  Validation: PASSED")
                results['steps_completed'].append('validation_passed')
            else:
                print(f"  Validation: {validation_results['overall_status']}")
                results['warnings'].extend(validation_results['warnings'])
                if validation_results['errors']:
                    results['errors'].extend(validation_results['errors'])
            
        except Exception as e:
            results['warnings'].append(f"Validation failed: {e}")


def main():
    """Main entry point for setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup PBUF Dataset Registry")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Run interactive setup (default)")
    parser.add_argument("--create-migration-script", action="store_true",
                       help="Create standalone migration script")
    parser.add_argument("--create-example-config", metavar="FILE",
                       help="Create example configuration file")
    
    args = parser.parse_args()
    
    if args.create_migration_script:
        create_migration_script()
        print("Migration script created: migrate_to_registry.py")
        return
    
    if args.create_example_config:
        create_example_dataset_registry_config(args.create_example_config)
        print(f"Example configuration created: {args.create_example_config}")
        return
    
    # Run interactive setup
    setup = RegistrySetup()
    results = setup.run_interactive_setup()
    
    print("\n" + "=" * 40)
    print("Setup Results")
    print("=" * 40)
    
    print(f"Success: {results['success']}")
    print(f"Steps completed: {len(results['steps_completed'])}")
    
    if results['steps_completed']:
        print("Completed steps:")
        for step in results['steps_completed']:
            print(f"  ✓ {step}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  ✗ {error}")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    
    if results['success']:
        print("\nSetup completed successfully!")
        print("You can now use the dataset registry system.")
        print("\nNext steps:")
        print("1. Add datasets to the manifest file")
        print("2. Test dataset fetching with the CLI")
        print("3. Run validation: python -m pipelines.dataset_registry.deployment_validation")
    else:
        print("\nSetup completed with errors. Please review and fix the issues above.")
    
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()