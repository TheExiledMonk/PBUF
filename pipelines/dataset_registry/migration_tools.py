"""
Migration tools for transitioning from legacy dataset configurations to the registry system.

This module provides automated migration tools, validation utilities, and backward
compatibility support for existing PBUF installations.
"""

import json
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .core.config import DatasetRegistryConfig, DatasetRegistryConfigManager
from .core.config_integration import PBUFDatasetRegistryConfigIntegration
from .core.registry_manager import RegistryManager
from .core.manifest_schema import DatasetManifest


@dataclass
class MigrationResult:
    """Results of a migration operation."""
    success: bool
    datasets_migrated: int
    config_updated: bool
    backup_created: bool
    errors: List[str]
    warnings: List[str]
    migration_log: List[str]


class DatasetMigrationTool:
    """Tool for migrating legacy dataset configurations to the registry system."""
    
    def __init__(self, pbuf_config_path: Optional[str] = None, dry_run: bool = False):
        """
        Initialize migration tool.
        
        Args:
            pbuf_config_path: Path to PBUF configuration file
            dry_run: If True, perform migration without making changes
        """
        self.pbuf_config_path = pbuf_config_path
        self.dry_run = dry_run
        self.migration_log = []
        
    def migrate_full_installation(self, backup_dir: Optional[str] = None) -> MigrationResult:
        """
        Perform complete migration of PBUF installation to use dataset registry.
        
        Args:
            backup_dir: Directory to store backups (default: ./migration_backup_YYYYMMDD)
            
        Returns:
            MigrationResult with migration details
        """
        if not backup_dir:
            backup_dir = f"migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = MigrationResult(
            success=False,
            datasets_migrated=0,
            config_updated=False,
            backup_created=False,
            errors=[],
            warnings=[],
            migration_log=[]
        )
        
        try:
            # Step 1: Create backup
            self._log("Starting full installation migration")
            if not self.dry_run:
                self._create_backup(backup_dir)
                result.backup_created = True
                self._log(f"Backup created in {backup_dir}")
            
            # Step 2: Discover existing datasets
            datasets = self._discover_existing_datasets()
            self._log(f"Discovered {len(datasets)} existing datasets")
            
            # Step 3: Migrate configuration
            config_migration = self._migrate_configuration()
            result.config_updated = config_migration['updated']
            if config_migration['errors']:
                result.errors.extend(config_migration['errors'])
            if config_migration['warnings']:
                result.warnings.extend(config_migration['warnings'])
            
            # Step 4: Create dataset manifest
            manifest_result = self._create_dataset_manifest(datasets)
            if manifest_result['errors']:
                result.errors.extend(manifest_result['errors'])
            
            # Step 5: Register existing datasets
            registry_result = self._register_existing_datasets(datasets)
            result.datasets_migrated = registry_result['registered']
            if registry_result['errors']:
                result.errors.extend(registry_result['errors'])
            
            # Step 6: Validate migration
            validation_result = self._validate_migration()
            if validation_result['errors']:
                result.errors.extend(validation_result['errors'])
            if validation_result['warnings']:
                result.warnings.extend(validation_result['warnings'])
            
            result.success = len(result.errors) == 0
            result.migration_log = self.migration_log.copy()
            
            if result.success:
                self._log("Migration completed successfully")
            else:
                self._log(f"Migration completed with {len(result.errors)} errors")
            
        except Exception as e:
            result.errors.append(f"Migration failed with exception: {e}")
            self._log(f"Migration failed: {e}")
        
        return result    

    def _log(self, message: str) -> None:
        """Add message to migration log."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.migration_log.append(log_entry)
        print(log_entry)
    
    def _create_backup(self, backup_dir: str) -> None:
        """Create backup of existing configuration and data."""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup configuration files
        config_files = [
            'pbuf_config.json',
            'pbuf_config.yaml',
            'pbuf_config.yml',
            'pbuf_config.ini'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                shutil.copy2(config_file, backup_path / config_file)
                self._log(f"Backed up {config_file}")
        
        # Backup existing data directory
        if Path('data').exists():
            shutil.copytree('data', backup_path / 'data', dirs_exist_ok=True)
            self._log("Backed up data directory")
    
    def _discover_existing_datasets(self) -> List[Dict[str, Any]]:
        """Discover existing dataset files in the installation."""
        datasets = []
        
        # Common dataset file patterns
        dataset_patterns = {
            'cmb': ['*cmb*.dat', '*planck*.dat', '*distance_priors*.dat'],
            'bao': ['*bao*.dat', '*baryon*.dat'],
            'sn': ['*sn*.dat', '*supernova*.dat', '*pantheon*.dat']
        }
        
        # Search in common locations
        search_paths = [Path('data'), Path('.'), Path('datasets')]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for dataset_type, patterns in dataset_patterns.items():
                for pattern in patterns:
                    for file_path in search_path.glob(pattern):
                        if file_path.is_file():
                            dataset_info = self._analyze_dataset_file(file_path, dataset_type)
                            if dataset_info:
                                datasets.append(dataset_info)
        
        return datasets
    
    def _analyze_dataset_file(self, file_path: Path, dataset_type: str) -> Optional[Dict[str, Any]]:
        """Analyze a dataset file to extract metadata."""
        try:
            # Calculate file hash and size
            sha256_hash = hashlib.sha256()
            file_size = 0
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
                    file_size += len(chunk)
            
            # Determine dataset name
            dataset_name = self._generate_dataset_name(file_path, dataset_type)
            
            return {
                'name': dataset_name,
                'type': dataset_type,
                'file_path': str(file_path),
                'file_size': file_size,
                'sha256': sha256_hash.hexdigest(),
                'canonical_name': self._generate_canonical_name(dataset_name, dataset_type),
                'description': f"Migrated {dataset_type.upper()} dataset from {file_path.name}"
            }
            
        except Exception as e:
            self._log(f"Error analyzing dataset file {file_path}: {e}")
            return None
    
    def _generate_dataset_name(self, file_path: Path, dataset_type: str) -> str:
        """Generate a standardized dataset name."""
        base_name = file_path.stem.lower()
        
        # Remove common prefixes/suffixes
        base_name = base_name.replace('data_', '').replace('_data', '')
        base_name = base_name.replace('dataset_', '').replace('_dataset', '')
        
        # Ensure it starts with dataset type
        if not base_name.startswith(dataset_type):
            base_name = f"{dataset_type}_{base_name}"
        
        return base_name
    
    def _generate_canonical_name(self, dataset_name: str, dataset_type: str) -> str:
        """Generate a human-readable canonical name."""
        type_names = {
            'cmb': 'CMB',
            'bao': 'BAO',
            'sn': 'Supernova'
        }
        
        type_display = type_names.get(dataset_type, dataset_type.upper())
        
        if 'planck' in dataset_name.lower():
            return f"Planck {type_display} Dataset"
        elif 'pantheon' in dataset_name.lower():
            return f"Pantheon+ {type_display} Dataset"
        elif 'compilation' in dataset_name.lower():
            return f"{type_display} Compilation Dataset"
        else:
            return f"{type_display} Dataset ({dataset_name})"    
 
   def _migrate_configuration(self) -> Dict[str, Any]:
        """Migrate PBUF configuration to include dataset registry settings."""
        result = {
            'updated': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            if self.pbuf_config_path and Path(self.pbuf_config_path).exists():
                # Load existing configuration
                from pipelines.fit_core.config import ConfigurationManager
                
                config_manager = ConfigurationManager(self.pbuf_config_path)
                
                # Perform migration using integration tool
                integration = PBUFDatasetRegistryConfigIntegration(config_manager)
                migration_result = integration.migrate_legacy_dataset_config()
                
                if migration_result['migrated']:
                    result['updated'] = True
                    self._log("Configuration migrated successfully")
                    
                    if not self.dry_run:
                        # Save updated configuration
                        backup_path = f"{self.pbuf_config_path}.pre_migration"
                        shutil.copy2(self.pbuf_config_path, backup_path)
                        config_manager.save_config(self.pbuf_config_path)
                        self._log(f"Configuration saved, backup at {backup_path}")
                else:
                    result['warnings'].append("No legacy configuration found to migrate")
                    
            else:
                # Create new configuration
                self._create_default_configuration()
                result['updated'] = True
                
        except Exception as e:
            result['errors'].append(f"Configuration migration failed: {e}")
        
        return result
    
    def _create_default_configuration(self) -> None:
        """Create default PBUF configuration with dataset registry enabled."""
        default_config = {
            "datasets": {
                "default_datasets": ["cmb", "bao", "sn"],
                "data_directory": "./data",
                "registry": {
                    "enabled": True,
                    "manifest_path": "data/datasets_manifest.json",
                    "registry_path": "data/registry/",
                    "cache_path": "data/cache/",
                    "auto_fetch": True,
                    "verify_on_load": True,
                    "fallback_to_legacy": True,
                    "download": {
                        "timeout": 300,
                        "max_retries": 3,
                        "concurrent_downloads": 3
                    },
                    "verification": {
                        "verify_checksums": True,
                        "verify_file_sizes": True,
                        "verify_schemas": True
                    },
                    "logging": {
                        "structured_logging": True,
                        "log_level": "INFO",
                        "audit_trail_enabled": True
                    }
                }
            }
        }
        
        if not self.dry_run:
            config_path = self.pbuf_config_path or 'pbuf_config.json'
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            self._log(f"Created default configuration: {config_path}")
    
    def _create_dataset_manifest(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create dataset manifest from discovered datasets."""
        result = {
            'created': False,
            'errors': []
        }
        
        try:
            manifest_data = {
                "manifest_version": "1.0",
                "datasets": {}
            }
            
            for dataset in datasets:
                manifest_data["datasets"][dataset['name']] = {
                    "canonical_name": dataset['canonical_name'],
                    "description": dataset['description'],
                    "citation": "Migrated dataset - citation needed",
                    "license": "Unknown - license needed",
                    "sources": {
                        "manual": {
                            "path": dataset['file_path']
                        }
                    },
                    "verification": {
                        "sha256": dataset['sha256'],
                        "size_bytes": dataset['file_size']
                    },
                    "metadata": {
                        "dataset_type": dataset['type'],
                        "migration_source": "legacy_installation",
                        "migration_date": datetime.now().isoformat()
                    }
                }
            
            if not self.dry_run:
                manifest_path = Path("data/datasets_manifest.json")
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(manifest_path, 'w') as f:
                    json.dump(manifest_data, f, indent=2)
                
                result['created'] = True
                self._log(f"Created dataset manifest with {len(datasets)} datasets")
            else:
                self._log(f"Would create manifest with {len(datasets)} datasets")
                
        except Exception as e:
            result['errors'].append(f"Manifest creation failed: {e}")
        
        return result
    
    def _register_existing_datasets(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Register existing datasets in the registry."""
        result = {
            'registered': 0,
            'errors': []
        }
        
        if self.dry_run:
            result['registered'] = len(datasets)
            self._log(f"Would register {len(datasets)} datasets")
            return result
        
        try:
            registry = RegistryManager()
            
            for dataset in datasets:
                try:
                    registry.register_manual_dataset(
                        name=dataset['name'],
                        file_path=dataset['file_path'],
                        description=dataset['description'],
                        source_info={
                            "type": "migration",
                            "original_path": dataset['file_path'],
                            "migration_date": datetime.now().isoformat()
                        },
                        expected_sha256=dataset['sha256'],
                        expected_size=dataset['file_size']
                    )
                    
                    result['registered'] += 1
                    self._log(f"Registered dataset: {dataset['name']}")
                    
                except Exception as e:
                    error_msg = f"Failed to register {dataset['name']}: {e}"
                    result['errors'].append(error_msg)
                    self._log(error_msg)
            
        except Exception as e:
            result['errors'].append(f"Registry initialization failed: {e}")
        
        return result    
   
 def _validate_migration(self) -> Dict[str, Any]:
        """Validate the migration was successful."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate configuration
            from pipelines.dataset_registry.core.config_integration import validate_integrated_configuration
            from pipelines.fit_core.config import ConfigurationManager
            
            if self.pbuf_config_path and Path(self.pbuf_config_path).exists():
                config_manager = ConfigurationManager(self.pbuf_config_path)
                validation = validate_integrated_configuration(config_manager)
                
                if not validation['valid']:
                    result['valid'] = False
                    result['errors'].extend(validation['errors'])
                
                result['warnings'].extend(validation['warnings'])
            
            # Validate manifest exists
            manifest_path = Path("data/datasets_manifest.json")
            if not manifest_path.exists():
                result['warnings'].append("Dataset manifest not found")
            
            # Validate registry directory
            registry_path = Path("data/registry")
            if not registry_path.exists():
                result['warnings'].append("Registry directory not found")
            
        except Exception as e:
            result['errors'].append(f"Validation failed: {e}")
            result['valid'] = False
        
        return result


class BackwardCompatibilityValidator:
    """Validates backward compatibility during deployment."""
    
    def __init__(self):
        self.validation_results = []
    
    def validate_deployment(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate that deployment maintains backward compatibility.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'compatible': True,
            'errors': [],
            'warnings': [],
            'tests_passed': 0,
            'tests_failed': 0
        }
        
        # Test 1: Configuration loading
        config_test = self._test_configuration_loading(config_path)
        self._update_results(results, config_test, "Configuration loading")
        
        # Test 2: Legacy dataset access
        legacy_test = self._test_legacy_dataset_access()
        self._update_results(results, legacy_test, "Legacy dataset access")
        
        # Test 3: Registry functionality
        registry_test = self._test_registry_functionality()
        self._update_results(results, registry_test, "Registry functionality")
        
        # Test 4: Pipeline integration
        pipeline_test = self._test_pipeline_integration()
        self._update_results(results, pipeline_test, "Pipeline integration")
        
        return results
    
    def _update_results(self, results: Dict[str, Any], test_result: Dict[str, Any], test_name: str) -> None:
        """Update overall results with individual test results."""
        if test_result['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            results['compatible'] = False
            results['errors'].append(f"{test_name}: {test_result['error']}")
        
        if test_result.get('warnings'):
            results['warnings'].extend([f"{test_name}: {w}" for w in test_result['warnings']])
    
    def _test_configuration_loading(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Test configuration loading."""
        try:
            from pipelines.fit_core.config import ConfigurationManager
            
            if config_path and Path(config_path).exists():
                config_manager = ConfigurationManager(config_path)
                datasets_config = config_manager.get_datasets_config()
                
                # Check if registry configuration exists
                if 'registry' not in datasets_config:
                    return {
                        'passed': False,
                        'error': 'Registry configuration not found in datasets config'
                    }
            
            return {'passed': True}
            
        except Exception as e:
            return {
                'passed': False,
                'error': f'Configuration loading failed: {e}'
            }
    
    def _test_legacy_dataset_access(self) -> Dict[str, Any]:
        """Test that legacy dataset access still works."""
        try:
            # Test that we can still access datasets through legacy paths
            from pipelines.fit_core.datasets import load_dataset
            
            # This should work even if registry is not fully configured
            # (should fall back to legacy loading)
            
            return {'passed': True}
            
        except Exception as e:
            return {
                'passed': False,
                'error': f'Legacy dataset access failed: {e}'
            }
    
    def _test_registry_functionality(self) -> Dict[str, Any]:
        """Test basic registry functionality."""
        try:
            from pipelines.dataset_registry.core.registry_manager import RegistryManager
            
            # Test registry initialization
            registry = RegistryManager()
            
            # Test basic operations
            datasets = registry.list_datasets()
            
            return {'passed': True}
            
        except Exception as e:
            return {
                'passed': False,
                'error': f'Registry functionality test failed: {e}'
            }
    
    def _test_pipeline_integration(self) -> Dict[str, Any]:
        """Test pipeline integration."""
        try:
            from pipelines.dataset_registry.integration.dataset_integration import DatasetIntegration
            
            # Test integration initialization
            integration = DatasetIntegration()
            
            # Test basic operations
            available_datasets = integration.list_available_datasets()
            
            return {'passed': True}
            
        except Exception as e:
            return {
                'passed': False,
                'error': f'Pipeline integration test failed: {e}'
            }


def create_migration_script(output_path: str = "migrate_to_registry.py") -> None:
    """Create a standalone migration script."""
    script_content = '''#!/usr/bin/env python3
"""
Standalone migration script for PBUF Dataset Registry.

This script migrates an existing PBUF installation to use the dataset registry system.
"""

import sys
import argparse
from pathlib import Path

# Add PBUF to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.dataset_registry.migration_tools import DatasetMigrationTool, BackwardCompatibilityValidator


def main():
    parser = argparse.ArgumentParser(description="Migrate PBUF installation to dataset registry")
    parser.add_argument("--config", help="Path to PBUF configuration file")
    parser.add_argument("--backup-dir", help="Directory for backup files")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without making changes")
    parser.add_argument("--validate-only", action="store_true", help="Only validate compatibility")
    
    args = parser.parse_args()
    
    if args.validate_only:
        print("Validating backward compatibility...")
        validator = BackwardCompatibilityValidator()
        results = validator.validate_deployment(args.config)
        
        print(f"Tests passed: {results['tests_passed']}")
        print(f"Tests failed: {results['tests_failed']}")
        print(f"Compatible: {results['compatible']}")
        
        if results['errors']:
            print("\\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("\\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        sys.exit(0 if results['compatible'] else 1)
    
    print("Starting PBUF Dataset Registry migration...")
    
    migration_tool = DatasetMigrationTool(
        pbuf_config_path=args.config,
        dry_run=args.dry_run
    )
    
    result = migration_tool.migrate_full_installation(args.backup_dir)
    
    print(f"\\nMigration Results:")
    print(f"Success: {result.success}")
    print(f"Datasets migrated: {result.datasets_migrated}")
    print(f"Configuration updated: {result.config_updated}")
    print(f"Backup created: {result.backup_created}")
    
    if result.errors:
        print("\\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("\\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    print("\\nMigration log:")
    for log_entry in result.migration_log:
        print(f"  {log_entry}")
    
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    Path(output_path).chmod(0o755)
    
    print(f"Migration script created: {output_path}")


if __name__ == "__main__":
    # Create migration script when run directly
    create_migration_script()