#!/usr/bin/env python3
"""
Deployment validation script for the PBUF Dataset Registry system.

This script validates that the dataset registry is properly deployed and configured,
ensuring all components are working correctly and maintaining backward compatibility.
"""

import sys
import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .core.config import DatasetRegistryConfigManager
from .core.config_integration import validate_integrated_configuration
from .migration_tools import BackwardCompatibilityValidator


class DeploymentValidator:
    """Comprehensive deployment validation for the dataset registry system."""
    
    def __init__(self, config_path: str = None, verbose: bool = False):
        """
        Initialize deployment validator.
        
        Args:
            config_path: Path to configuration file
            verbose: Enable verbose output
        """
        self.config_path = config_path
        self.verbose = verbose
        self.validation_results = []
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive deployment validation.
        
        Returns:
            Dictionary with complete validation results
        """
        results = {
            'overall_status': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'config_path': self.config_path,
            'tests': {},
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            },
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Test categories
        test_categories = [
            ('configuration', self._validate_configuration),
            ('file_system', self._validate_file_system),
            ('registry_core', self._validate_registry_core),
            ('integration', self._validate_integration),
            ('backward_compatibility', self._validate_backward_compatibility),
            ('performance', self._validate_performance),
            ('security', self._validate_security)
        ]
        
        for category, test_func in test_categories:
            self._log(f"Running {category} validation...")
            
            try:
                test_result = test_func()
                results['tests'][category] = test_result
                
                # Update summary
                results['summary']['total_tests'] += 1
                if test_result['status'] == 'PASS':
                    results['summary']['passed'] += 1
                elif test_result['status'] == 'FAIL':
                    results['summary']['failed'] += 1
                    results['errors'].extend(test_result.get('errors', []))
                
                if test_result.get('warnings'):
                    results['summary']['warnings'] += len(test_result['warnings'])
                    results['warnings'].extend(test_result['warnings'])
                
                if test_result.get('recommendations'):
                    results['recommendations'].extend(test_result['recommendations'])
                
            except Exception as e:
                results['tests'][category] = {
                    'status': 'ERROR',
                    'errors': [f"Test execution failed: {e}"]
                }
                results['summary']['failed'] += 1
                results['errors'].append(f"{category} validation failed: {e}")
        
        # Determine overall status
        if results['summary']['failed'] == 0:
            results['overall_status'] = 'PASS'
        elif results['summary']['passed'] > results['summary']['failed']:
            results['overall_status'] = 'PARTIAL'
        else:
            results['overall_status'] = 'FAIL'
        
        return results
    
    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration setup."""
        result = {
            'status': 'PASS',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Test configuration loading
            if self.config_path:
                config_manager = DatasetRegistryConfigManager(self.config_path)
                config = config_manager.get_config()
                result['details']['config_loaded'] = True
                
                # Validate configuration
                validation = config_manager.validate_config()
                result['details']['config_valid'] = validation['valid']
                
                if not validation['valid']:
                    result['status'] = 'FAIL'
                    result['errors'].extend(validation['errors'])
                
                if validation['warnings']:
                    result['warnings'].extend(validation['warnings'])
            
            # Test PBUF integration
            try:
                from pipelines.fit_core.config import ConfigurationManager
                
                if self.config_path and Path(self.config_path).exists():
                    pbuf_config = ConfigurationManager(self.config_path)
                    registry_config = pbuf_config.get_dataset_registry_config()
                    result['details']['pbuf_integration'] = True
                else:
                    result['warnings'].append("No PBUF configuration file found")
                    result['details']['pbuf_integration'] = False
                
            except Exception as e:
                result['warnings'].append(f"PBUF integration test failed: {e}")
                result['details']['pbuf_integration'] = False
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Configuration validation failed: {e}")
        
        return result
    
    def _validate_file_system(self) -> Dict[str, Any]:
        """Validate file system setup and permissions."""
        result = {
            'status': 'PASS',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        # Required directories
        required_dirs = [
            'data',
            'data/registry',
            'data/cache'
        ]
        
        for dir_path in required_dirs:
            path_obj = Path(dir_path)
            
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    result['details'][f'{dir_path}_created'] = True
                except Exception as e:
                    result['status'] = 'FAIL'
                    result['errors'].append(f"Cannot create directory {dir_path}: {e}")
                    continue
            
            # Check permissions
            if not os.access(path_obj, os.R_OK | os.W_OK):
                result['status'] = 'FAIL'
                result['errors'].append(f"Insufficient permissions for {dir_path}")
            else:
                result['details'][f'{dir_path}_permissions'] = 'OK'
        
        # Check manifest file
        manifest_path = Path('data/datasets_manifest.json')
        if manifest_path.exists():
            result['details']['manifest_exists'] = True
            try:
                with open(manifest_path, 'r') as f:
                    json.load(f)
                result['details']['manifest_valid'] = True
            except Exception as e:
                result['warnings'].append(f"Manifest file is invalid: {e}")
                result['details']['manifest_valid'] = False
        else:
            result['warnings'].append("Dataset manifest not found")
            result['details']['manifest_exists'] = False
        
        return result
    
    def _validate_registry_core(self) -> Dict[str, Any]:
        """Validate core registry functionality."""
        result = {
            'status': 'PASS',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Test registry manager initialization
            from pipelines.dataset_registry.core.registry_manager import RegistryManager
            
            registry = RegistryManager()
            result['details']['registry_init'] = True
            
            # Test basic operations
            datasets = registry.list_datasets()
            result['details']['dataset_count'] = len(datasets)
            
            # Test manifest loading
            from pipelines.dataset_registry.core.manifest_schema import DatasetManifest
            
            manifest_path = Path('data/datasets_manifest.json')
            if manifest_path.exists():
                manifest = DatasetManifest(str(manifest_path))
                available_datasets = manifest.list_datasets()
                result['details']['manifest_datasets'] = len(available_datasets)
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Registry core validation failed: {e}")
        
        return result
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate integration with PBUF pipelines."""
        result = {
            'status': 'PASS',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Test dataset integration
            from pipelines.dataset_registry.integration.dataset_integration import DatasetIntegration
            
            integration = DatasetIntegration()
            available_datasets = integration.list_available_datasets()
            result['details']['integration_datasets'] = len(available_datasets)
            
            # Test dataset loading interface
            from pipelines.fit_core.datasets import load_dataset
            result['details']['load_dataset_available'] = True
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Integration validation failed: {e}")
        
        return result
    
    def _validate_backward_compatibility(self) -> Dict[str, Any]:
        """Validate backward compatibility."""
        result = {
            'status': 'PASS',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            validator = BackwardCompatibilityValidator()
            compat_results = validator.validate_deployment(self.config_path)
            
            result['details']['compatibility_tests_passed'] = compat_results['tests_passed']
            result['details']['compatibility_tests_failed'] = compat_results['tests_failed']
            result['details']['compatible'] = compat_results['compatible']
            
            if not compat_results['compatible']:
                result['status'] = 'FAIL'
                result['errors'].extend(compat_results['errors'])
            
            if compat_results['warnings']:
                result['warnings'].extend(compat_results['warnings'])
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Backward compatibility validation failed: {e}")
        
        return result
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics."""
        result = {
            'status': 'PASS',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            import time
            
            # Test configuration loading performance
            start_time = time.time()
            config_manager = DatasetRegistryConfigManager(self.config_path)
            config_load_time = time.time() - start_time
            
            result['details']['config_load_time_ms'] = round(config_load_time * 1000, 2)
            
            if config_load_time > 1.0:
                result['warnings'].append("Configuration loading is slow (>1s)")
            
            # Test registry operations performance
            from pipelines.dataset_registry.core.registry_manager import RegistryManager
            
            start_time = time.time()
            registry = RegistryManager()
            datasets = registry.list_datasets()
            registry_time = time.time() - start_time
            
            result['details']['registry_list_time_ms'] = round(registry_time * 1000, 2)
            
            if registry_time > 2.0:
                result['warnings'].append("Registry operations are slow (>2s)")
        
        except Exception as e:
            result['warnings'].append(f"Performance validation failed: {e}")
        
        return result
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security configuration."""
        result = {
            'status': 'PASS',
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'details': {}
        }
        
        try:
            # Check file permissions
            sensitive_files = [
                'pbuf_config.json',
                'data/registry/',
                'data/cache/'
            ]
            
            for file_path in sensitive_files:
                path_obj = Path(file_path)
                if path_obj.exists():
                    stat_info = path_obj.stat()
                    permissions = oct(stat_info.st_mode)[-3:]
                    
                    result['details'][f'{file_path}_permissions'] = permissions
                    
                    # Check for overly permissive permissions
                    if path_obj.is_file() and permissions in ['777', '666']:
                        result['warnings'].append(f"File {file_path} has overly permissive permissions: {permissions}")
                    elif path_obj.is_dir() and permissions == '777':
                        result['warnings'].append(f"Directory {file_path} has overly permissive permissions: {permissions}")
            
            # Check configuration security settings
            if self.config_path:
                config_manager = DatasetRegistryConfigManager(self.config_path)
                config = config_manager.get_config()
                
                if not config.verify_checksums:
                    result['warnings'].append("Checksum verification is disabled")
                
                if config.checksum_algorithm not in ['sha256', 'sha512']:
                    result['warnings'].append(f"Weak checksum algorithm: {config.checksum_algorithm}")
                
                result['details']['checksum_verification'] = config.verify_checksums
                result['details']['checksum_algorithm'] = config.checksum_algorithm
        
        except Exception as e:
            result['warnings'].append(f"Security validation failed: {e}")
        
        return result


def main():
    """Main entry point for deployment validation."""
    parser = argparse.ArgumentParser(description="Validate PBUF Dataset Registry deployment")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", help="Output file for validation results (JSON)")
    parser.add_argument("--category", choices=['configuration', 'file_system', 'registry_core', 
                                              'integration', 'backward_compatibility', 'performance', 'security'],
                       help="Run only specific validation category")
    
    args = parser.parse_args()
    
    validator = DeploymentValidator(config_path=args.config, verbose=args.verbose)
    
    if args.category:
        # Run single category
        test_func = getattr(validator, f'_validate_{args.category}')
        result = test_func()
        
        print(f"{args.category.upper()} Validation: {result['status']}")
        
        if result.get('errors'):
            print("Errors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        if result.get('warnings'):
            print("Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        sys.exit(0 if result['status'] in ['PASS', 'PARTIAL'] else 1)
    
    else:
        # Run full validation
        results = validator.run_full_validation()
        
        print(f"Dataset Registry Deployment Validation")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Tests: {results['summary']['passed']}/{results['summary']['total_tests']} passed")
        
        if results['summary']['warnings'] > 0:
            print(f"Warnings: {results['summary']['warnings']}")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
        
        sys.exit(0 if results['overall_status'] in ['PASS', 'PARTIAL'] else 1)


if __name__ == "__main__":
    main()