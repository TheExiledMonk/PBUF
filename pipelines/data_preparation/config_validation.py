#!/usr/bin/env python3
"""
Configuration validation script for data preparation framework.

This script validates configuration files and deployment settings,
providing detailed feedback on configuration issues and recommendations
for production deployment.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from core.cmb_config_integration import (
    DataPreparationConfigManager,
    validate_deployment_config,
    print_environment_variables_help,
    create_example_data_preparation_config
)


def validate_config_file(config_file: str, deployment_check: bool = False) -> bool:
    """
    Validate a configuration file.
    
    Args:
        config_file: Path to configuration file
        deployment_check: Whether to perform deployment-specific validation
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        if deployment_check:
            validation_results = validate_deployment_config(config_file)
        else:
            manager = DataPreparationConfigManager(config_file)
            validation_results = manager.validate_config()
        
        print(f"Configuration validation results for: {config_file}")
        print("=" * 60)
        
        if validation_results["valid"]:
            print("✓ Configuration is VALID")
        else:
            print("✗ Configuration is INVALID")
        
        if validation_results["errors"]:
            print(f"\nErrors ({len(validation_results['errors'])}):")
            for i, error in enumerate(validation_results["errors"], 1):
                print(f"  {i}. {error}")
        
        if validation_results["warnings"]:
            print(f"\nWarnings ({len(validation_results['warnings'])}):")
            for i, warning in enumerate(validation_results["warnings"], 1):
                print(f"  {i}. {warning}")
        
        if not validation_results["errors"] and not validation_results["warnings"]:
            print("\nNo issues found.")
        
        return validation_results["valid"]
        
    except Exception as e:
        print(f"Error validating configuration file {config_file}: {e}")
        return False


def show_config_structure(config_file: str) -> None:
    """
    Display the structure of a configuration file.
    
    Args:
        config_file: Path to configuration file
    """
    try:
        manager = DataPreparationConfigManager(config_file)
        config = manager.get_config()
        
        print(f"Configuration structure for: {config_file}")
        print("=" * 60)
        
        config_dict = config.to_dict()
        print(json.dumps(config_dict, indent=2))
        
    except Exception as e:
        print(f"Error reading configuration file {config_file}: {e}")


def check_environment_config() -> None:
    """Check configuration from environment variables."""
    import os
    
    print("Environment Variable Configuration Check")
    print("=" * 60)
    
    # Check for relevant environment variables
    env_vars = [
        "PBUF_OUTPUT_PATH", "PBUF_CACHE_PATH", "PBUF_FRAMEWORK_ENABLED",
        "PBUF_AUTO_PROCESS", "PBUF_VALIDATE_OUTPUTS", "PBUF_LOG_LEVEL",
        "PBUF_ENV", "CMB_USE_RAW_PARAMETERS", "CMB_Z_RECOMBINATION",
        "CMB_JACOBIAN_STEP_SIZE", "CMB_FALLBACK_TO_LEGACY"
    ]
    
    found_vars = []
    for var in env_vars:
        if var in os.environ:
            found_vars.append((var, os.environ[var]))
    
    if found_vars:
        print("Found environment variables:")
        for var, value in found_vars:
            print(f"  {var} = {value}")
    else:
        print("No relevant environment variables found.")
    
    # Try to create configuration from environment
    try:
        manager = DataPreparationConfigManager()
        config = manager.get_config()
        
        print("\nResulting configuration:")
        print("-" * 30)
        
        # Show key configuration values
        print(f"Framework enabled: {config.framework_enabled}")
        print(f"Log level: {config.log_level}")
        print(f"CMB raw parameters: {config.cmb.use_raw_parameters}")
        print(f"CMB fallback to legacy: {config.cmb.fallback_to_legacy}")
        print(f"Performance monitoring: {config.performance_monitoring}")
        
        # Validate environment-based configuration
        validation_results = manager.validate_config()
        if not validation_results["valid"]:
            print("\nEnvironment configuration issues:")
            for error in validation_results["errors"]:
                print(f"  ✗ {error}")
        else:
            print("\n✓ Environment configuration is valid")
            
    except Exception as e:
        print(f"\nError creating configuration from environment: {e}")


def create_deployment_configs() -> None:
    """Create example deployment configuration files."""
    configs = {
        "development": {
            "data_preparation": {
                "framework_enabled": True,
                "output_path": "data/prepared/",
                "cache_path": "data/cache/",
                "auto_process": True,
                "validate_outputs": False,
                "fallback_to_legacy": True,
                "performance": {
                    "parallel_processing": False,
                    "max_workers": 2,
                    "memory_limit_gb": 4.0
                },
                "logging": {
                    "structured_logging": True,
                    "log_level": "DEBUG",
                    "log_file": "logs/data_preparation_dev.log",
                    "performance_monitoring": True
                },
                "derivation": {
                    "cmb": {
                        "use_raw_parameters": True,
                        "z_recombination": 1089.8,
                        "jacobian_step_size": 1e-5,
                        "validation_tolerance": 1e-7,
                        "fallback_to_legacy": True,
                        "cache_computations": False,
                        "performance_monitoring": True
                    }
                }
            }
        },
        "production": {
            "data_preparation": {
                "framework_enabled": True,
                "output_path": "/data/pbuf/prepared/",
                "cache_path": "/data/pbuf/cache/",
                "auto_process": True,
                "validate_outputs": True,
                "fallback_to_legacy": True,
                "performance": {
                    "parallel_processing": True,
                    "max_workers": 8,
                    "memory_limit_gb": 16.0
                },
                "logging": {
                    "structured_logging": True,
                    "log_level": "INFO",
                    "log_file": "/var/log/pbuf/data_preparation.log",
                    "performance_monitoring": False
                },
                "derivation": {
                    "cmb": {
                        "use_raw_parameters": True,
                        "z_recombination": 1089.8,
                        "jacobian_step_size": 1e-6,
                        "validation_tolerance": 1e-8,
                        "fallback_to_legacy": True,
                        "cache_computations": True,
                        "performance_monitoring": False
                    }
                }
            }
        },
        "testing": {
            "data_preparation": {
                "framework_enabled": False,
                "output_path": "test_data/prepared/",
                "cache_path": "test_data/cache/",
                "auto_process": False,
                "validate_outputs": True,
                "fallback_to_legacy": True,
                "performance": {
                    "parallel_processing": False,
                    "max_workers": 1,
                    "memory_limit_gb": 2.0
                },
                "logging": {
                    "structured_logging": True,
                    "log_level": "WARNING",
                    "log_file": None,
                    "performance_monitoring": False
                },
                "derivation": {
                    "cmb": {
                        "use_raw_parameters": False,
                        "z_recombination": 1089.8,
                        "jacobian_step_size": 1e-5,
                        "validation_tolerance": 1e-7,
                        "fallback_to_legacy": True,
                        "cache_computations": False,
                        "performance_monitoring": False
                    }
                }
            }
        }
    }
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    for env_name, config_data in configs.items():
        config_file = config_dir / f"data_preparation_{env_name}.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"Created {config_file}")
    
    print(f"\nCreated {len(configs)} deployment configuration files in config/")
    print("Use with: export PBUF_ENV=development (or production/testing)")


def main():
    """Main configuration validation script."""
    parser = argparse.ArgumentParser(
        description="Validate data preparation framework configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --validate config/data_preparation.json
  %(prog)s --deployment-check config/production.json
  %(prog)s --show-structure config/development.json
  %(prog)s --check-environment
  %(prog)s --create-examples
  %(prog)s --help-env-vars
        """
    )
    
    parser.add_argument(
        "--validate", "-v",
        metavar="CONFIG_FILE",
        help="Validate configuration file"
    )
    
    parser.add_argument(
        "--deployment-check", "-d",
        metavar="CONFIG_FILE",
        help="Perform deployment-specific validation"
    )
    
    parser.add_argument(
        "--show-structure", "-s",
        metavar="CONFIG_FILE",
        help="Show configuration file structure"
    )
    
    parser.add_argument(
        "--check-environment", "-e",
        action="store_true",
        help="Check configuration from environment variables"
    )
    
    parser.add_argument(
        "--create-examples", "-c",
        action="store_true",
        help="Create example configuration files for different environments"
    )
    
    parser.add_argument(
        "--help-env-vars",
        action="store_true",
        help="Show help for environment variables"
    )
    
    args = parser.parse_args()
    
    if args.help_env_vars:
        print_environment_variables_help()
        return
    
    if args.create_examples:
        create_deployment_configs()
        return
    
    if args.check_environment:
        check_environment_config()
        return
    
    if args.show_structure:
        show_config_structure(args.show_structure)
        return
    
    if args.validate:
        success = validate_config_file(args.validate, deployment_check=False)
        sys.exit(0 if success else 1)
    
    if args.deployment_check:
        success = validate_config_file(args.deployment_check, deployment_check=True)
        sys.exit(0 if success else 1)
    
    # If no arguments provided, show help
    parser.print_help()


if __name__ == "__main__":
    main()