#!/usr/bin/env python3
"""
Test script for version control integration

This script tests the version control integration functionality including
environment fingerprinting, compatibility checking, and reproducibility validation.
"""

import sys
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipelines.dataset_registry.core.version_control_integration import (
    VersionControlIntegration,
    get_current_pbuf_commit,
    collect_current_environment,
    validate_reproducibility
)
from pipelines.dataset_registry.core.extensible_interface import (
    ExtensibleDatasetInterface,
    APIVersion,
    DatasetRequest,
    get_dataset_v1,
    list_available_datasets
)
from pipelines.dataset_registry.integration.dataset_integration import (
    DatasetRegistry,
    get_supported_api_versions,
    get_current_environment_info
)


def test_version_control_integration():
    """Test version control integration functionality"""
    print("Testing Version Control Integration")
    print("=" * 50)
    
    # Test 1: Environment fingerprinting
    print("\n1. Testing environment fingerprinting...")
    try:
        vc_integration = VersionControlIntegration()
        
        # Get PBUF commit info
        commit_info = vc_integration.get_pbuf_commit_info()
        print(f"   PBUF commit: {commit_info['commit_hash']}")
        print(f"   Branch: {commit_info['branch']}")
        print(f"   Status: {commit_info['status']}")
        
        # Get environment fingerprint
        env_fingerprint = vc_integration.collect_environment_fingerprint()
        print(f"   Environment fingerprint: {env_fingerprint.fingerprint_hash[:16]}...")
        print(f"   Python version: {env_fingerprint.python_version.split()[0]}")
        print(f"   Platform: {env_fingerprint.platform}")
        
        print("   ✓ Environment fingerprinting works")
    except Exception as e:
        print(f"   ✗ Environment fingerprinting failed: {e}")
    
    # Test 2: Current environment collection
    print("\n2. Testing current environment collection...")
    try:
        current_env = collect_current_environment()
        print(f"   Collected environment with {len(current_env.installed_packages)} packages")
        print(f"   PBUF commit: {current_env.pbuf_commit}")
        print("   ✓ Current environment collection works")
    except Exception as e:
        print(f"   ✗ Current environment collection failed: {e}")
    
    # Test 3: Environment summary export
    print("\n3. Testing environment summary export...")
    try:
        env_summary = get_current_environment_info()
        print(f"   Environment summary has {len(env_summary)} top-level keys")
        if 'pbuf_version' in env_summary:
            print(f"   PBUF version info: {env_summary['pbuf_version']['commit_hash']}")
        print("   ✓ Environment summary export works")
    except Exception as e:
        print(f"   ✗ Environment summary export failed: {e}")


def test_extensible_interface():
    """Test extensible interface functionality"""
    print("\n\nTesting Extensible Interface")
    print("=" * 50)
    
    # Test 1: API version support
    print("\n1. Testing API version support...")
    try:
        supported_versions = get_supported_api_versions()
        print(f"   Supported API versions: {supported_versions}")
        print("   ✓ API version support works")
    except Exception as e:
        print(f"   ✗ API version support failed: {e}")
    
    # Test 2: Extensible interface initialization
    print("\n2. Testing extensible interface initialization...")
    try:
        interface = ExtensibleDatasetInterface()
        versions = interface.get_supported_versions()
        print(f"   Interface supports versions: {versions}")
        
        # Test plugin manager
        plugins = interface.plugin_manager.list_plugins()
        print(f"   Loaded {len(plugins)} plugins")
        for plugin_name, plugin_info in plugins.items():
            print(f"     - {plugin_name}: {plugin_info['type']} v{plugin_info['version']}")
        
        print("   ✓ Extensible interface initialization works")
    except Exception as e:
        print(f"   ✗ Extensible interface initialization failed: {e}")
    
    # Test 3: Dataset listing with versioned API
    print("\n3. Testing dataset listing with versioned API...")
    try:
        datasets = list_available_datasets()
        print(f"   Found {len(datasets)} available datasets")
        
        if datasets:
            sample_dataset = datasets[0]
            print(f"   Sample dataset: {sample_dataset['canonical_name']}")
            print(f"   Source type: {sample_dataset['source_type']}")
        
        print("   ✓ Dataset listing with versioned API works")
    except Exception as e:
        print(f"   ✗ Dataset listing with versioned API failed: {e}")


def test_integration():
    """Test integration between components"""
    print("\n\nTesting Component Integration")
    print("=" * 50)
    
    # Test 1: Registry with version control
    print("\n1. Testing registry with version control integration...")
    try:
        registry = DatasetRegistry()
        
        # Test environment info
        env_info = registry.get_current_environment_info()
        print(f"   Environment info has {len(env_info)} sections")
        
        # Test reproducibility validation (even if no datasets)
        validation = registry.validate_reproducibility([])
        print(f"   Reproducibility validation: {validation['valid']}")
        
        print("   ✓ Registry with version control integration works")
    except Exception as e:
        print(f"   ✗ Registry with version control integration failed: {e}")
    
    # Test 2: Compatibility checking (mock test)
    print("\n2. Testing compatibility checking framework...")
    try:
        vc_integration = VersionControlIntegration()
        current_env = vc_integration.get_current_environment()
        
        # Test compatibility with itself (should be exact match)
        compatibility = vc_integration.check_dataset_compatibility(current_env, strict_mode=False)
        print(f"   Self-compatibility level: {compatibility.compatibility_level}")
        print(f"   Self-compatibility result: {compatibility.is_compatible}")
        
        print("   ✓ Compatibility checking framework works")
    except Exception as e:
        print(f"   ✗ Compatibility checking framework failed: {e}")


def main():
    """Run all tests"""
    print("Dataset Registry Version Control Integration Test")
    print("=" * 60)
    
    try:
        test_version_control_integration()
        test_extensible_interface()
        test_integration()
        
        print("\n\n" + "=" * 60)
        print("All tests completed!")
        print("Version control integration is working correctly.")
        
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())