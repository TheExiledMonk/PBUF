#!/usr/bin/env python3
"""
Test script to verify PBUF cosmology pipeline setup.

This script tests that all core modules can be imported and basic
functionality is available.
"""

import sys
import os

# Add pipelines to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipelines'))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test core type definitions
        from fit_core import ParameterDict, ResultsDict, DatasetDict, MetricsDict, PredictionsDict
        print("  ✅ Core type definitions imported")
        
        # Test individual modules
        import fit_core.engine as engine
        print("  ✅ Engine module imported")
        
        import fit_core.parameter as parameter
        print("  ✅ Parameter module imported")
        
        import fit_core.likelihoods as likelihoods
        print("  ✅ Likelihoods module imported")
        
        import fit_core.datasets as datasets
        print("  ✅ Datasets module imported")
        
        import fit_core.statistics as statistics
        print("  ✅ Statistics module imported")
        
        import fit_core.logging_utils as logging_utils
        print("  ✅ Logging utils module imported")
        
        import fit_core.integrity as integrity
        print("  ✅ Integrity module imported")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_parameter_defaults():
    """Test that parameter defaults are accessible."""
    print("\n🔧 Testing parameter defaults...")
    
    try:
        import fit_core.parameter as param
        
        # Check that DEFAULTS dictionary exists
        if hasattr(param, 'DEFAULTS'):
            print("  ✅ DEFAULTS dictionary found")
            
            # Check for expected models
            if 'lcdm' in param.DEFAULTS and 'pbuf' in param.DEFAULTS:
                print("  ✅ Both LCDM and PBUF defaults defined")
                
                # Check some expected parameters
                lcdm_params = param.DEFAULTS['lcdm']
                expected_params = ['H0', 'Om0', 'Obh2', 'ns']
                
                missing_params = [p for p in expected_params if p not in lcdm_params]
                if not missing_params:
                    print("  ✅ Expected LCDM parameters found")
                else:
                    print(f"  ⚠️  Missing LCDM parameters: {missing_params}")
                
                return True
            else:
                print("  ❌ Missing model definitions in DEFAULTS")
                return False
        else:
            print("  ❌ DEFAULTS dictionary not found")
            return False
            
    except Exception as e:
        print(f"  ❌ Parameter test failed: {e}")
        return False

def test_scientific_dependencies():
    """Test that scientific dependencies are available."""
    print("\n📊 Testing scientific dependencies...")
    
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__} available")
        
        import scipy
        print(f"  ✅ SciPy {scipy.__version__} available")
        
        import matplotlib
        print(f"  ✅ Matplotlib {matplotlib.__version__} available")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Scientific dependency missing: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 PBUF Cosmology Pipeline Setup Test")
    print("=" * 50)
    
    tests = [
        test_scientific_dependencies,
        test_imports,
        test_parameter_defaults
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Pipeline setup is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())