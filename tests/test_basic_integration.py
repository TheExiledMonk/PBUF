#!/usr/bin/env python3
"""
Basic integration test to verify the unified system works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_integration():
    """Test basic integration of core components."""
    
    print("Testing basic integration...")
    
    try:
        # Test parameter building
        from pipelines.fit_core.parameter import build_params, get_defaults
        
        print("✓ Parameter module imported successfully")
        
        # Test LCDM parameters
        lcdm_params = build_params("lcdm")
        print(f"✓ LCDM parameters built: {len(lcdm_params)} parameters")
        
        # Test PBUF parameters  
        pbuf_params = build_params("pbuf")
        print(f"✓ PBUF parameters built: {len(pbuf_params)} parameters")
        
        # Test parameter overrides
        override_params = build_params("lcdm", overrides={"H0": 70.0})
        assert override_params["H0"] == 70.0
        print("✓ Parameter overrides working")
        
        # Test datasets module
        from pipelines.fit_core.datasets import load_dataset
        
        print("✓ Datasets module imported successfully")
        
        # Test statistics module
        from pipelines.fit_core.statistics import chi2_generic, compute_metrics
        
        print("✓ Statistics module imported successfully")
        
        # Test engine module
        from pipelines.fit_core.engine import run_fit
        
        print("✓ Engine module imported successfully")
        
        print("\n" + "="*50)
        print("BASIC INTEGRATION TEST PASSED")
        print("All core modules imported and basic functionality verified")
        print("="*50)
        
        
    except Exception as e:
        print(f"\n❌ BASIC INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Basic integration test failed: {e}"

if __name__ == "__main__":
    success = test_basic_integration()
    sys.exit(0 if success else 1)