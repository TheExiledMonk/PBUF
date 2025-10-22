#!/usr/bin/env python3
"""
Test script for BAO likelihood functions.
"""

import sys
import numpy as np
sys.path.append('pipelines')

from fit_core.likelihoods import likelihood_bao, likelihood_bao_ani
from fit_core.datasets import load_dataset
from fit_core.parameter import build_params

def test_bao_likelihood():
    """Test isotropic BAO likelihood computation."""
    print("Testing isotropic BAO likelihood function...")
    
    # Build test parameters
    params = build_params("lcdm")
    print(f"Parameters: H0={params['H0']}, Om0={params['Om0']}, r_s_drag={params['r_s_drag']}")
    
    # Load BAO dataset
    data = load_dataset("bao")
    print(f"Dataset keys: {list(data.keys())}")
    print(f"Observations: {data['observations']}")
    
    # Compute likelihood
    chi2, predictions = likelihood_bao(params, data)
    
    print(f"Chi-squared: {chi2}")
    print(f"Predictions: {predictions}")
    
    # Basic validation
    assert isinstance(chi2, float), "Chi-squared should be a float"
    assert chi2 >= 0, "Chi-squared should be non-negative"
    assert isinstance(predictions, dict), "Predictions should be a dictionary"
    assert "DV_over_rs" in predictions, "Predictions should contain DV_over_rs"
    
    print("✓ Isotropic BAO likelihood test passed!")

def test_bao_ani_likelihood():
    """Test anisotropic BAO likelihood computation."""
    print("\nTesting anisotropic BAO likelihood function...")
    
    # Build test parameters
    params = build_params("lcdm")
    
    # Load anisotropic BAO dataset
    data = load_dataset("bao_ani")
    print(f"Dataset keys: {list(data.keys())}")
    print(f"Observations: {data['observations']}")
    
    # Compute likelihood
    chi2, predictions = likelihood_bao_ani(params, data)
    
    print(f"Chi-squared: {chi2}")
    print(f"Predictions: {predictions}")
    
    # Basic validation
    assert isinstance(chi2, float), "Chi-squared should be a float"
    assert chi2 >= 0, "Chi-squared should be non-negative"
    assert isinstance(predictions, dict), "Predictions should be a dictionary"
    assert "DM_over_rd" in predictions, "Predictions should contain DM_over_rd"
    assert "DH_over_rd" in predictions, "Predictions should contain DH_over_rd"
    
    print("✓ Anisotropic BAO likelihood test passed!")

if __name__ == "__main__":
    test_bao_likelihood()
    test_bao_ani_likelihood()