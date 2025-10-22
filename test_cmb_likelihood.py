#!/usr/bin/env python3
"""
Test script for CMB likelihood function implementation.
"""

import sys
import numpy as np
sys.path.append('pipelines')

from fit_core.likelihoods import likelihood_cmb
from fit_core.datasets import load_dataset
from fit_core.parameter import build_params

def test_cmb_likelihood():
    """Test CMB likelihood computation."""
    print("Testing CMB likelihood function...")
    
    # Build test parameters
    params = build_params("lcdm")
    print(f"Parameters: {params}")
    
    # Load CMB dataset
    data = load_dataset("cmb")
    print(f"Dataset keys: {list(data.keys())}")
    print(f"Observations: {data['observations']}")
    
    # Compute likelihood
    chi2, predictions = likelihood_cmb(params, data)
    
    print(f"Chi-squared: {chi2}")
    print(f"Predictions: {predictions}")
    
    # Basic validation
    assert isinstance(chi2, float), "Chi-squared should be a float"
    assert chi2 >= 0, "Chi-squared should be non-negative"
    assert isinstance(predictions, dict), "Predictions should be a dictionary"
    assert "R" in predictions, "Predictions should contain R"
    assert "l_A" in predictions, "Predictions should contain l_A"
    assert "theta_star" in predictions, "Predictions should contain theta_star"
    
    print("✓ CMB likelihood test passed!")
    return chi2, predictions

if __name__ == "__main__":
    test_cmb_likelihood()