#!/usr/bin/env python3
"""
Test script for supernova likelihood function.
"""

import sys
import numpy as np
sys.path.append('pipelines')

from fit_core.likelihoods import likelihood_sn
from fit_core.datasets import load_dataset
from fit_core.parameter import build_params

def test_sn_likelihood():
    """Test supernova likelihood computation."""
    print("Testing supernova likelihood function...")
    
    # Build test parameters
    params = build_params("lcdm")
    print(f"Parameters: H0={params['H0']}, Om0={params['Om0']}")
    
    # Load supernova dataset
    data = load_dataset("sn")
    print(f"Dataset keys: {list(data.keys())}")
    print(f"Observations keys: {list(data['observations'].keys())}")
    print(f"Number of supernovae: {len(data['observations']['redshift'])}")
    print(f"Redshift range: {data['observations']['redshift'].min():.3f} - {data['observations']['redshift'].max():.3f}")
    print(f"Distance modulus range: {data['observations']['distance_modulus'].min():.2f} - {data['observations']['distance_modulus'].max():.2f}")
    
    # Compute likelihood
    chi2, predictions = likelihood_sn(params, data)
    
    print(f"Chi-squared: {chi2}")
    print(f"Predictions shape: {predictions['distance_modulus'].shape}")
    print(f"Predicted distance modulus range: {predictions['distance_modulus'].min():.2f} - {predictions['distance_modulus'].max():.2f}")
    
    # Compare a few values
    obs_mu = data['observations']['distance_modulus']
    pred_mu = predictions['distance_modulus']
    
    print(f"\nComparison (first 5 supernovae):")
    for i in range(min(5, len(obs_mu))):
        z = data['observations']['redshift'][i]
        print(f"  z={z:.3f}: observed μ={obs_mu[i]:.2f}, predicted μ={pred_mu[i]:.2f}, diff={obs_mu[i]-pred_mu[i]:.2f}")
    
    # Basic validation
    assert isinstance(chi2, float), "Chi-squared should be a float"
    assert chi2 >= 0, "Chi-squared should be non-negative"
    assert isinstance(predictions, dict), "Predictions should be a dictionary"
    assert "distance_modulus" in predictions, "Predictions should contain distance_modulus"
    assert len(predictions["distance_modulus"]) == len(data["observations"]["redshift"]), "Prediction length should match data"
    
    print("✓ Supernova likelihood test passed!")

if __name__ == "__main__":
    test_sn_likelihood()