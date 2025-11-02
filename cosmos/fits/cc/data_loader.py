"""
Load cosmic chronometer H(z) measurements from standardized caches.

This module wraps the shared dataset loader so that CC fits never
touch the legacy CSV inputs directly.
"""

from cosmos.fits._dataset_loader import load_cc_dataset


def load_cc_data():
    """Load standardized cosmic chronometer dataset."""
    return load_cc_dataset()
