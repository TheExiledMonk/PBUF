"""
Load redshift-space distortion (RSD) fσ8(z) data from standardized caches.

This thin wrapper keeps legacy callers working while ensuring the
fits use the canonical ``data/standardized/rsd*.npz`` files.
"""

from cosmos.fits._dataset_loader import load_rsd_dataset


def load_rsd_data():
    """Load standardized RSD dataset."""
    return load_rsd_dataset()
