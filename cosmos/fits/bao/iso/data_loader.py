"""Load isotropic BAO measurements from standardized caches."""

from cosmos.fits._dataset_loader import load_bao_iso_dataset


def load_bao_iso_data():
    """Load standardized BAO isotropic dataset."""
    return load_bao_iso_dataset()
