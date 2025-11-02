"""Load anisotropic BAO measurements from standardized caches."""

from cosmos.fits._dataset_loader import load_bao_aniso_dataset


def load_bao_aniso_data():
    """Load standardized BAO anisotropic dataset."""
    return load_bao_aniso_dataset()
