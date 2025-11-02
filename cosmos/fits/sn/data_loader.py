"""Load supernova datasets from standardized caches."""

from cosmos.fits._dataset_loader import load_sn_dataset


def load_sn_data():
    """Load standardized SN dataset."""
    return load_sn_dataset()
