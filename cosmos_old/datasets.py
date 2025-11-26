"""Dataset helpers shared by the optimisation stack."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict

from cosmos.fits.bao_aniso import load_bao_aniso_dataset
from cosmos.fits.bao_iso import load_bao_iso_dataset
from cosmos.fits.cc import load_cc_dataset
from cosmos.fits.galaxy_pk import load_galaxy_pk_dataset
from cosmos.fits.rsd import load_rsd_dataset
from cosmos.fits.lensing_cross import load_lensing_cross_dataset
from cosmos.fits.wl import load_wl_s8_dataset
from fits.cmb.data_loader import CMBDataset, load_planck_priors
from fits.sh0es.sh0es_prior import load_sh0es_dataset
from fits.sn.sn_pantheon import load_sn_pantheon_dataset


Dataset = Any


_LOADERS: Dict[str, Callable[[], Dataset]] = {
    "cmb": load_planck_priors,
    "sn": load_sn_pantheon_dataset,
    "sh0es": load_sh0es_dataset,
    "bao_iso": load_bao_iso_dataset,
    "bao_aniso": load_bao_aniso_dataset,
    "cc": load_cc_dataset,
    "rsd": load_rsd_dataset,
    "wl_s8": load_wl_s8_dataset,
    "lensing_cross": load_lensing_cross_dataset,
    "galaxy_pk": load_galaxy_pk_dataset,
}


@lru_cache(maxsize=None)
def get_dataset(name: str) -> Dataset:
    """
    Return a standardized dataset by name, caching the payload in-memory to
    avoid re-reading .npz files across optimisation iterations.
    """

    normalized = name.strip().lower()
    loader = _LOADERS.get(normalized)
    if loader is None:
        raise ValueError(f"Dataset '{name}' is not supported.")
    return loader()


__all__ = ["CMBDataset", "Dataset", "get_dataset"]
