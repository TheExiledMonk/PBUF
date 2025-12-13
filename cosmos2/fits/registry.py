"""Registry mapping dataset names to their joint-fit callables for cosmos2."""

from __future__ import annotations

from typing import Any, Callable, Dict

from cosmos2.fits.bao_aniso import run_fit as run_bao_aniso_fit
from cosmos2.fits.bao_iso import run_fit as run_bao_iso_fit
from cosmos2.fits.cc import run_fit as run_cc_fit
from cosmos2.fits.cmb import run_fit as run_cmb_fit
from cosmos2.fits.galaxy_pk import run_fit as run_galaxy_pk_fit
from cosmos2.fits.lensing_cross import run_fit as run_lensing_cross_fit
from cosmos2.fits.rsd import run_fit as run_rsd_fit
from cosmos2.fits.sh0es import run_fit as run_sh0es_prior
from cosmos2.fits.sn import run_fit as run_sn_pantheon_fit
from cosmos2.fits.wl import run_fit as run_wl_s8_fit
from cosmos2.fits.weak_lensing_kids1000 import run_fit as run_wl_kids1000_fit

FIT_REGISTRY: Dict[str, Callable[[Any], Any]] = {
    "cmb": run_cmb_fit,
    "sn": run_sn_pantheon_fit,
    "sn_pantheon": run_sn_pantheon_fit,
    "sn_pantheonplus": run_sn_pantheon_fit,
    "sn_pantheon_shoes": run_sn_pantheon_fit,
    "sh0es": run_sh0es_prior,
    "bao_iso": run_bao_iso_fit,
    "bao_aniso": run_bao_aniso_fit,
    "cc": run_cc_fit,
    "rsd": run_rsd_fit,
    "wl_s8": run_wl_s8_fit,
    "galaxy_pk": run_galaxy_pk_fit,
    "lensing_cross": run_lensing_cross_fit,
    "lensing_x": run_lensing_cross_fit,
    "weak_lensing_kids1000": run_wl_kids1000_fit,
    "wl_kids1000": run_wl_kids1000_fit,
}


def get_fit(name: str) -> Callable[[Any], Any]:
    normalized = name.strip().lower()
    if normalized not in FIT_REGISTRY:
        raise ValueError(f"Unknown fit '{name}'.")
    return FIT_REGISTRY[normalized]


__all__ = ["FIT_REGISTRY", "get_fit"]
