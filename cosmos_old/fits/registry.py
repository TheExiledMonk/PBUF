"""Registry mapping dataset names to their joint-fit callables."""

from __future__ import annotations

from typing import Callable, Dict

from cosmos.fits.bao_aniso import run_fit as run_bao_aniso_fit
from cosmos.fits.bao_iso import run_fit as run_bao_iso_fit
from cosmos.fits.cc import run_fit as run_cc_fit
from cosmos.fits.cmb import run_fit as run_cmb_fit
from cosmos.fits.galaxy_pk import run_fit as run_galaxy_pk_fit
from cosmos.fits.lensing_cross import run_fit as run_lensing_cross_fit
from cosmos.fits.rsd import run_fit as run_rsd_fit
from cosmos.fits.sh0es import run_fit as run_sh0es_prior
from cosmos.fits.sn import run_fit as run_sn_pantheon_fit
from cosmos.fits.wl import run_fit as run_wl_s8_fit

FIT_REGISTRY: Dict[str, Callable] = {
    "cmb": run_cmb_fit,
    "sn": run_sn_pantheon_fit,
    "sh0es": run_sh0es_prior,
    "bao_iso": run_bao_iso_fit,
    "bao_aniso": run_bao_aniso_fit,
    "cc": run_cc_fit,
    "rsd": run_rsd_fit,
    "wl_s8": run_wl_s8_fit,
    "galaxy_pk": run_galaxy_pk_fit,
    "lensing_x": run_lensing_cross_fit,
}
