"""Fermi GBM trigger helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from math import isfinite
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from . import log_pipeline_error
from .event_builder import SIGMA_T_GBM_DEFAULT

DETECTOR_SELECTION_STRATEGY = "brightest"
TIME_WINDOW_S = 10.0
_NAI_DETECTORS = [f"n{i}" for i in range(12)]
_BGO_DETECTORS = ["b0", "b1"]
_ALL_DETECTORS = _NAI_DETECTORS + _BGO_DETECTORS
_KEV_TO_EV = 1.0e3
# Approximate CTIME energy channel edges (keV) for NaI detectors; derived from GBM calibration docs.
_CTIME_CHANNEL_BOUNDS_KEV = {
    0: (4.0, 12.0),
    1: (12.0, 27.0),
    2: (27.0, 50.0),
    3: (50.0, 100.0),
    4: (100.0, 300.0),
    5: (300.0, 500.0),
    6: (500.0, 1000.0),
    7: (1000.0, 2000.0),
}
_SCAT_BAND_PATTERNS = [
    ("glg_scat_all_*_pflx_band_*.fit", ("PHTFLUX", "NRGFLUX")),
    ("glg_scat_all_*_flnc_band_*.fit", ("PHTFLNC", "NRGFLNC")),
]


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fits_version_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"_v(\d+)", path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def _decode_det_mask(mask: str | None) -> List[str]:
    if not mask:
        return []
    bits = [ch for ch in mask.strip() if ch in {"0", "1"}]
    return [name for bit, name in zip(bits, _ALL_DETECTORS) if bit == "1"]


def _rank_detectors_by_rate(rates: Sequence[float] | None) -> List[Tuple[str, float]]:
    if not rates:
        return []
    ranked = sorted(
        ((detector, float(rate)) for detector, rate in zip(_NAI_DETECTORS, rates)),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked


def _select_detector(
    triggered: Sequence[str],
    ranked: Sequence[Tuple[str, float]],
    strategy: str,
) -> str | None:
    normalized = strategy.lower()
    if normalized in _ALL_DETECTORS:
        return normalized if normalized in triggered or normalized in dict(ranked) else None
    if normalized == "brightest" and ranked:
        return ranked[0][0]
    if normalized == "catalog_peak" and triggered:
        return triggered[0]
    if triggered:
        return triggered[0]
    if ranked:
        return ranked[0][0]
    return None


def _normalize_channel_range(channel_range: Any) -> Tuple[int, int] | None:
    if not channel_range:
        return None
    if isinstance(channel_range, (list, tuple)) and len(channel_range) == 2:
        try:
            lo = int(channel_range[0])
            hi = int(channel_range[1])
        except (TypeError, ValueError):
            return None
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi
    return None


def _channel_bounds_from_indices(lo: int, hi: int) -> Tuple[float, float] | None:
    try:
        lo_bounds = _CTIME_CHANNEL_BOUNDS_KEV[lo]
        hi_bounds = _CTIME_CHANNEL_BOUNDS_KEV[hi]
    except KeyError:
        return None
    return lo_bounds[0], hi_bounds[1]


def _derive_energy_from_channel_range(channel_range: Tuple[int, int] | None) -> Dict[str, Tuple[float, float] | float] | None:
    if channel_range is None:
        return None
    lo, hi = channel_range
    bounds = _channel_bounds_from_indices(lo, hi)
    if bounds is None:
        return None
    lo_keV, hi_keV = bounds
    representative_keV = 0.5 * (lo_keV + hi_keV)
    return {
        "bounds_keV": (lo_keV, hi_keV),
        "representative_keV": representative_keV,
        "E_eV": representative_keV * _KEV_TO_EV,
    }


def _load_trigger_from_json(path: Path, strategy: str) -> Dict[str, Any]:
    data = _load_json_file(path)
    try:
        trig_met = float(data["trig_met"])
        trigger_name = data["trigger_name"]
    except KeyError as exc:
        raise KeyError(f"Missing key {exc} in {path}") from exc

    sigma_t = float(data.get("sigma_t", SIGMA_T_GBM_DEFAULT))
    if sigma_t <= 0.0:
        raise ValueError(f"Invalid sigma_t {sigma_t} in {path}")

    detectors = data.get("detectors", {}) or {}
    triggered = list(detectors.keys()) if isinstance(detectors, dict) else []
    ranked = []
    if isinstance(detectors, dict) and detectors:
        metric = "significance" if strategy.lower() == "brightest" else "counts"
        ranked = sorted(
            ((name, float(details.get(metric, 0.0))) for name, details in detectors.items()),
            key=lambda item: item[1],
            reverse=True,
        )
    selected_detector = _select_detector(triggered, ranked, strategy)
    channel_range = _normalize_channel_range(data.get("channel_range"))
    energy_info = _derive_energy_from_channel_range(channel_range)
    spectral = data.get("spectral") or None
    spectral_energy_eV = None
    if spectral and "E_peak_keV" in spectral:
        try:
            spectral_energy_eV = float(spectral["E_peak_keV"]) * _KEV_TO_EV
        except (TypeError, ValueError):
            spectral_energy_eV = None

    return {
        "trigger_name": str(trigger_name),
        "trig_met": trig_met,
        "sigma_t": sigma_t,
        "detector": selected_detector,
        "detectors": detectors,
        "channel_range": channel_range,
        "channel_energy_keV": energy_info["bounds_keV"] if energy_info else None,
        "E_eV": spectral_energy_eV or (energy_info["E_eV"] if energy_info else None),
        "spectral": spectral,
        "source": str(path),
    }


def _read_tcat_metadata(path: Path) -> Dict[str, Any]:
    from astropy.io import fits

    with fits.open(path) as hdul:
        header = hdul[0].header
        trig_met = float(header.get("TRIGTIME"))
        trigger_name = header.get("OBJECT") or path.stem
        mask = header.get("DET_MASK")
        trigscal_ms = header.get("TRIGSCAL")
        sigma = SIGMA_T_GBM_DEFAULT
        if trigscal_ms is not None:
            try:
                sigma = max(SIGMA_T_GBM_DEFAULT, float(trigscal_ms) / 1000.0)
            except (TypeError, ValueError):
                sigma = SIGMA_T_GBM_DEFAULT
        channel_range = _normalize_channel_range((header.get("CHAN_LO"), header.get("CHAN_HI")))
        energy_info = _derive_energy_from_channel_range(channel_range)
        metadata = {
            "timescale_ms": trigscal_ms,
            "algorithm": header.get("TRIG_ALG"),
            "channel_range": channel_range,
            "channel_energy_keV": energy_info["bounds_keV"] if energy_info else None,
            "file": str(path),
        }
        return {
            "trigger_name": trigger_name,
            "trig_met": trig_met,
            "detector_mask": mask,
            "sigma_t": sigma,
            "channel_energy_eV": energy_info["E_eV"] if energy_info else None,
            "metadata": metadata,
        }


def _read_trigdat_locrates(path: Path) -> Sequence[float] | None:
    from astropy.io import fits

    with fits.open(path) as hdul:
        if "OB_CALC" not in hdul:
            return None
        data = hdul["OB_CALC"].data
        if data is None or len(data) == 0:
            return None
        return data[0]["LOCRATES"].tolist()


def _load_trigger_from_directory(directory: Path, strategy: str) -> Dict[str, Any]:
    tcat = _select_latest_file(directory, "glg_tcat_all_*.fit")
    if tcat is None:
        raise FileNotFoundError(f"No catalog FITS in {directory}")
    metadata = _read_tcat_metadata(tcat)
    trigdat = _select_latest_file(directory, "glg_trigdat_all_*.fit")
    locrates = _read_trigdat_locrates(trigdat) if trigdat else None
    ranked = _rank_detectors_by_rate(locrates)
    triggered = _decode_det_mask(metadata.get("detector_mask"))
    detector = _select_detector(triggered, ranked, strategy)
    channel_range = metadata["metadata"].get("channel_range")
    energy_keV = metadata["metadata"].get("channel_energy_keV")
    energy_eV = metadata.get("channel_energy_eV")
    spectral = _derive_spectral_energy(directory)
    if spectral:
        energy_eV = spectral["E_peak_keV"] * _KEV_TO_EV
    return {
        "trigger_name": metadata["trigger_name"],
        "trig_met": metadata["trig_met"],
        "sigma_t": float(metadata.get("sigma_t", SIGMA_T_GBM_DEFAULT)),
        "detector": detector,
        "detectors": {
            "triggered": triggered,
            "ranked_rates": ranked,
            "mask": metadata.get("detector_mask"),
        },
        "channel_range": channel_range,
        "channel_energy_keV": energy_keV,
        "E_eV": energy_eV,
        "spectral": spectral,
        "source": {
            "directory": str(directory),
            "tcat": metadata["metadata"]["file"],
            "trigdat": str(trigdat) if trigdat else None,
        },
    }


def _select_latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=_fits_version_key)


def _derive_spectral_energy(directory: Path) -> Dict[str, Any] | None:
    candidates: List[Dict[str, Any]] = []
    for pattern, flux_fields in _SCAT_BAND_PATTERNS:
        path = _select_latest_file(directory, pattern)
        if not path:
            continue
        info = _read_band_spectrum(path, flux_fields)
        if info:
            candidates.append(info)
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda item: (item.get("flux") is not None, item.get("flux", 0.0)),
    )
    return best


def _read_band_spectrum(path: Path, flux_fields: Sequence[str]) -> Dict[str, Any] | None:
    try:
        from astropy.io import fits
    except ImportError:  # pragma: no cover - optional dependency already validated elsewhere
        return None

    try:
        with fits.open(path) as hdul:
            table = hdul["FIT PARAMS"] if "FIT PARAMS" in hdul else hdul[2]
            best_row: Dict[str, Any] | None = None
            data = table.data
            if data is None or len(data) == 0:
                return None
            for row in data:
                epeak = _extract_param_value(row, "PARAM1")
                if epeak is None or epeak <= 0 or not isfinite(epeak):
                    continue
                flux, flux_label = _extract_flux_metric(row, flux_fields)
                candidate = {
                    "E_peak_keV": float(epeak),
                    "flux": flux,
                    "flux_field": flux_label,
                    "source": str(path),
                }
                if best_row is None:
                    best_row = candidate
                    continue
                existing_flux = best_row.get("flux")
                if flux is not None and (existing_flux is None or flux > existing_flux):
                    best_row = candidate
            return best_row
    except OSError:
        return None
    return None


def _extract_param_value(row: Any, column: str) -> float | None:
    try:
        values = row[column]
    except (KeyError, IndexError, TypeError):
        return None
    if values is None:
        return None
    try:
        first = values[0]
    except Exception:  # noqa: BLE001 - propagate gracefully for scalars
        first = values
    try:
        value = float(first)
    except (TypeError, ValueError):
        return None
    if not isfinite(value):
        return None
    return value


def _extract_flux_metric(row: Any, flux_fields: Sequence[str]) -> Tuple[float | None, str | None]:
    for name in flux_fields:
        try:
            values = row[name]
        except (KeyError, IndexError, TypeError):
            continue
        if values is None:
            continue
        try:
            first = values[0]
        except Exception:  # noqa: BLE001
            first = values
        try:
            flux = float(first)
        except (TypeError, ValueError):
            continue
        if isfinite(flux) and flux > 0:
            return flux, name
    return None, None


def load_fermi_trigger(filepath: str | Path, strategy: str = DETECTOR_SELECTION_STRATEGY) -> Dict[str, Any]:
    """Load a single Fermi trigger from JSON or FITS."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_trigger_from_json(path, strategy)
    if suffix in {".fits", ".fit"}:
        return _load_trigger_from_directory(path.parent, strategy)
    raise ValueError(f"Unsupported trigger format: {path}")


def _iter_trigger_sources(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    json_files = sorted(path.glob("*.json"))
    if json_files:
        yield from json_files
        return
    fits_files = list(path.glob("*.fit")) + list(path.glob("*.fits"))
    if fits_files:
        yield path
        return
    for directory in sorted(p for p in path.iterdir() if p.is_dir()):
        yield directory


def load_fermi_directory(dirpath: str | Path, strategy: str = DETECTOR_SELECTION_STRATEGY) -> List[Dict[str, Any]]:
    """Load Fermi triggers from JSON files or per-trigger directories."""
    base_path = Path(dirpath)
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    triggers: List[Dict[str, Any]] = []
    for source in _iter_trigger_sources(base_path):
        try:
            if source.is_file():
                triggers.append(load_fermi_trigger(source, strategy=strategy))
            else:
                triggers.append(_load_trigger_from_directory(source, strategy))
        except Exception as exc:  # pragma: no cover - logging path
            log_pipeline_error(f"Error loading {source}: {exc}")
    return triggers


__all__ = [
    "DETECTOR_SELECTION_STRATEGY",
    "TIME_WINDOW_S",
    "load_fermi_directory",
    "load_fermi_trigger",
]
