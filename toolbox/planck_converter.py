"""Planck raw likelihood extraction and conversion helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits

PLANCK_COMPONENTS = (
    "cmb_raw",
    "cmb_foregrounds",
    "cmb_masks",
    "cmb_bandpasses",
    "cmb_lensing",
    "cmb_planck_config",
)
DEFAULT_VERSION = "v1"
MODEL_FLAGS = {"model_neutral": True}

def convert_planck_raw(raw_dir: Path, output_root: Path, components: Sequence[str] | None = None) -> dict[str, object]:
    """Convert the Planck raw likelihood bundle into NPZ bundles."""
    release_root = _find_planck_release(raw_dir)
    release_label = _planck_release_label(release_root)
    selected = {comp for comp in (components or PLANCK_COMPONENTS) if comp in PLANCK_COMPONENTS}
    selected = selected or set(PLANCK_COMPONENTS)

    target_dir = output_root
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    summary: dict[str, object] = {
        "release": release_label,
        "raw_root": str(release_root),
        "extracted_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if "cmb_raw" in selected:
        payload, component_summary = _build_cmb_raw(release_root)
        path = target_dir / f"cmb_raw_{DEFAULT_VERSION}.npz"
        _save_npz_component(path, payload, "cmb_raw", release_label, component_summary)
        outputs["cmb_raw"] = str(path)

    if "cmb_foregrounds" in selected:
        payload, component_summary = _build_foregrounds(release_root)
        path = target_dir / f"cmb_foregrounds_{DEFAULT_VERSION}.npz"
        _save_npz_component(path, payload, "cmb_foregrounds", release_label, component_summary)
        outputs["cmb_foregrounds"] = str(path)

    if "cmb_masks" in selected:
        payload, component_summary = _build_masks(release_root)
        path = target_dir / f"cmb_masks_{DEFAULT_VERSION}.npz"
        _save_npz_component(path, payload, "cmb_masks", release_label, component_summary)
        outputs["cmb_masks"] = str(path)

    if "cmb_bandpasses" in selected:
        payload, component_summary = _build_bandpasses(release_root)
        path = target_dir / f"cmb_bandpasses_{DEFAULT_VERSION}.npz"
        _save_npz_component(path, payload, "cmb_bandpasses", release_label, component_summary)
        outputs["cmb_bandpasses"] = str(path)

    if "cmb_lensing" in selected:
        payload, component_summary = _build_lensing(release_root)
        path = target_dir / f"cmb_lensing_{DEFAULT_VERSION}.npz"
        _save_npz_component(path, payload, "cmb_lensing", release_label, component_summary)
        outputs["cmb_lensing"] = str(path)

    if "cmb_planck_config" in selected:
        payload, component_summary = _collect_planck_configs(release_root)
        path = target_dir / f"cmb_planck_config_{DEFAULT_VERSION}.npz"
        _save_npz_component(path, payload, "cmb_planck_config", release_label, component_summary)
        outputs["cmb_planck_config"] = str(path)

    summary["outputs"] = outputs
    print("✅ Planck conversion complete:")
    for comp, location in outputs.items():
        print(f"   - {comp}: {location}")

    return {
        "name": "planck_2018_raw",
        "release": release_label,
        "summary": summary,
        "outputs": outputs,
    }


def _save_npz_component(path: Path, payload: dict[str, object], component: str, release: str, summary: dict[str, object]) -> None:
    meta = _build_component_meta(component, release, summary)
    payload["meta"] = json.dumps(meta)
    np.savez(path, **payload)


def _build_component_meta(component: str, release: str, summary: dict[str, object]) -> dict[str, object]:
    return {
        "component": component,
        "version": DEFAULT_VERSION,
        "planck_release": release,
        "purpose": _component_purpose(component),
        "extracted_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "flags": MODEL_FLAGS,
    }


def _component_purpose(component: str) -> str:
    return {
        "cmb_raw": "High-/low-ell Planck TT spectra, noise, beams, and covariance",
        "cmb_foregrounds": "Foreground templates (CIB, dust, SZ, point sources, synchrotron)",
        "cmb_masks": "Planck TT/EE/apodized masks",
        "cmb_bandpasses": "LFI/HFI bandpass response curves",
        "cmb_lensing": "Lensing bandpowers, noise, covariance, and masks",
        "cmb_planck_config": "Planck CosmoMC configuration files",
    }.get(component, "Planck raw dataset component")

def _find_planck_release(raw_dir: Path) -> Path:
    candidates = list(raw_dir.glob("plc_*"))
    if candidates:
        return candidates[0]
    for candidate in raw_dir.rglob("plc_*"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Planck release directory not found under {raw_dir}")


def _planck_release_label(release_root: Path) -> str:
    name = release_root.name.lower()
    if name.startswith("plc_"):
        return "r" + name.split("plc_")[-1]
    return name


def _build_cmb_raw(release_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload: dict[str, object] = {}
    summary: dict[str, object] = {}

    cl_paths = list(release_root.rglob("cl_cmb_plik_v22.dat"))
    preferred = _select_preferred_file(cl_paths, ["ttteee", "tt"])
    if preferred:
        data = np.loadtxt(preferred)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        ell = data[:, 0].astype(int)
        tt = data[:, 1]
        noise = data[:, 2] if data.shape[1] > 2 else np.full_like(tt, np.nan)
        payload["ell"] = ell
        payload["tt"] = tt
        payload["noise"] = noise
        summary["ell_points"] = int(ell.size)
        summary["ell_range"] = f"{int(ell[0])}-{int(ell[-1])}"
    else:
        summary["notes"] = "cl_cmb file not found"

    cov_paths = list(release_root.rglob("c_matrix*.dat"))
    if cov_paths:
        cov = _read_covariance(cov_paths[0])
        payload["cov"] = cov
        summary["cov_shape"] = cov.shape
    else:
        summary.setdefault("notes", "covariance file missing")

    blmin = _load_text_floats(release_root, "blmin.dat")
    blmax = _load_text_floats(release_root, "blmax.dat")
    if blmin.size and blmax.size:
        payload["beam_blmin"] = blmin
        payload["beam_blmax"] = blmax
        summary["beam_bins"] = blmin.size

    bweight = _load_text_floats(release_root, "bweight.dat")
    if bweight.size:
        payload["beam_weights"] = bweight

    bf_candidates = list(release_root.rglob("bf*.dat"))
    if bf_candidates:
        bf_path = bf_candidates[0]
        payload["beam_filter"] = np.loadtxt(bf_path)

    summary.setdefault("notes", "TT spectra extracted; TE/EE pending")
    return payload, summary


def _load_text_floats(root: Path, filename: str) -> np.ndarray:
    for external in root.rglob("_external"):
        candidate = external / filename
        if candidate.exists():
            try:
                return np.loadtxt(candidate)
            except Exception:
                continue
    return np.array([], dtype=float)


def _select_preferred_file(path_candidates: list[Path], tags: Sequence[str]) -> Path | None:
    if not path_candidates:
        return None
    lowered = [(p, str(p).lower()) for p in path_candidates]
    for tag in tags:
        for p, lowered_path in lowered:
            if tag in lowered_path:
                return p
    return path_candidates[0]


def _read_covariance(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) <= 8:
        return np.array([])
    payload = raw[4:-4]
    data = np.frombuffer(payload, dtype=np.float64)
    size = int(np.sqrt(data.size))
    if size * size != data.size:
        raise ValueError(f"Covariance matrix size {data.size} is not square")
    return data.reshape((size, size))


def _build_foregrounds(release_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload: dict[str, object] = {}
    metadata_map: dict[str, dict[str, object]] = {}
    seen: set[str] = set()

    component_dirs = [p for p in release_root.rglob("component_*") if p.is_dir()]
    component_dirs.sort()

    for comp_dir in component_dirs:
        slug = _unique_slug(comp_dir, seen)
        template = _load_fits_array(comp_dir / "template")
        if template is not None:
            payload[f"{slug}_template"] = template
        dfreq = _load_fits_array(comp_dir / "dfreq")
        if dfreq is not None:
            payload[f"{slug}_dfreq"] = dfreq
        payload[f"{slug}_category"] = _infer_foreground_category(comp_dir)

        arrays = {
            "amplitude": _load_fits_array(comp_dir / "A_cmb"),
            "color": _load_fits_array(comp_dir / "color"),
            "defaults": _load_fits_array(comp_dir / "defaults"),
        }
        for key, value in arrays.items():
            if value is not None:
                payload[f"{slug}_{key}"] = value

        keys = _read_null_strings(comp_dir / "keys")
        values = _read_null_strings(comp_dir / "values")
        metadata_map[slug] = {
            "original": comp_dir.name,
            "keys": keys,
            "values": values,
            "rename_from": _read_null_strings(comp_dir / "rename_from"),
            "rename_to": _read_null_strings(comp_dir / "rename_to"),
        }

    payload["component_metadata"] = json.dumps(metadata_map)
    summary: dict[str, object] = {"components": len(metadata_map)}
    return payload, summary


def _unique_slug(path: Path, seen: set[str]) -> str:
    keys = _read_null_strings(path / "keys")
    candidate = keys[0] if keys else path.name
    slug = _slugify(candidate)
    idx = 1
    base = slug
    while slug in seen:
        slug = f"{base}_{idx}"
        idx += 1
    seen.add(slug)
    return slug


def _infer_foreground_category(path: Path) -> str:
    keys = [k.lower() for k in _read_null_strings(path / "keys")]
    if any("cib" in key for key in keys):
        return "CIB"
    if any("dust" in key for key in keys):
        return "dust"
    if any("sz" in key for key in keys):
        return "SZ"
    if any("ps" in key for key in keys):
        return "point_source"
    if any("synch" in key for key in keys):
        return "synchrotron"
    return "other"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    return slug.lower() or "component"


def _load_fits_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
    except OSError:
        return None
    if data is None:
        return None
    return np.asarray(data, dtype=float)


def _read_null_strings(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    parts = raw.split(b"\x00")
    return [part.decode("utf-8", errors="ignore") for part in parts if part.strip()]

def _build_masks(release_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload: dict[str, object] = {}
    masks = []
    for path in release_root.rglob("*mask*"):
        if not path.is_file() or path.name.lower().endswith(".md"):
            continue
        arr = _load_fits_array(path)
        if arr is None:
            continue
        key = _slugify(path.stem)
        payload[f"mask_{key}"] = arr
        masks.append(key)
    summary: dict[str, object] = {"count": len(masks)}
    if not masks:
        summary["notes"] = "No mask files located"
    return payload, summary


def _build_bandpasses(release_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload: dict[str, object] = {}
    found = []
    for path in release_root.rglob("*band*"):
        if not path.is_file():
            continue
        arr = _load_fits_array(path)
        if arr is None:
            try:
                arr = np.loadtxt(path)
            except Exception:
                continue
        key = _slugify(path.stem)
        payload[f"band_{key}"] = arr
        found.append(path.name)
    summary: dict[str, object] = {"count": len(found)}
    if not found:
        summary["notes"] = "Planck bandpass files not present in release"
    return payload, summary


def _build_lensing(release_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload: dict[str, object] = {}
    count = 0
    for lens_dir in release_root.rglob("*.clik_lensing"):
        if not lens_dir.is_dir():
            continue
        base = _slugify(lens_dir.name)
        subdir = lens_dir / "clik_lensing"
        for name in ("cl_fid", "siginv", "bins", "cor0", "cors", "pp_hat"):
            candidate = subdir / name
            arr = _load_fits_array(candidate)
            if arr is None and candidate.exists():
                try:
                    arr = np.loadtxt(candidate)
                except Exception:
                    arr = None
            if arr is not None:
                payload[f"{base}_{name}"] = arr
                count += 1
    summary = {"entries": count}
    if count == 0:
        summary["notes"] = "No lensing payload discovered"
    return payload, summary


def _collect_planck_configs(release_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload: dict[str, object] = {}
    configs = []
    for pattern in ("*.ini", "*.paramnames"):
        for path in release_root.rglob(pattern):
            if not path.is_file():
                continue
            key = _slugify(path.relative_to(release_root).as_posix())
            try:
                payload[f"config_{key}"] = path.read_text(encoding="utf-8")
                configs.append(str(path.relative_to(release_root)))
            except Exception:
                continue
    summary = {"count": len(configs)}
    if not configs:
        summary["notes"] = "No Planck configuration files found"
    return payload, summary
