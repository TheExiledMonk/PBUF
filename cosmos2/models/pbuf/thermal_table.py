"""Thermal table loader and interpolator for the PBUF model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cosmos2.kernels.pbuf_thermal import (
    FIELD_ALPHA,
    FIELD_DLN_ALPHA,
    FIELD_DLN_EPS,
    FIELD_EPS,
    FIELD_GSTAR,
    FIELD_GSTARS,
    FIELD_T,
    interp_field_njit,
)

FieldRows = List[Mapping[str, Any]]
InMemoryTable = Mapping[str, Any] | Sequence[Mapping[str, Any]]


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Encountered empty string while parsing thermal table value.")
        return float(text)

    raise ValueError(f"Cannot convert value of type {type(value)} to float.")


def _clean_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for raw_key, raw_value in row.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip()
        if not key:
            continue
        clean[key] = raw_value
    return clean


def _load_rows_from_json(path: Path) -> Tuple[FieldRows, Dict[str, Any]]:
    payload = json.loads(path.read_text())
    metadata: Dict[str, Any] = {}
    rows: Optional[FieldRows] = None

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        # Common container keys produced by the Quantum export helpers.
        for key in ("rows", "data", "table"):
            maybe_rows = payload.get(key)
            if isinstance(maybe_rows, list):
                rows = maybe_rows
                break
        if rows is None:
            # Fallback: treat top-level mapping as {column: values}.
            sequences = {key: value for key, value in payload.items() if isinstance(value, list)}
            if sequences:
                rows = [dict(zip(sequences.keys(), values)) for values in zip(*sequences.values())]
        metadata = {key: value for key, value in payload.items() if key not in {"rows", "data", "table"}}

    if rows is None:
        raise ValueError(f"Could not locate row data in thermal table JSON '{path}'.")

    cleaned = []
    for entry in rows:
        if not isinstance(entry, Mapping):
            raise ValueError("Thermal table JSON rows must be mapping objects.")
        cleaned.append(_clean_row(entry))
    return cleaned, metadata


def _load_rows_from_csv(path: Path) -> Tuple[FieldRows, Dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [_clean_row(row) for row in reader]
    return rows, {}


def _load_rows(path: Path) -> Tuple[FieldRows, Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_rows_from_json(path)
    if suffix == ".csv":
        return _load_rows_from_csv(path)
    raise ValueError(f"Unsupported thermal table format '{suffix}'. Expected .json or .csv.")


def _rows_from_payload(payload: InMemoryTable) -> Tuple[FieldRows, Dict[str, Any]]:
    rows: Optional[FieldRows] = None
    metadata: Dict[str, Any] = {}

    if isinstance(payload, Mapping):
        potential_rows = payload.get("rows") or payload.get("data") or payload.get("table")
        if isinstance(potential_rows, list):
            rows = potential_rows
        elif isinstance(potential_rows, Sequence):
            rows = list(potential_rows)
        else:
            sequences = {key: value for key, value in payload.items() if isinstance(value, Sequence)}
            if sequences:
                rows = [dict(zip(sequences.keys(), values)) for values in zip(*sequences.values())]
        metadata = {key: value for key, value in payload.items() if key not in {"rows", "data", "table"}}
    elif isinstance(payload, Sequence):
        rows = list(payload)

    if rows is None:
        raise ValueError("Could not interpret in-memory thermal table payload.")

    cleaned = []
    for entry in rows:
        if not isinstance(entry, Mapping):
            raise ValueError("In-memory thermal table rows must be mapping objects.")
        cleaned.append(_clean_row(entry))
    return cleaned, metadata


def _find_column_key(rows: FieldRows, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if all(name in row for row in rows):
            return name
    return None


def _column(rows: FieldRows, key: str) -> np.ndarray:
    values: List[float] = []
    for idx, row in enumerate(rows):
        if key not in row:
            raise ValueError(f"Row {idx} is missing '{key}' in thermal table.")
        try:
            values.append(_as_float(row[key]))
        except ValueError as err:
            raise ValueError(f"Could not parse '{key}' value in row {idx}: {err}") from err
    return np.asarray(values, dtype=float)


class ThermalTable:
    """Container that exposes interpolated thermal quantities."""

    _T_ALIASES = ("T", "T_K", "temperature", "temperature_K")
    _A_ALIASES = ("a", "scale_factor")
    _Z_ALIASES = ("z", "redshift", "z_from_T")
    _EPS_ALIASES = ("epsilon0_T", "epsilon_T", "eps0_T", "eps_T")
    _ALPHA_ALIASES = ("alpha_T", "alpha")
    _DLN_EPS_ALIASES = ("dln_epsilon0_dlnT", "dln_eps_dlnT")
    _DLN_ALPHA_ALIASES = ("dln_alpha_dlnT", "dln_alphaT_dlnT")
    _GSTAR_ALIASES = ("g_star", "gStar", "gstar")
    _GSTAR_S_ALIASES = ("g_starS", "gStarS", "gstarS")

    def __init__(self, source: str | Path | InMemoryTable) -> None:
        rows: FieldRows
        metadata: Dict[str, Any]
        path_obj: Optional[Path] = None

        if isinstance(source, (str, Path)):
            path_obj = Path(source)
            if not path_obj.exists():
                raise FileNotFoundError(f"Thermal table '{path_obj}' does not exist.")
            rows, metadata = _load_rows(path_obj)
        else:
            rows, metadata = _rows_from_payload(source)
            metadata.setdefault("source", "in-memory")
            metadata.setdefault("row_count", len(rows))

        if not rows:
            location = f" '{path_obj}'" if path_obj else ""
            raise ValueError(f"Thermal table{location} is empty.")

        a_key = _find_column_key(rows, self._A_ALIASES)
        t_key = _find_column_key(rows, self._T_ALIASES)
        z_key = _find_column_key(rows, self._Z_ALIASES)
        eps_key = _find_column_key(rows, self._EPS_ALIASES)
        alpha_key = _find_column_key(rows, self._ALPHA_ALIASES)
        dln_eps_key = _find_column_key(rows, self._DLN_EPS_ALIASES)
        dln_alpha_key = _find_column_key(rows, self._DLN_ALPHA_ALIASES)
        g_star_key = _find_column_key(rows, self._GSTAR_ALIASES)
        g_starS_key = _find_column_key(rows, self._GSTAR_S_ALIASES)

        required_fields = (
            ("a", a_key),
            ("z", z_key),
            ("T", t_key),
            ("epsilon0_T", eps_key),
            ("alpha_T", alpha_key),
            ("dln_epsilon0_dlnT", dln_eps_key),
            ("dln_alpha_dlnT", dln_alpha_key),
            ("g_star", g_star_key),
            ("g_starS", g_starS_key),
        )
        missing = [label for label, key in required_fields if key is None]
        if missing:
            location = f" '{path_obj}'" if path_obj else ""
            raise ValueError(f"Thermal table{location} is missing required columns: {missing}")

        a_vals = _column(rows, a_key)
        z_vals = _column(rows, z_key)
        t_vals = _column(rows, t_key)
        eps_vals = _column(rows, eps_key)
        alpha_vals = _column(rows, alpha_key)
        dln_eps_vals = _column(rows, dln_eps_key)
        dln_alpha_vals = _column(rows, dln_alpha_key)
        g_star_vals = _column(rows, g_star_key)
        g_starS_vals = _column(rows, g_starS_key)

        order = np.argsort(a_vals)
        a_sorted = a_vals[order]
        z_sorted = z_vals[order]
        t_sorted = t_vals[order]
        eps_sorted = eps_vals[order]
        alpha_sorted = alpha_vals[order]
        dln_eps_sorted = dln_eps_vals[order]
        dln_alpha_sorted = dln_alpha_vals[order]
        g_star_sorted = g_star_vals[order]
        g_starS_sorted = g_starS_vals[order]

        if np.any(a_sorted <= 0.0):
            raise ValueError("Thermal table scale factors must be positive.")
        if not np.all(np.diff(a_sorted) > 0.0):
            raise ValueError("Scale factor column must be strictly increasing.")
        if not np.all(np.diff(t_sorted) < 0.0):
            raise ValueError("Temperature column must be strictly decreasing with increasing scale factor.")

        self.path = path_obj
        self.metadata: Dict[str, Any] = dict(metadata)
        self.a: np.ndarray = a_sorted
        self.z: np.ndarray = z_sorted
        self.T: np.ndarray = t_sorted
        self.eps: np.ndarray = eps_sorted
        self.alpha: np.ndarray = alpha_sorted
        self.dln_eps: np.ndarray = dln_eps_sorted
        self.dln_alpha: np.ndarray = dln_alpha_sorted
        self.g_star: np.ndarray = g_star_sorted
        self.g_starS: np.ndarray = g_starS_sorted

        # Register fields with both canonical names and the original column names.
        self._fields: Dict[str, np.ndarray] = {}
        self._register_field("a", self.a)
        self._register_field(a_key, self.a)
        self._register_field("z", self.z)
        self._register_field(z_key, self.z)
        self._register_field("T", self.T)
        self._register_field(t_key, self.T)
        self._register_field("epsilon0_T", self.eps)
        self._register_field(eps_key, self.eps)
        self._register_field("alpha_T", self.alpha)
        self._register_field(alpha_key, self.alpha)
        self._register_field("dln_epsilon0_dlnT", self.dln_eps)
        self._register_field(dln_eps_key, self.dln_eps)
        self._register_field("dln_alpha_dlnT", self.dln_alpha)
        self._register_field(dln_alpha_key, self.dln_alpha)
        self._register_field("g_star", self.g_star)
        self._register_field(g_star_key, self.g_star)
        self._register_field("g_starS", self.g_starS)
        self._register_field(g_starS_key, self.g_starS)

        extra_keys = set()
        for row in rows:
            extra_keys.update(row.keys())
        extra_keys.difference_update(
            {
                a_key,
                z_key,
                t_key,
                eps_key,
                alpha_key,
                dln_eps_key,
                dln_alpha_key,
                g_star_key,
                g_starS_key,
            }
        )

        for key in extra_keys:
            try:
                data = _column(rows, key)[order]
            except (ValueError, KeyError):
                continue
            self._register_field(key, data)

        self._a_min = float(self.a[0])
        self._a_max = float(self.a[-1])
        self._log_a = np.log(self.a)
        self._log_fields = {"T", t_key}
        self._njit_payload = (
            self.a,
            self._log_a,
            self.T,
            self.eps,
            self.alpha,
            self.dln_eps,
            self.dln_alpha,
            self.g_star,
            self.g_starS,
        )

    def _register_field(self, name: str, values: np.ndarray) -> None:
        self._fields[name] = values

    def numba_payload(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return arrays needed for numba interpolation in a fixed order.

        Order: (a, log_a, T, eps, alpha, dln_eps, dln_alpha, g_star, g_starS)
        """

        return self._njit_payload

    @staticmethod
    def numba_field_id(name: str) -> int:
        """
        Map a canonical field name to its numba field id.
        """

        normalized = name.strip()
        if normalized in {"T", "temperature", "temperature_K", "T_K"}:
            return 0
        if normalized in {"epsilon0_T", "epsilon_T", "eps0_T", "eps_T"}:
            return 1
        if normalized in {"alpha_T", "alpha"}:
            return 2
        if normalized in {"dln_epsilon0_dlnT", "dln_eps_dlnT"}:
            return 3
        if normalized in {"dln_alpha_dlnT", "dln_alphaT_dlnT"}:
            return 4
        if normalized in {"g_star", "gStar", "gstar"}:
            return 5
        if normalized in {"g_starS", "gStarS", "gstarS"}:
            return 6
        return -1

    def fast_get(self, field: str, *, at_scale_factor: float) -> float:
        """Numba-backed interpolation for known fields; falls back to get on unknown."""

        field_id = self.numba_field_id(field)
        if field_id < 0:
            return self.get(field, at_scale_factor=at_scale_factor)
        a_arr, log_a, T, eps, alpha, dln_eps, dln_alpha, g_star, g_starS = self._njit_payload
        return interp_field_njit(
            float(at_scale_factor),
            a_arr,
            log_a,
            field_id,
            T,
            eps,
            alpha,
            dln_eps,
            dln_alpha,
            g_star,
            g_starS,
        )

    def clamp_scale_factor(self, a: float) -> float:
        """Clamp a scale factor to the bounds covered by the table."""

        return float(np.clip(a, self._a_min, self._a_max))

    def available_fields(self) -> List[str]:
        """Return the names of all numeric fields provided by the table."""

        return sorted(self._fields.keys())

    def interp(self, a: float, field: str) -> float:
        """
        Interpolate the requested field at the supplied scale factor.

        Temperatures use log-linear interpolation in both a and T, while all
        other fields rely on linear interpolation in a.
        """

        if field not in self._fields:
            raise KeyError(f"Field '{field}' not found in thermal table. Available fields: {self.available_fields()}")

        a_val = self.clamp_scale_factor(float(a))
        values = self._fields[field]

        if field in self._log_fields:
            log_values = np.log(np.clip(values, 1e-50, None))
            result = np.exp(np.interp(np.log(a_val), self._log_a, log_values))
        else:
            result = np.interp(a_val, self.a, values)

        return float(result)

    def get(self, field: str, *, at_scale_factor: float) -> float:
        """Public wrapper that interpolates any field at the supplied a."""

        return self.interp(at_scale_factor, field)

    def get_by_z(self, field: str, *, at_redshift: float) -> float:
        """Interpolate a field at the supplied redshift."""

        z_val = float(at_redshift)
        if z_val <= -1.0:
            raise ValueError("Redshift must be greater than -1.")
        a_val = 1.0 / (1.0 + z_val)
        return self.get(field, at_scale_factor=a_val)

    def metadata_summary(self) -> Dict[str, Any]:
        """Shallow copy of table metadata for reporting."""

        return dict(self.metadata)

    @classmethod
    def load(cls, source: str | Path | InMemoryTable) -> "ThermalTable":
        """Convenience helper to instantiate a table from a file path or payload."""

        return cls(source)


__all__ = ["ThermalTable"]
