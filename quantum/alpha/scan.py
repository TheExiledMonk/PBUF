"""
Generate α_QM scan artifacts for the legacy CLI entry point.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from quantum.core.constants import (
    FIELD_CONTENT_DEGREES,
    REGULATOR_COEFFICIENTS,
)
from quantum.core.types import IslandSummary, ScanMetadata


@dataclass(frozen=True)
class ScanSample:
    regulator: str
    field_set: str
    mixing_strength: float
    coupling_fraction: float
    cutoff_ratio: float
    alpha_value: float
    in_band: bool

    def as_row(self) -> Tuple[str, str, float, float, float, float, int]:
        return (
            self.regulator,
            self.field_set,
            self.mixing_strength,
            self.coupling_fraction,
            self.cutoff_ratio,
            self.alpha_value,
            int(self.in_band),
        )


def _logspace(min_value: float, max_value: float, samples: int) -> List[float]:
    if min_value <= 0 or max_value <= 0 or min_value >= max_value:
        raise ValueError("mixing_range bounds must be positive and strictly increasing")
    if samples < 1:
        raise ValueError("mixing_samples must be >= 1")
    if samples == 1:
        return [min_value]
    start = math.log10(min_value)
    stop = math.log10(max_value)
    step = (stop - start) / (samples - 1)
    return [10 ** (start + idx * step) for idx in range(samples)]


def _resolve_mapping(requested: Sequence[str] | None, reference: Dict[str, float], label: str) -> Dict[str, float]:
    if requested is None:
        return dict(reference)
    missing = [name for name in requested if name not in reference]
    if missing:
        raise ValueError(f"Unknown {label}: {', '.join(sorted(missing))}")
    return {name: reference[name] for name in requested}


def _f_coup(mixing: float) -> float:
    g = mixing
    return (g * g) / (1.0 + g * g)


def _f_cut(loop_coeff: float, n_eff: float) -> float:
    if loop_coeff <= 0 or n_eff <= 0:
        raise ValueError("Loop coefficients and N_eff must be positive")
    return math.sqrt(1.0 / (loop_coeff * n_eff))


def _alpha_value(coup: float, cutoff: float, eps0: float) -> float:
    if eps0 <= 0:
        raise ValueError("eps0 must be > 0")
    return coup * (cutoff ** 4) / eps0


def _build_samples(
    regulators: Dict[str, float],
    field_sets: Dict[str, float],
    mixing_strengths: Iterable[float],
    alpha_band: Tuple[float, float],
    eps0: float,
) -> List[ScanSample]:
    samples: List[ScanSample] = []
    for regulator, coeff in regulators.items():
        cutoff = None
        for field_set, n_eff in field_sets.items():
            cutoff = _f_cut(coeff, n_eff)
            for mixing in mixing_strengths:
                coup = _f_coup(mixing)
                alpha_val = _alpha_value(coup, cutoff, eps0)
                samples.append(
                    ScanSample(
                        regulator=regulator,
                        field_set=field_set,
                        mixing_strength=mixing,
                        coupling_fraction=coup,
                        cutoff_ratio=cutoff,
                        alpha_value=alpha_val,
                        in_band=alpha_band[0] <= alpha_val <= alpha_band[1],
                    )
                )
    return samples


def _summaries_from_samples(samples: Sequence[ScanSample]) -> List[IslandSummary]:
    summaries: List[IslandSummary] = []
    grouped: Dict[Tuple[str, str], List[ScanSample]] = {}
    for sample in samples:
        if not sample.in_band:
            continue
        grouped.setdefault((sample.regulator, sample.field_set), []).append(sample)

    for (regulator, field_set), hits in sorted(grouped.items()):
        alpha_values = [sample.alpha_value for sample in hits]
        mixing_values = [sample.mixing_strength for sample in hits]
        summaries.append(
            IslandSummary(
                regulator=regulator,
                field_set=field_set,
                hits=len(hits),
                mixing_min=min(mixing_values),
                mixing_max=max(mixing_values),
                alpha_min=min(alpha_values),
                alpha_max=max(alpha_values),
                alpha_mean=sum(alpha_values) / len(alpha_values),
            )
        )
    return summaries


def run_scan(
    *,
    regulators: Sequence[str],
    field_sets: Sequence[str],
    mixing_samples: int,
    mixing_range: Tuple[float, float],
    alpha_band: Tuple[float, float],
    eps0: float,
) -> Tuple[List[ScanSample], List[IslandSummary], ScanMetadata]:
    resolved_regs = _resolve_mapping(regulators, REGULATOR_COEFFICIENTS, "regulators")
    resolved_fields = _resolve_mapping(field_sets, FIELD_CONTENT_DEGREES, "field sets")
    mixing_strengths = _logspace(mixing_range[0], mixing_range[1], mixing_samples)

    samples = _build_samples(resolved_regs, resolved_fields, mixing_strengths, alpha_band, eps0)
    summaries = _summaries_from_samples(samples)

    alpha_values = [sample.alpha_value for sample in samples]
    metadata = ScanMetadata(
        global_alpha_min=min(alpha_values),
        global_alpha_max=max(alpha_values),
        total_island_hits=sum(summary.hits for summary in summaries),
    )
    return samples, summaries, metadata


def export_scan_artifacts(
    *,
    samples: Sequence[ScanSample],
    summaries: Sequence[IslandSummary],
    metadata: ScanMetadata,
    output_dir: str | Path,
) -> Tuple[Path, Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_csv = out_dir / "alpha_qm_scan_results.csv"
    with scan_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "regulator",
                "field_set",
                "mixing_strength",
                "coupling_fraction",
                "cutoff_ratio",
                "alpha_value",
                "in_band",
            ]
        )
        for sample in samples:
            writer.writerow(sample.as_row())

    summary_csv = out_dir / "alpha_qm_island_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "regulator",
                "field_set",
                "hits",
                "mixing_min",
                "mixing_max",
                "alpha_min",
                "alpha_max",
                "alpha_mean",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.regulator,
                    summary.field_set,
                    summary.hits,
                    summary.mixing_min,
                    summary.mixing_max,
                    summary.alpha_min,
                    summary.alpha_max,
                    summary.alpha_mean,
                ]
            )

    json_path = out_dir / "alpha_qm_island.json"
    payload = {
        "metadata": asdict(metadata),
        "candidates": [asdict(summary) for summary in summaries],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return scan_csv, summary_csv, json_path


__all__ = ["run_scan", "export_scan_artifacts", "ScanSample"]
