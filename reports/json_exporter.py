"""
JSON Exporter — Data Export for PBUF Framework
==============================================

Exports computed statistics to JSON format for:
  - Scientific reproducibility
  - Data sharing and archiving
  - Integration with other analysis tools
  - Publication supplementary materials

The JSON export includes all computed metrics, parameter values,
and metadata needed to fully reproduce the analysis.

Usage:
------
    from reports.json_exporter import export_json
    export_json(stats, "reports/output/results.json")
"""

from pathlib import Path
import json
from typing import Dict, Any


def export_json(stats: Dict[str, Any], output_file: str = "reports/output/results.json"):
    """
    Export statistics dictionary to JSON file.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_model_stats()
    output_file : str
        Path to output JSON file

    Returns
    -------
    str
        Path to the created JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(f"[OK] JSON data exported to {output_path.resolve()}")
    return str(output_path)
