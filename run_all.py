"""
Convenience orchestration script to run the mock SN fits and build the unified report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"


def _run_pipeline(args_list):
    subprocess.check_call(args_list)


def _latest_result(out_dir: Path, model: str) -> Path:
    matches = sorted(out_dir.glob(f"{model.upper()}/*/fit_results.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No fit_results.json found for model {model} under {out_dir}")
    return matches[-1]


def _append_manifest(manifest_path: Path, result_json: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entry = json.loads(result_json.read_text(encoding="utf-8"))
    provenance = entry.get("provenance", {})
    dataset = entry.get("dataset", {})
    record = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "run_id": entry.get("run_id"),
        "model": entry.get("model"),
        "dataset": {
            "name": dataset.get("name"),
            "release_tag": dataset.get("release_tag"),
            "covariance_components_used": dataset.get("covariance_components_used"),
        },
        "git_commit": provenance.get("commit"),
        "parameters": entry.get("parameters"),
        "result_json": str(result_json.resolve()),
    }
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = []
    data.append(record)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run default SN fits and unified report.")
    parser.add_argument("--out", default="proofs/results", help="Base output directory")
    parser.add_argument("--report", default="reports/output/unified_report.html", help="Unified report path")
    parser.add_argument("--skip-unified", action="store_true", help="Skip unified report generation")
    parser.add_argument("--no-sn", action="store_true", help="Skip running SN fits")
    parser.add_argument("--dataset", default="pantheon_plus", help="Dataset key defined in config/datasets.yml")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    manifest_path = out_dir / MANIFEST_NAME

    if not args.no_sn:
        _run_pipeline(["python", "pipelines/fit_sn.py", "--dataset", args.dataset, "--model", "lcdm", "--out", str(out_dir)])
        latest_lcdm = _latest_result(out_dir, "lcdm")
        _append_manifest(manifest_path, latest_lcdm)

        _run_pipeline(["python", "pipelines/fit_sn.py", "--dataset", args.dataset, "--model", "pbuf", "--out", str(out_dir)])
        latest_pbuf = _latest_result(out_dir, "pbuf")
        _append_manifest(manifest_path, latest_pbuf)

    if not args.skip_unified:
        _run_pipeline(
            [
                "python",
                "pipelines/generate_unified_report.py",
                "--inputs",
                f"{out_dir}/**/fit_results.json",
                "--out",
                args.report,
            ]
        )


if __name__ == "__main__":
    main()
