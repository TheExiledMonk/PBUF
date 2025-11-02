#!/usr/bin/env bash

# One-command bootstrapper for the PBUF science run.
# Creates a virtual environment (if missing), installs dependencies,
# activates the environment, and launches the science runner.

set -euo pipefail

# Resolve project root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv_science"

echo "🔧 Preparing Python virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

echo "✅ Activating virtual environment"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "📦 Installing/upgrading dependencies"
pip install --upgrade pip
pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "🚀 Launching science run via CLI"
python "${PROJECT_ROOT}/cli.py" run science --config "${PROJECT_ROOT}/configs/science_run.json" "$@"
science_status=$?
if [[ $science_status -ne 0 ]]; then
  echo "❌ Science run failed (exit code $science_status). Aborting."
  exit $science_status
fi

echo "🧾 Generating reports via cli.py"
python "${PROJECT_ROOT}/cli.py" report generate --output "${PROJECT_ROOT}/reports/output"
report_status=$?
if [[ $report_status -ne 0 ]]; then
  echo "⚠️  Report generation failed (exit code $report_status). See logs above."
  exit $report_status
fi

echo "✅ Workflow complete. Reports stored in ${PROJECT_ROOT}/reports/output"
