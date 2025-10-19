#!/usr/bin/env bash
set -euo pipefail

# Optional positional argument to override the virtual environment path.
VENV_DIR="${1:-.venv}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} not found on PATH. Set the PYTHON environment variable." >&2
  exit 1
fi

VENV_ABS="$("${PYTHON_BIN}" -c 'import sys, pathlib; root, venv = sys.argv[1:3]; print(str((pathlib.Path(root) / venv).resolve()))' "${ROOT_DIR}" "${VENV_DIR}")"

echo "[INFO] Using Python interpreter: ${PYTHON_BIN}"
echo "[INFO] Creating virtual environment at: ${VENV_ABS}"

if [ -d "${VENV_ABS}" ]; then
  echo "[INFO] Virtual environment already exists — skipping creation."
else
  "${PYTHON_BIN}" -m venv "${VENV_ABS}"
fi

source "${VENV_ABS}/bin/activate"

echo "[INFO] Upgrading pip tooling"
pip install --upgrade pip setuptools wheel

REQ_FILE="${ROOT_DIR}/requirements.txt"
if [ -f "${REQ_FILE}" ]; then
  echo "[INFO] Installing requirements from ${REQ_FILE}"
  pip install -r "${REQ_FILE}"
else
  echo "[WARN] requirements.txt not found; installing core dependencies explicitly."
  pip install numpy scipy pandas matplotlib jinja2 pyyaml
fi

echo "[INFO] Environment ready. Activate it via:"
echo "source ${VENV_ABS}/bin/activate"
