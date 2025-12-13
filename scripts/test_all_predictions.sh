#!/usr/bin/env bash
set -euo pipefail

# Run all prediction modules with fixed parameter overrides for PBUF and LCDM.
#
#   * PBUF: H0=74.675105, Rmax=90000000.000000
#   * LCDM: H0=67.500000, Omega_m0=0.337500, Omega_b0=0.042500, Omega_k0=0.000000
#
# Predictions serialize their JSON payload into tmp/prediction_tests/<module>_<model>.json
# so this script can be rerun without stomping existing canonical outputs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="$REPO_ROOT/tmp/prediction_tests"
mkdir -p "$OUTPUT_ROOT"

mapfile -t MODULES < <(
  python3 - "$REPO_ROOT" <<'PY'
import json
from cosmos2.predictions import predictions_available

modules = sorted(predictions_available())
print("\n".join(modules))
PY
)

PBUF_PARAMS=(
  --param H0=74.675105
  --param Rmax=90000000.000000
)
LCDM_PARAMS=(
  --param H0=67.500000
  --param Omega_m0=0.337500
  --param Omega_b0=0.042500
  --param Omega_k0=0.000000
)

run_prediction() {
  local module="$1"
  local model="$2"
  shift 2
  local extra_params=("$@")
  local output="$OUTPUT_ROOT/${module}_${model}.json"

  printf "=== Running %s (%s) → %s\n" "$module" "$model" "$output"
  if ! python3 "$REPO_ROOT/cosmos_cli.py" predict "$module" --model "$model" \
    "${extra_params[@]}" \
    --save-json "$output"; then
    printf "!!! Prediction %s (%s) failed (see above)\n" "$module" "$model"
  fi
}

for module in "${MODULES[@]}"; do
  run_prediction "$module" pbuf "${PBUF_PARAMS[@]}"
done

for module in "${MODULES[@]}"; do
  run_prediction "$module" lcdm "${LCDM_PARAMS[@]}"
done
