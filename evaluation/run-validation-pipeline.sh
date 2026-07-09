#!/usr/bin/env bash
# =============================================================================
# End-to-end wrapper for the CI validation simulation pipeline.
#
# Runs steps 01-04 sequentially against a given YAML configuration.
# Step 01 is invoked once per variation defined in the YAML
# (variation count is auto-detected from the YAML).
#
# Location: <repo>/evaluation/
# Usage:
#   ./run-validation-pipeline.sh --config <yaml-path> --output <workspace-dir>
#
# Examples:
#   ./run-validation-pipeline.sh --config config/validation-evaluation-test-run.yaml --output /tmp/uml-testrun
# =============================================================================

set -euo pipefail

# ---- arg parsing -------------------------------------------------------------
CONFIG=""
OUTPUT=""

usage() {
  sed -n '2,/^# =====/p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)  CONFIG="$2";  shift 2 ;;
    -o|--output)  OUTPUT="$2";  shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" || -z "$OUTPUT" ]]; then
  echo "Error: --config and --output are both required." >&2
  usage
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 2
fi

# ---- resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
STEP_DIR="$SCRIPT_DIR/R_scripts/ci-validation-evaluation"

CONFIG_BASENAME="$(basename "$CONFIG")"
CONFIG_STEM="${CONFIG_BASENAME%.yaml}"
WORKDIR="$OUTPUT/$CONFIG_STEM"

mkdir -p "$OUTPUT"

# ---- detect number of variations from the YAML -------------------------------
N_VARIATIONS=$(Rscript -e "cat(length(yaml::read_yaml('$CONFIG')\$variations))")
if [[ -z "$N_VARIATIONS" || "$N_VARIATIONS" -lt 1 ]]; then
  echo "Error: could not determine variation count from $CONFIG" >&2
  exit 3
fi

echo "=============================================================="
echo "CI validation simulation pipeline"
echo "  config:     $CONFIG"
echo "  output:     $OUTPUT"
echo "  workdir:    $WORKDIR"
echo "  variations: $N_VARIATIONS"
echo "=============================================================="

# ---- Step 01 -----------------------------------------------------------------
for i in $(seq 1 "$N_VARIATIONS"); do
  echo
  echo "--- Step 01 [$i/$N_VARIATIONS] ---"
  Rscript "$STEP_DIR/01-draw-subsamples-and-bootstrap.R" \
          --output_path         "$OUTPUT" \
          --config_file         "$CONFIG" \
          --configuration_index "$i"
done

# ---- Step 02 -----------------------------------------------------------------
echo
echo "--- Step 02 ---"
Rscript "$STEP_DIR/02-extract-bootstrap-percentiles.R" \
        --data_parent_dir "$WORKDIR" \
        --config_file     "$CONFIG_BASENAME"

# ---- Step 03 -----------------------------------------------------------------
echo
echo "--- Step 03 ---"
Rscript "$STEP_DIR/03-merge-with-true-values.R" \
        --parent_dir "$WORKDIR"

# ---- Step 04 -----------------------------------------------------------------
echo
echo "--- Step 04 ---"
Rscript "$STEP_DIR/04-compute-and-aggregate-cis.R" \
        --datafile_path "$WORKDIR"

echo
echo "=============================================================="
echo "Pipeline finished."
echo "Primary artefact:"
echo "  $WORKDIR/final_results/overview_signTrafo.Rdata"
echo "=============================================================="

