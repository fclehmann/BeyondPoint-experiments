#!/usr/bin/env bash
#
# run-classification-pipeline.sh
#
# End-to-end runner for the classification use cases CI evaluation pipeline
# (R_scripts/ci-classification-evaluation/{01,02,03,04}-*.R).
#
# Runs all four steps in order:
#   01 draw subsamples and bootstrap        (once per experiment in the YAML)
#   02 extract bootstrap percentiles
#   03 merge with empirical truth
#   04 compute and aggregate CIs
#
# Location: <repo>/evaluation/
# Usage:
#   ./run-classification-pipeline.sh --config <yaml> --output <dir> [--data <dir>]


set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-classification-pipeline.sh --config <yaml> --output <dir> --data <dir>

Arguments:
  --config    Path to the YAML config
              (e.g. config/simpsons-evaluation-test-run.yaml)
  --output    Directory where results should be written; a sub-directory
              named after the config (without extension) is created inside
  --data      Directory containing the .Rdata file referenced by the
              'data_file:' entry in the YAML
  -h, --help  Show this help and exit
EOF
}

# --- parse arguments ---
CONFIG=""
OUTPUT=""
DATA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --data)   DATA="$2";   shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -z "$CONFIG" || -z "$OUTPUT" ]] && { usage; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Config file not found: $CONFIG" >&2; exit 1; }

mkdir -p "$OUTPUT"

# --- resolve the pipeline directory relative to the wrapper's own location ---
# The wrapper lives at <repo>/evaluation/; the R scripts live at ./R_scripts/;
# the data directory defaults to ../data relative to the wrapper.
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIPELINE_DIR="$ROOT_DIR/R_scripts/ci-classification-evaluation"

# Set default for --data now that ROOT_DIR is known, then validate.
DATA="${DATA:-$ROOT_DIR/../data}"
[[ -d "$DATA" ]] || { echo "Data directory not found: $DATA" >&2; exit 1; }
DATA="$( cd "$DATA" && pwd )"   # canonicalise for cleaner logs
echo "Using data directory: $DATA"



S01="$PIPELINE_DIR/01-draw-subsamples-and-bootstrap.R"
S02="$PIPELINE_DIR/02-extract-bootstrap-percentiles.R"
S03="$PIPELINE_DIR/03-merge-with-empirical-truth.R"
S04="$PIPELINE_DIR/04-compute-and-aggregate-cis.R"

for f in "$S01" "$S02" "$S03" "$S04"; do
  [[ -f "$f" ]] || { echo "Missing script: $f" >&2; exit 1; }
done

# --- determine number of experiments from the YAML ---
# Mirrors how the R scripts read 'experiments:' from the config.
NUM=$(Rscript -e 'cat(length(yaml::read_yaml(commandArgs(TRUE)[1])$experiments))' "$CONFIG")
echo "Config defines $NUM experiments."

# --- derive workspace path (matches what step 01 constructs internally) ---
CONFIG_BASE=$(basename "$CONFIG")
PARENT_NAME="${CONFIG_BASE%.*}"
WORKSPACE="$OUTPUT/$PARENT_NAME"

# --- step 01: one Rscript call per experiment ---
for (( i=1; i<=NUM; i++ )); do
  echo "=== Step 01 (experiment $i/$NUM) ==="
  Rscript "$S01" \
    --output_path         "$OUTPUT" \
    --config_file         "$CONFIG" \
    --configuration_index "$i" \
    --datafile_path       "$DATA"
done

# --- step 02: extract bootstrap percentiles ---
# The R script reads the yaml from within WORKSPACE (provenance copy),
# so --config_file is just the basename here.
echo "=== Step 02 ==="
Rscript "$S02" \
  --data_parent_dir "$WORKSPACE" \
  --config_file     "$CONFIG_BASE"

# --- step 03: merge with empirical truth ---
echo "=== Step 03 ==="
Rscript "$S03" \
  --parent_dir "$WORKSPACE"

# --- step 04: compute and aggregate CIs ---
echo "=== Step 04 ==="
Rscript "$S04" \
  --datafile_path "$WORKSPACE"

echo "=== Pipeline finished successfully ==="
echo "Results in: $WORKSPACE/final_results/"

