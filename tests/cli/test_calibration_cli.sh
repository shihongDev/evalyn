#!/bin/bash
# =============================================================================
# Calibration CLI Tests
# =============================================================================
# Tests for calibration commands:
#   - calibrate (basic, gepa, gepa-native, ape, opro optimizers)
#   - list-calibrations
#   - cluster-failures
#   - cluster-misalignments
# =============================================================================

set +e  # Continue on errors

# Load configuration and helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/helpers.sh"

parse_args "$@"

# =============================================================================
# Setup
# =============================================================================

log_header "Calibration CLI Tests"
ensure_test_dir
ensure_test_traces

# Use the calibration fixture which has real eval runs with subjective metrics
FIXTURE_DIR="$TEST_FIXTURES/calibration"
DATASET_DIR="$FIXTURE_DIR"
CALIBRATION_DIR="$TEST_OUTPUT_DIR/calibrations"
ANNOTATIONS_FILE="$FIXTURE_DIR/annotations.jsonl"

rm -rf "$CALIBRATION_DIR"
mkdir -p "$CALIBRATION_DIR"

if [ ! -f "$DATASET_DIR/dataset.jsonl" ]; then
    echo "Calibration fixture missing: $DATASET_DIR/dataset.jsonl"
    echo "Skipping calibration tests."
    exit 0
fi

log_info "Using calibration fixture: $FIXTURE_DIR"

# =============================================================================
# Test: list-calibrations
# =============================================================================

log_header "list-calibrations Command"

run_cmd "list-calibrations --dataset" run_evalyn list-calibrations --dataset "$DATASET_DIR"
run_cmd "list-calibrations --format json" run_evalyn list-calibrations --dataset "$DATASET_DIR" --format json
run_cmd_allow_fail "list-calibrations --latest" run_evalyn list-calibrations --latest

# =============================================================================
# Test: calibrate (LLM optimizer)
# =============================================================================

log_header "calibrate: Basic Optimizer"

if require_api_key "calibrate --optimizer basic"; then
    run_cmd "calibrate --optimizer basic" run_evalyn calibrate \
        --metric-id "helpfulness" \
        --annotations "$ANNOTATIONS_FILE" \
        --dataset "$DATASET_DIR" \
        --optimizer basic \
        --output "$CALIBRATION_DIR/calibration_basic.json"

    run_cmd "calibrate --show-examples" run_evalyn calibrate \
        --metric-id "helpfulness" \
        --annotations "$ANNOTATIONS_FILE" \
        --dataset "$DATASET_DIR" \
        --optimizer basic \
        --show-examples
fi

# =============================================================================
# Test: calibrate (GEPA optimizer)
# =============================================================================

log_header "calibrate: GEPA Optimizer"

if require_api_key "calibrate --optimizer gepa"; then
    run_cmd "calibrate --optimizer gepa" run_evalyn calibrate \
        --metric-id "helpfulness" \
        --annotations "$ANNOTATIONS_FILE" \
        --dataset "$DATASET_DIR" \
        --optimizer gepa \
        --gepa-max-calls 5 \
        --output "$CALIBRATION_DIR/calibration_gepa.json"
fi

# =============================================================================
# Test: calibrate (GEPA-Native optimizer)
# =============================================================================

log_header "calibrate: GEPA-Native Optimizer"

if require_api_key "calibrate --optimizer gepa-native"; then
    run_cmd "calibrate --optimizer gepa-native" run_evalyn calibrate \
        --metric-id "helpfulness" \
        --annotations "$ANNOTATIONS_FILE" \
        --dataset "$DATASET_DIR" \
        --optimizer gepa-native \
        --gepa-max-calls 5 \
        --output "$CALIBRATION_DIR/calibration_gepa_native.json"
fi

# =============================================================================
# Test: calibrate (APE optimizer)
# =============================================================================

log_header "calibrate: APE Optimizer"

if require_api_key "calibrate --optimizer ape"; then
    run_cmd "calibrate --optimizer ape" run_evalyn calibrate \
        --metric-id "helpfulness" \
        --annotations "$ANNOTATIONS_FILE" \
        --dataset "$DATASET_DIR" \
        --optimizer ape \
        --output "$CALIBRATION_DIR/calibration_ape.json"
fi

# =============================================================================
# Test: calibrate (OPRO optimizer)
# =============================================================================

log_header "calibrate: OPRO Optimizer"

if require_api_key "calibrate --optimizer opro"; then
    run_cmd "calibrate --optimizer opro" run_evalyn calibrate \
        --metric-id "helpfulness" \
        --annotations "$ANNOTATIONS_FILE" \
        --dataset "$DATASET_DIR" \
        --optimizer opro \
        --output "$CALIBRATION_DIR/calibration_opro.json"
fi

# =============================================================================
# Test: cluster-failures
# =============================================================================

log_header "cluster-failures Command"

# First need to run an eval to have failures to cluster
if require_api_key "cluster-failures"; then
    run_cmd_allow_fail "cluster-failures" run_evalyn cluster-failures \
        --dataset "$DATASET_DIR" \
        --num-clusters 3 \
        --output "$CALIBRATION_DIR/failure_clusters.json"

    run_cmd_allow_fail "cluster-failures --run-id" run_evalyn cluster-failures \
        --run-id "$(get_run_id)" \
        --num-clusters 3
fi

# =============================================================================
# Test: cluster-misalignments
# =============================================================================

log_header "cluster-misalignments Command"

if require_api_key "cluster-misalignments"; then
    run_cmd_allow_fail "cluster-misalignments" run_evalyn cluster-misalignments \
        --dataset "$DATASET_DIR" \
        --annotations "$ANNOTATIONS_FILE" \
        --num-clusters 3 \
        --output "$CALIBRATION_DIR/misalignment_clusters.json"
fi

# =============================================================================
# Summary
# =============================================================================

print_summary "Calibration CLI"
exit $?
