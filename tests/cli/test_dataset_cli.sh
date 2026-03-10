#!/bin/bash
# =============================================================================
# Dataset CLI Tests
# =============================================================================
# Tests for dataset building commands:
#   - build-dataset (with various filters)
#   - validate
#   - status
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

log_header "Dataset CLI Tests"
ensure_test_dir
ensure_test_traces

DATASET_DIR="$TEST_OUTPUT_DIR/dataset-test"
rm -rf "$DATASET_DIR"

# =============================================================================
# Test: build-dataset
# =============================================================================

log_header "build-dataset Command"

# Basic dataset build
run_cmd "build-dataset (basic)" run_evalyn build-dataset \
    --project "$TEST_PROJECT" \
    --output "$DATASET_DIR/basic" \
    --limit 10

# With version filter
run_cmd "build-dataset --version" run_evalyn build-dataset \
    --project "$TEST_PROJECT" \
    --version "$TEST_VERSION" \
    --output "$DATASET_DIR/versioned" \
    --limit 5

# Production only
run_cmd "build-dataset --production" run_evalyn build-dataset \
    --project "$TEST_PROJECT" \
    --production \
    --output "$DATASET_DIR/production" \
    --limit 5

# Simulation only
run_cmd "build-dataset --simulation" run_evalyn build-dataset \
    --project "$TEST_PROJECT" \
    --simulation \
    --output "$DATASET_DIR/simulation" \
    --limit 5

# Include errors
run_cmd "build-dataset --include-errors" run_evalyn build-dataset \
    --project "$TEST_PROJECT" \
    --include-errors \
    --output "$DATASET_DIR/with-errors" \
    --limit 5

# =============================================================================
# Test: validate
# =============================================================================

log_header "validate Command"

# Validate the basic dataset
if [ -f "$DATASET_DIR/basic/dataset.jsonl" ]; then
    run_cmd "validate --dataset" run_evalyn validate --dataset "$DATASET_DIR/basic"
    run_cmd "validate --latest" run_evalyn validate --latest
fi

# =============================================================================
# Test: status
# =============================================================================

log_header "status Command"

if [ -d "$DATASET_DIR/basic" ]; then
    run_cmd "status --dataset" run_evalyn status --dataset "$DATASET_DIR/basic"
    # --latest searches data/ dirs, may not find test-output datasets
    run_cmd_allow_fail "status --latest" run_evalyn status --latest
fi

# =============================================================================
# Verify Dataset Contents
# =============================================================================

log_header "Dataset Verification"

log_test "Verify dataset files exist"
((TESTS_RUN++))
if [ -f "$DATASET_DIR/basic/dataset.jsonl" ] && [ -f "$DATASET_DIR/basic/meta.json" ]; then
    ITEM_COUNT=$(wc -l < "$DATASET_DIR/basic/dataset.jsonl")
    log_pass "Dataset created with $ITEM_COUNT items"
else
    log_fail "Dataset files missing"
fi

# =============================================================================
# Summary
# =============================================================================

print_summary "Dataset CLI"
exit $?
