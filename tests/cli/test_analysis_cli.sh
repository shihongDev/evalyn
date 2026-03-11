#!/bin/bash
# =============================================================================
# Analysis CLI Tests
# =============================================================================
# Tests for analysis commands:
#   - analyze
#   - compare
#   - trend
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

log_header "Analysis CLI Tests"
ensure_test_dir
ensure_test_traces

DATASET_DIR="$TEST_OUTPUT_DIR/analysis-dataset"
rm -rf "$DATASET_DIR"

# Create test dataset and run eval to have data for analysis
log_info "Creating test dataset..."
create_test_dataset "$DATASET_DIR" 5

# Create simple metrics and run eval
mkdir -p "$DATASET_DIR/metrics"
cat > "$DATASET_DIR/metrics/test-metrics.json" << 'EOF'
[
    {"id": "output_nonempty", "type": "objective", "description": "Check output is not empty"},
    {"id": "latency_ms", "type": "objective", "description": "Measure latency"}
]
EOF

log_info "Running evaluation for analysis tests..."
run_evalyn run-eval --dataset "$DATASET_DIR" --metrics "$DATASET_DIR/metrics/test-metrics.json" >/dev/null 2>&1

# =============================================================================
# Test: analyze
# =============================================================================

log_header "analyze Command"

run_cmd "analyze --dataset" run_evalyn analyze --dataset "$DATASET_DIR"
run_cmd_allow_fail "analyze --latest" run_evalyn analyze --latest

# Analyze specific run
RUN_ID=$(get_run_id)
if [ -n "$RUN_ID" ]; then
    run_cmd "analyze --run (specific)" run_evalyn analyze --run "$RUN_ID"
fi

# =============================================================================
# Test: trend
# =============================================================================

log_header "trend Command"

run_cmd "trend --project" run_evalyn trend --project "$TEST_PROJECT" --limit 10
run_cmd "trend --format json" run_evalyn trend --project "$TEST_PROJECT" --limit 5 --format json
run_cmd "trend --limit 3" run_evalyn trend --project "$TEST_PROJECT" --limit 3

# =============================================================================
# Test: compare
# =============================================================================

log_header "compare Command"

# Run another eval to have two runs to compare
log_info "Running second evaluation for comparison..."
run_evalyn run-eval --dataset "$DATASET_DIR" --metrics "$DATASET_DIR/metrics/test-metrics.json" --dataset-name "compare-run-2" >/dev/null 2>&1

# Get two run IDs and compare if available
RUN_IDS=$(get_two_run_ids)
RUN_ID_1=$(echo "$RUN_IDS" | head -1)
RUN_ID_2=$(echo "$RUN_IDS" | tail -1)

if [ -n "$RUN_ID_1" ] && [ -n "$RUN_ID_2" ] && [ "$RUN_ID_1" != "$RUN_ID_2" ]; then
    run_cmd "compare (two runs)" run_evalyn compare --run1 "$RUN_ID_1" --run2 "$RUN_ID_2"
fi

# =============================================================================
# Summary
# =============================================================================

print_summary "Analysis CLI"
exit $?
