#!/bin/bash
# =============================================================================
# Production Data Evaluation Tests
# =============================================================================
# Tests run-eval with real production data from calibration fixture.
#
# Dataset: tests/fixtures/calibration/
#   - 100 research agent calls
#   - 100 human annotations
#   - Pre-defined metrics (LLM judge)
#   - Calibrated prompts
#
# Usage:
#   ./tests/cli/test_eval_prod.sh           # Full tests (requires API key)
#   ./tests/cli/test_eval_prod.sh --quick   # Objective-only tests (no API)
#   ./tests/cli/test_eval_prod.sh --verbose # Verbose output
# =============================================================================

set +e  # Continue on errors

# Load configuration and helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/helpers.sh"

# =============================================================================
# Parse Arguments
# =============================================================================

QUICK_MODE=false
for arg in "$@"; do
    case $arg in
        --quick|-q) QUICK_MODE=true ;;
        --verbose|-v) VERBOSE=true ;;
        --help|-h)
            head -18 "$0" | tail -15
            exit 0
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

log_header "Production Data Evaluation Tests"

FIXTURE_DIR="$TEST_FIXTURES/calibration"
PROD_OUTPUT_DIR="$TEST_OUTPUT_DIR/eval-prod"
OBJECTIVE_METRICS="$PROD_OUTPUT_DIR/objective-metrics.json"

rm -rf "$PROD_OUTPUT_DIR"
mkdir -p "$PROD_OUTPUT_DIR"

# Verify fixture exists
if [ ! -f "$FIXTURE_DIR/dataset.jsonl" ]; then
    log_fail "Calibration fixture not found: $FIXTURE_DIR/dataset.jsonl"
    exit 1
fi
log_info "Fixture: $FIXTURE_DIR"
log_info "Output: $PROD_OUTPUT_DIR"
log_info "Quick mode: $QUICK_MODE"

# Create objective-only metrics for quick tests
create_objective_metrics "$OBJECTIVE_METRICS" --with-latency

# =============================================================================
# Test: run-eval with objective metrics (no API required)
# =============================================================================

log_header "run-eval: Objective Metrics"

run_cmd "run-eval objective metrics" run_evalyn run-eval \
    --dataset "$FIXTURE_DIR" \
    --metrics "$OBJECTIVE_METRICS" \
    --dataset-name "test-objective"

# Verify output created
EVAL_RUN_DIR="$FIXTURE_DIR/eval_runs"
if [ -d "$EVAL_RUN_DIR" ]; then
    latest_run=$(ls -t "$EVAL_RUN_DIR" 2>/dev/null | head -1)
    if [ -n "$latest_run" ] && [ -f "$EVAL_RUN_DIR/$latest_run/results.json" ]; then
        log_info "Eval run created: $latest_run"
    fi
fi

# =============================================================================
# Test: run-eval output formats
# =============================================================================

log_header "run-eval: Output Formats"

run_cmd_check "run-eval --format table" "output_nonempty|latency" \
    run-eval --dataset "$FIXTURE_DIR" --metrics "$OBJECTIVE_METRICS" --format table

run_cmd "run-eval --format json" run_evalyn run-eval \
    --dataset "$FIXTURE_DIR" --metrics "$OBJECTIVE_METRICS" --format json

# =============================================================================
# Test: run-eval with workers (parallel execution)
# =============================================================================

log_header "run-eval: Parallel Execution"

run_cmd "run-eval -w 2 (2 workers)" run_evalyn run-eval \
    --dataset "$FIXTURE_DIR" \
    --metrics "$OBJECTIVE_METRICS" \
    -w 2 \
    --dataset-name "test-workers-2"

run_cmd "run-eval -w 4 (4 workers)" run_evalyn run-eval \
    --dataset "$FIXTURE_DIR" \
    --metrics "$OBJECTIVE_METRICS" \
    -w 4 \
    --dataset-name "test-workers-4"

# =============================================================================
# Test: run-eval verbose output (cost breakdown)
# =============================================================================

log_header "run-eval: Verbose Output"

run_cmd_check "run-eval --verbose cost breakdown" "output_nonempty|Avg Score|Pass Rate" \
    run-eval --dataset "$FIXTURE_DIR" --metrics "$OBJECTIVE_METRICS" --verbose

# =============================================================================
# Test: LLM Judge metrics (requires API key)
# =============================================================================

if [ "$QUICK_MODE" = false ]; then
    log_header "run-eval: LLM Judge Metrics"

    if require_api_key "run-eval LLM judge"; then
        run_cmd "run-eval fixture metrics (LLM judge)" run_evalyn run-eval \
            --dataset "$FIXTURE_DIR" \
            --metrics "$FIXTURE_DIR/metrics.json" \
            --dataset-name "test-llm-judge"

        # Test with calibrated prompts
        run_cmd_allow_fail "run-eval --use-calibrated" run_evalyn run-eval \
            --dataset "$FIXTURE_DIR" \
            --use-calibrated \
            --dataset-name "test-calibrated"
    fi
else
    log_skip "LLM judge tests (quick mode)"
fi

# =============================================================================
# Test: list-runs and show-run
# =============================================================================

log_header "list-runs and show-run"

run_cmd "list-runs" run_evalyn list-runs --limit 5
run_cmd "list-runs --format json" run_evalyn list-runs --format json --limit 3

# Get latest run and show it
RUN_ID=$(get_run_id)
if [ -n "$RUN_ID" ]; then
    run_cmd "show-run --id" run_evalyn show-run --id "$RUN_ID"
fi
run_cmd_allow_fail "show-run --last" run_evalyn show-run --last

# =============================================================================
# Test: Error handling
# =============================================================================

log_header "run-eval: Error Handling"

# Missing dataset
run_cmd_allow_fail "run-eval missing dataset" run_evalyn run-eval \
    --dataset "/nonexistent/path" \
    --metrics "$OBJECTIVE_METRICS"

# Invalid metrics file
run_cmd_allow_fail "run-eval invalid metrics" run_evalyn run-eval \
    --dataset "$FIXTURE_DIR" \
    --metrics "/nonexistent/metrics.json"

# =============================================================================
# Summary
# =============================================================================

print_summary "Production Data Evaluation"
exit $?
