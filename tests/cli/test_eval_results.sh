#!/bin/bash
# =============================================================================
# Evaluation Results Validation Tests
# =============================================================================
# Validates the structure and content of eval-run output.
# Checks that results.json and report.html are valid.
#
# Usage:
#   ./tests/cli/test_eval_results.sh           # Run validation tests
#   ./tests/cli/test_eval_results.sh --verbose # Verbose output
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

log_header "Evaluation Results Validation Tests"

FIXTURE_DIR="$TEST_FIXTURES/calibration"
RESULTS_OUTPUT_DIR="$TEST_OUTPUT_DIR/eval-results"
OBJECTIVE_METRICS="$RESULTS_OUTPUT_DIR/metrics.json"

rm -rf "$RESULTS_OUTPUT_DIR"
mkdir -p "$RESULTS_OUTPUT_DIR"

# Create objective metrics
create_objective_metrics "$OBJECTIVE_METRICS"

# =============================================================================
# Create test eval run
# =============================================================================

log_header "Setup: Create Eval Run"

log_info "Running evaluation to generate results..."
run_evalyn run-eval \
    --dataset "$FIXTURE_DIR" \
    --metrics "$OBJECTIVE_METRICS" \
    --dataset-name "validation-test" 2>&1 | tail -5

# Find the latest run
EVAL_RUNS_DIR="$FIXTURE_DIR/eval_runs"
LATEST_RUN=$(ls -t "$EVAL_RUNS_DIR" 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    log_fail "No eval run created"
    print_summary "Results Validation"
    exit 1
fi

RUN_DIR="$EVAL_RUNS_DIR/$LATEST_RUN"
log_info "Validating run: $LATEST_RUN"

# =============================================================================
# Test: Required files exist
# =============================================================================

log_header "Validate: Required Files"

# Check results.json exists
log_test "results.json exists"
((TESTS_RUN++))
if [ -f "$RUN_DIR/results.json" ]; then
    log_pass "results.json exists"
else
    log_fail "results.json missing"
fi

# Check report.html exists
log_test "report.html exists"
((TESTS_RUN++))
if [ -f "$RUN_DIR/report.html" ]; then
    log_pass "report.html exists"
else
    log_fail "report.html missing"
fi

# =============================================================================
# Test: results.json is valid JSON
# =============================================================================

log_header "Validate: JSON Structure"

log_test "results.json is valid JSON"
((TESTS_RUN++))
if python3 -c "import json; json.load(open('$RUN_DIR/results.json'))" 2>/dev/null; then
    log_pass "results.json is valid JSON"
else
    log_fail "results.json is invalid JSON"
fi

# =============================================================================
# Test: results.json has expected structure
# =============================================================================

log_header "Validate: Results Structure"

# Check required top-level fields
check_json_field() {
    local field="$1"
    local desc="$2"
    log_test "$desc"
    ((TESTS_RUN++))
    if python3 -c "import json; data=json.load(open('$RUN_DIR/results.json')); exit(0 if '$field' in data else 1)" 2>/dev/null; then
        log_pass "$desc"
    else
        log_fail "$desc"
    fi
}

check_json_field "id" "results.json has run id"
check_json_field "created_at" "results.json has timestamp"
check_json_field "summary" "results.json has summary"

# Check metric_results is non-empty (requires custom check)
log_test "results.json has metric_results"
((TESTS_RUN++))
if python3 -c "import json; data=json.load(open('$RUN_DIR/results.json')); exit(0 if data.get('metric_results') else 1)" 2>/dev/null; then
    log_pass "results.json has metric_results"
else
    log_fail "results.json missing metric_results"
fi

# Check metric results have expected fields
log_test "metric results have expected fields"
((TESTS_RUN++))
VALID_FIELDS=$(python3 << EOF
import json
try:
    data = json.load(open('$RUN_DIR/results.json'))
    results = data.get('metric_results', [])
    if not results:
        print("no_results")
    else:
        result = results[0]
        required = ['metric_id', 'score']
        if all(k in result for k in required):
            print("valid")
        else:
            print("missing_fields")
except Exception as e:
    print(f"error: {e}")
EOF
)

if [ "$VALID_FIELDS" = "valid" ]; then
    log_pass "Metric results have expected fields"
else
    log_fail "Metric results invalid: $VALID_FIELDS"
fi

# =============================================================================
# Test: report.html is valid
# =============================================================================

log_header "Validate: Report HTML"

log_test "report.html contains expected content"
((TESTS_RUN++))
if grep -q "output_nonempty\|Evaluation\|Results" "$RUN_DIR/report.html" 2>/dev/null; then
    log_pass "report.html contains expected content"
else
    log_fail "report.html missing expected content"
fi

# =============================================================================
# Test: CLI retrieval matches files
# =============================================================================

log_header "Validate: CLI Consistency"

# Get run via list-runs
log_test "list-runs shows recent run"
((TESTS_RUN++))
LIST_OUTPUT=$(run_evalyn list-runs --format json --limit 5 2>/dev/null)
if echo "$LIST_OUTPUT" | grep -q "validation-test\|$LATEST_RUN"; then
    log_pass "list-runs shows recent run"
else
    log_fail "list-runs does not show recent run"
fi

# Show-run works for latest
log_test "show-run --last works"
((TESTS_RUN++))
if run_evalyn show-run --last 2>/dev/null | grep -qE "output_nonempty|validation"; then
    log_pass "show-run --last shows results"
else
    log_fail "show-run --last failed"
fi

# =============================================================================
# Summary
# =============================================================================

print_summary "Results Validation"
exit $?
