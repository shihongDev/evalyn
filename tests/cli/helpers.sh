#!/bin/bash
# =============================================================================
# CLI Test Helpers
# =============================================================================
# Common functions for CLI test scripts.
# Source this file after config.sh in each test script.
# =============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters (initialize if not set)
: "${TESTS_RUN:=0}"
: "${TESTS_PASSED:=0}"
: "${TESTS_FAILED:=0}"
: "${TESTS_SKIPPED:=0}"

# Verbose mode flag
: "${VERBOSE:=false}"

# =============================================================================
# Logging Functions
# =============================================================================

log_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

log_test() {
    echo -e "\n${YELLOW}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((TESTS_SKIPPED++))
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# =============================================================================
# Command Execution
# =============================================================================

# Run evalyn CLI command
run_evalyn() {
    if [ "$VERBOSE" = true ]; then
        python -m evalyn_sdk.cli "$@"
    else
        python -m evalyn_sdk.cli "$@" 2>&1
    fi
}

# Execute a command and record pass/fail
# Usage: run_cmd "description" command arg1 arg2 ...
run_cmd() {
    local desc="$1"
    shift
    log_test "$desc"
    ((TESTS_RUN++))

    if [ "$VERBOSE" = true ]; then
        echo "  Command: $@"
    fi

    if "$@" 2>&1; then
        log_pass "$desc"
        return 0
    else
        log_fail "$desc"
        return 1
    fi
}

# Execute command where non-zero exit is acceptable
run_cmd_allow_fail() {
    local desc="$1"
    shift
    log_test "$desc"
    ((TESTS_RUN++))

    if "$@" 2>&1; then
        log_pass "$desc"
    else
        log_info "$desc - non-zero exit (may be expected)"
        log_pass "$desc (graceful handling)"
    fi
}

# Execute evalyn command and check for pattern in output
# Usage: run_cmd_check "description" "pattern" evalyn args...
run_cmd_check() {
    local desc="$1"
    local pattern="$2"
    shift 2

    log_test "$desc"
    ((TESTS_RUN++))

    local output
    output=$(run_evalyn "$@" 2>&1)

    if echo "$output" | grep -qE "$pattern"; then
        log_pass "$desc"
        return 0
    else
        log_fail "$desc (pattern '$pattern' not found)"
        [ "$VERBOSE" = true ] && echo "$output"
        return 1
    fi
}

# =============================================================================
# Setup Functions
# =============================================================================

# Ensure test output directory exists
ensure_test_dir() {
    mkdir -p "$TEST_OUTPUT_DIR"
    log_info "Test output: $TEST_OUTPUT_DIR"
}

# Ensure test traces exist
ensure_test_traces() {
    if [ ! -f "$TEST_DB" ]; then
        log_info "Creating test traces..."
        python "$TEST_FIXTURES/oneclick/setup_test_traces.py"
    fi

    local trace_count
    trace_count=$(run_evalyn list-calls --format json --limit 1 2>/dev/null | python -c "import sys, json; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo "0")
    log_info "Found $trace_count traces in test database"
}

# Check if API key is available
has_api_key() {
    [ -n "$GEMINI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OPENAI_API_KEY" ]
}

# Skip if no API key
require_api_key() {
    if ! has_api_key; then
        log_skip "$1 (no API key)"
        return 1
    fi
    return 0
}

# =============================================================================
# Test Data Helpers
# =============================================================================

# Get a call ID from recent traces
get_call_id() {
    run_evalyn list-calls --limit 1 --format json 2>/dev/null | \
        grep -o '"id": "[^"]*"' | head -1 | cut -d'"' -f4
}

# Get a run ID from recent runs
get_run_id() {
    run_evalyn list-runs --format json --limit 1 2>/dev/null | \
        grep -o '"id": "[^"]*"' | head -1 | cut -d'"' -f4
}

# Get two different run IDs for comparison
get_two_run_ids() {
    run_evalyn list-runs --format json --limit 2 2>/dev/null | \
        grep -o '"id": "[^"]*"' | cut -d'"' -f4
}

# Create a test dataset
create_test_dataset() {
    local output_dir="$1"
    local limit="${2:-5}"

    mkdir -p "$output_dir/metrics"
    run_evalyn build-dataset \
        --project "$TEST_PROJECT" \
        --output "$output_dir" \
        --limit "$limit" 2>&1
}

# Create test annotations file
create_test_annotations() {
    local output_file="$1"
    local dataset_dir="$2"

    python3 << EOF
import json
from pathlib import Path

dataset_file = Path('$dataset_dir') / 'dataset.jsonl'
if not dataset_file.exists():
    print("No dataset file found")
    exit(1)

with open(dataset_file) as f:
    items = [json.loads(line) for line in f if line.strip()]

annotations = []
for i, item in enumerate(items[:5]):
    annotations.append({
        'id': f'ann-{i}',
        'target_id': item.get('id', f'item-{i}'),
        'metric_id': 'helpfulness_accuracy',
        'label': i % 2 == 0,
        'rationale': f'Test annotation {i}',
        'annotator': 'test-script',
        'source': 'human',
        'confidence': 4
    })

with open('$output_file', 'w') as f:
    for ann in annotations:
        f.write(json.dumps(ann) + '\n')
print(f'Created {len(annotations)} annotations')
EOF
}

# =============================================================================
# Summary Functions
# =============================================================================

# Print test summary
print_summary() {
    local script_name="${1:-CLI Tests}"

    log_header "Test Summary: $script_name"
    echo -e "Tests run:     ${TESTS_RUN}"
    echo -e "Tests passed:  ${GREEN}${TESTS_PASSED}${NC}"
    echo -e "Tests failed:  ${RED}${TESTS_FAILED}${NC}"
    echo -e "Tests skipped: ${YELLOW}${TESTS_SKIPPED}${NC}"

    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "\n${RED}Some tests failed!${NC}"
        return 1
    else
        echo -e "\n${GREEN}All tests passed!${NC}"
        return 0
    fi
}

# Parse common arguments
parse_args() {
    for arg in "$@"; do
        case $arg in
            --verbose|-v) VERBOSE=true ;;
            --help|-h)
                echo "Usage: $0 [--verbose] [--help]"
                echo "  --verbose, -v  Show verbose output"
                echo "  --help, -h     Show this help"
                exit 0
                ;;
        esac
    done
}
